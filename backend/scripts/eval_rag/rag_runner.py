"""RAG Runner —— 对齐生产自适应路由的评测调用封装。

直接调用底层 retriever（不走 HTTP API），复刻生产 tool_registry._execute_rag：
  - contexts: 走生产检索方法（fact→retrieve_context(naive) / relationship→graph_augmented_retrieve），
    only_need_context=True，0 次 LightRAG 内部 LLM。
  - answer:   走主对话 LLM（core.llm.chat_complete）基于 contexts 生成——与生产一致。
    不用 retriever.query()（LightRAG 内部 LLM 端到端生成，是生产不走的死路径，会测错 faithfulness）。

mode 参数实际是生产 strategy：fact（默认，纯向量）/ relationship（图谱邻域+naive 事实）。
contexts 与 answer 来源独立（检索 vs 主 LLM），faithfulness 才有意义（不会 answer 自证）。

对照臂（表A 四臂评测，回答模型统一锁 EVAL_ANSWER_MODEL）：
  - no_rag      闭卷直答：不给任何片段，衡量模型自带知识天花板
  - naive_rag   朴素 RAG：固定字数切块索引 + 纯向量检索（course_id 须预先按
                CHUNKING__STRATEGY=sentence_splitter 重索引到独立 course，如 mycourse_naive600）
  - oracle_rag  上界臂（DeR² 受控输入）：把题目 gold 对应的源片段直接喂回答模型，
                区分「没找到」与「找到却没用好」。gold 片段由 ingest 审计 JSON 按题号映射。
fact/relationship 与上述三臂共存，同一 run_all_modes 可混用。

返回统一格式：
  {"answer": str, "contexts": list[str], "retrieve_ms": int, "query_ms": int}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


# retrieve_context / graph_augmented_retrieve 用 "\n\n---\n\n" 连接各证据块
# （见 lightrag._format_contexts_for_prompt），两路径格式一致 → _split_context_string 通用。
_CONTEXT_SEP = "\n\n---\n\n"

# answer 生成 system prompt：简化 RAG + 防幻觉（对齐 chat.yaml「资料不足就说明，不要编造」）。
# 评测复刻生产：主 LLM 拿 contexts 自己生成，不用 LightRAG 内部 query()（生产死路径）。
_RAG_SYSTEM = (
    "你是课程助教。只能根据下面提供的参考资料回答学生的课程问题；"
    "如果资料不足以回答，请直接说明“根据现有资料暂无法确认”，不要编造。"
    "引用资料中的原始术语、编号与接法表述时，同一概念前后表述必须一致，不要自相矛盾。"
)

# 闭卷臂 system prompt：无参考资料可用，但保持与生产一致的防编造约束——
# 课程专有事实（器件量程/成绩构成等）不在预训练记忆里时必须承认不知道，不许硬答。
_NO_RAG_SYSTEM = (
    "你是课程助教。请凭你已有的知识回答学生的课程问题；"
    "如果该问题涉及课程专有信息（具体器材型号与量程、本校考核方式、讲义原文数据等）"
    "而你不确定，请直接说明“根据现有资料暂无法确认”，不要编造。"
)


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _rerank_enabled() -> bool:
    """精排开关——与 build_rerank_fn 的闸门同一真相源（settings.rerank.enabled）。

    缓存键的精排状态必须与「检索时精排是否真发生」同源：那由 build_rerank_fn 读
    settings 快照决定，这里也读同一快照，而不是平行解析 RERANK__ENABLED 环境变量
    （词表/时序分叉会导致开精排却命中无精排旧缓存）。lazy import 与本文件对
    core.llm/core.rag 的既有 lazy import 同款。
    """
    from settings import get_settings

    return bool(get_settings().rerank.enabled)


# 不走检索路由、直接生成 answer 的对照臂：缓存键不贴精排后缀（answer 与精排无关），
# 与 _run_lightrag_query 的分支共用一份名单，防两处清单漂移
_ANSWER_ONLY_MODES = ("no_rag", "oracle_rag")


def _cache_path(qid: str, mode: str, course_id: str, top_k: int | None = None) -> Path:
    # 按课程分目录隔离：不同 course_id（基线 vs 新切块索引）必须各存各的缓存，
    # 否则跨课程共用 → 对比评测时新索引会读到基线答案，结论失效。
    # _v2：answer 改用主 LLM（chat_complete）后，旧 {qid}_{mode}.json（LightRAG 内部
    # query() 生成的 answer）必须失效，否则读到错误 answer。
    # _v2k{top_k}：带检索的 mode 缓存键追加 top_k——检索深度敏感性扫描（k=3/5/10/20）
    # 时不同 k 的 contexts/answer 必须各存各的，否则后跑的 k 覆盖先跑的缓存（k=10 读到
    # k=5 的答案，深度扫描结论失效）。k 为默认值（None→EVAL_TOP_K=5）时保持旧文件名
    # 不带后缀，向后兼容已有缓存。
    # _v2_rr：精排状态进缓存键（RERANK__ENABLED=true 时 fact 等检索臂的检索序不同，
    # 不得复用无精排旧答案）。
    safe_course = re.sub(r"[^A-Za-z0-9_-]", "_", course_id)
    suffix = "" if top_k in (None, config.EVAL_TOP_K) else f"_k{top_k}"
    if mode not in _ANSWER_ONLY_MODES and _rerank_enabled():
        suffix += "_rr"
    return config.CACHE_DIR / safe_course / f"{qid}_{mode}_v2{suffix}.json"


def load_cache(qid: str, mode: str, course_id: str, top_k: int | None = None) -> dict | None:
    p = _cache_path(qid, mode, course_id, top_k)
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
            # 向后兼容：旧缓存无 latency 字段，补 0
            data.setdefault("retrieve_ms", 0)
            data.setdefault("query_ms", 0)
            # 归一 contexts 形状：pg 线旧缓存是「整串单条」（拆分逻辑修复前落盘），
            # join 后重拆一遍两形状通吃——已逐条的不变，blob 的拆成逐块，消费者
            # （命中率按 rank 计 / RAGChecker/RAGAS 逐条粒度）拿到统一形状
            data["contexts"] = _split_context_string(
                _CONTEXT_SEP.join(data.get("contexts") or [])
            )
            return data
        except Exception:
            return None
    return None


def save_cache(
    qid: str, mode: str, course_id: str, data: dict, top_k: int | None = None
) -> None:
    p = _cache_path(qid, mode, course_id, top_k)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------------------
# retrieve_context 拼接字符串 → list[str]（production-parity 路径用）
# ---------------------------------------------------------------------------
def _split_context_string(text: str) -> list[str]:
    """把 retrieve_context() 返回的拼接字符串拆成逐块 list[str]。

    两条生产检索线的拼接协议不同：lightrag 用 ``\\n\\n---\\n\\n`` 连接各证据块，
    llamaindex_pg 用 ``\\n\\n``（llamaindex_pg._format_contexts）——只按 ``---`` 拆
    对 pg 线拆不开，整串退成单条 context（rank 语义全失，实测 68/68 个 fact 缓存均
    如此）。两条线共享的证据头 ``[证据N｜来源: ...]`` 是稳定契约，故两道拆分都用：
    先按 ``---`` 拆，每段再按 ``[证据N`` 头拆。已逐条的（每条至多一个头）拆完不变。
    """
    if not text or not text.strip():
        return []
    out: list[str] = []
    for part in text.split(_CONTEXT_SEP):
        out.extend(
            p.strip() for p in re.split(r"(?=\[证据\d+)", part) if p.strip()
        )
    return out or [text.strip()]


# ---------------------------------------------------------------------------
# oracle 臂：题目 gold → 源片段映射（借 DeR² 受控输入思路，零额外标注成本）
# ---------------------------------------------------------------------------
def load_oracle_contexts() -> dict[str, str]:
    """读 oracle_gold.json（{qid: 片段文本}），不存在返回空 dict。

    该文件由 make_oracle_gold.py 离线生成：对每道题的 ground_truth，在摄入切块审计
    JSON（data/ingest_chunks/course_<id>/latest.json）的全部 chunk 里找包含 gold 关键
    覆盖串的最短片段作「该题的源片段」。生成一次后零 LLM 成本。
    """
    p = config.EVAL_DIR / "oracle_gold.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


# ---------------------------------------------------------------------------
# LightRAG 查询（统一入口：retrieve 取 contexts + query 取 answer）
# ---------------------------------------------------------------------------
async def _answer_only(
    tag: str, system: str, user_msg: str, contexts: list[str]
) -> dict[str, Any]:
    """对照臂公共路径：不走检索，主 LLM 直接生成 answer（no_rag/oracle_rag 同款）。"""
    from core.llm.llm import chat_complete

    t0 = time.perf_counter()
    try:
        answer = await chat_complete(
            system, [], user_msg,
            model=config.EVAL_ANSWER_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as e:
        logger.error("[%s] generate answer 失败: %s", tag, e)
        answer = ""
    return {
        "answer": answer, "contexts": contexts,
        "retrieve_ms": 0, "query_ms": int((time.perf_counter() - t0) * 1000),
    }

async def _run_lightrag_query(
    course_id: str,
    question: str,
    mode: str,
    *,
    top_k: int | None = None,
    production_parity: bool = False,  # deprecated：contexts 现已恒走生产路径，保留仅为向后兼容
    retriever_backend: str = "lightrag",  # "lightrag"(默认) | "llamaindex_pg"(pgvector 线)
    question_id: str = "",  # oracle 臂按题号取 gold 片段
) -> dict[str, Any]:
    """对单条问题查询，返回 {answer, contexts, retrieve_ms, query_ms}。

    复刻生产自适应路由（tool_registry._execute_rag）：contexts 走生产检索方法，answer 走
    主对话 LLM（chat_complete）基于 contexts 生成——与生产一致。不再用 retriever.query()
    （LightRAG 内部 LLM 端到端生成，是生产不走的死路径，会测错 faithfulness）。

    mode 实际是生产 strategy：
      - "relationship" → graph_augmented_retrieve（图谱邻域 + naive 事实去重）
      - 其它（含 "fact"）→ retrieve_context(naive)（纯向量 chunk）

    表A 对照臂（与 strategy 正交，不触发检索路由）：
      - no_rag     闭卷直答（无 contexts）
      - oracle_rag 直接喂 gold 片段（不检索）

    contexts 与 answer 来源独立（检索 vs 主 LLM），faithfulness 才有意义（不会 answer 自证）。
    """
    del production_parity  # deprecated，不再分支
    from core.llm.llm import chat_complete

    # ---- 表A 对照臂：不走检索路由，只生成 answer ----
    if mode == "no_rag":
        return await _answer_only("no_rag", _NO_RAG_SYSTEM, question, [])

    if mode == "oracle_rag":
        ctx_str = load_oracle_contexts().get(question_id, "")
        if not ctx_str:
            logger.warning("[oracle_rag] %s 无 gold 片段映射（oracle_gold.json 缺该题）", question_id)
        user_msg = f"【参考资料】\n{ctx_str}\n\n【问题】\n{question}"
        return await _answer_only(
            "oracle_rag", _RAG_SYSTEM, user_msg, [ctx_str] if ctx_str else []
        )

    from core.rag import get_retriever

    retriever = get_retriever(retriever_backend)
    k = top_k if top_k is not None else config.EVAL_TOP_K

    contexts: list[str] = []
    ctx_str = ""
    answer = ""
    retrieve_ms = 0
    query_ms = 0

    # ---- 1. 取 contexts（生产检索路径，only_need_context=True，0 次 LightRAG 内部 LLM）----
    t0 = time.perf_counter()
    try:
        if mode == "relationship" and retriever_backend == "lightrag":
            ctx_str = await retriever.graph_augmented_retrieve(
                course_id=course_id, query=question, top_k=k
            )
        else:
            # fact（默认）；或 pgvector 等无图谱后端的 relationship 降级为纯向量证据
            if mode == "relationship":
                logger.warning(
                    "backend=%s 无 graph_augmented_retrieve，relationship 降级 retrieve_context(naive)",
                    retriever_backend,
                )
            ctx_str = await retriever.retrieve_context(
                course_id=course_id, query=question, top_k=k, mode="naive"
            )
        contexts = _split_context_string(ctx_str)
    except Exception as e:
        logger.error("[%s] retrieve contexts 失败: %s", mode, e)
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    # ---- 2. 取 answer（主对话 LLM 基于 contexts 生成，对齐生产）----
    # 生产里 retrieve_context/graph_augmented_retrieve 返回的 contexts 也是这样喂主 LLM 的。
    t0 = time.perf_counter()
    try:
        user_msg = f"【参考资料】\n{ctx_str}\n\n【问题】\n{question}"
        answer = await chat_complete(
            _RAG_SYSTEM, [], user_msg,
            model=config.EVAL_ANSWER_MODEL,  # 回答锁定 flash（用户要求 回答flash/阅卷pro）
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as e:
        logger.error("[%s] generate answer 失败: %s", mode, e)
    query_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "answer": answer,
        "contexts": contexts,
        "retrieve_ms": retrieve_ms,
        "query_ms": query_ms,
    }


# ---------------------------------------------------------------------------
# 单条查询
# ---------------------------------------------------------------------------
async def run_single_query(
    course_id: str,
    question_id: str,
    question: str,
    mode: str,
    *,
    top_k: int | None = None,
    production_parity: bool = False,
    retriever_backend: str = "lightrag",
) -> dict[str, Any]:
    """对单条问题用指定模式查询，返回 {answer, contexts, retrieve_ms, query_ms}。"""
    return await _run_lightrag_query(
        course_id, question, mode, top_k=top_k,
        production_parity=production_parity, retriever_backend=retriever_backend,
        question_id=question_id,
    )


# ---------------------------------------------------------------------------
# 批量查询（含缓存 + 限流）
# ---------------------------------------------------------------------------
async def run_all_modes(
    course_id: str,
    qa_items: list[dict],
    modes: list[str],
    *,
    no_cache: bool = False,
    top_k: int | None = None,
    production_parity: bool = False,
    retriever_backend: str = "lightrag",
    course_by_mode: dict[str, str] | None = None,
) -> dict[str, list[dict]]:
    """对全部问题和全部模式运行查询，返回 {mode: [results]}。

    每个 result：{answer, contexts, retrieve_ms, query_ms}
    course_by_mode：{mode: course_id}，允许某 mode 走另一个 course 的索引
    （naive_rag 臂指向固定字数切块的独立 course）；缺省用 course_id。
    """
    all_results: dict[str, list[dict]] = {}

    course_by_mode = course_by_mode or {}
    for mode in modes:
        mode_course = course_by_mode.get(mode, course_id)
        logger.info("=== 模式: %s (course=%s) ===", mode, mode_course)
        mode_results: list[dict] = []

        for idx, item in enumerate(qa_items):
            qid = item["id"]
            question = item["question"]

            # 检查缓存（带检索的 mode 缓存键含 top_k，深度扫描各 k 互不污染）
            if not no_cache:
                cached = load_cache(qid, mode, mode_course, top_k=top_k)
                if cached is not None:
                    logger.info("[%s] %s (%s) → 缓存命中", mode, qid, question[:30])
                    mode_results.append(cached)
                    continue

            # 执行查询
            logger.info("[%s] %s (%s) → 查询中...", mode, qid, question[:30])
            try:
                result = await run_single_query(
                    mode_course, qid, question, mode,
                    top_k=top_k, production_parity=production_parity,
                    retriever_backend=retriever_backend,
                )
            except Exception as e:
                logger.error("[%s] %s 查询失败: %s", mode, qid, e)
                result = {"answer": "", "contexts": [], "retrieve_ms": 0, "query_ms": 0}

            # 写入缓存
            save_cache(qid, mode, mode_course, result, top_k=top_k)
            mode_results.append(result)

            # 限流延迟
            await asyncio.sleep(config.QUERY_DELAY)

        all_results[mode] = mode_results
        logger.info("模式 %s 完成，%d 条结果", mode, len(mode_results))

    return all_results
