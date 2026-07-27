"""RAG Runner —— 对齐生产自适应路由的评测调用封装。

直接调用底层 retriever（不走 HTTP API），复刻生产 tool_registry._execute_rag：
  - contexts: 走生产检索方法（fact→retrieve_context(naive) / relationship→graph_augmented_retrieve），
    only_need_context=True，0 次 LightRAG 内部 LLM。
  - answer:   走主对话 LLM（core.llm.chat_complete）基于 contexts 生成——与生产一致。
    不用 retriever.query()（LightRAG 内部 LLM 端到端生成，是生产不走的死路径，会测错 faithfulness）。

mode 参数实际是生产 strategy：fact（默认，纯向量）/ relationship（图谱邻域+naive 事实）。
contexts 与 answer 来源独立（检索 vs 主 LLM），faithfulness 才有意义（不会 answer 自证）。

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


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _cache_path(qid: str, mode: str, course_id: str) -> Path:
    # 按课程分目录隔离：不同 course_id（基线 vs 新切块索引）必须各存各的缓存，
    # 否则跨课程共用 → 对比评测时新索引会读到基线答案，结论失效。
    # _v2：answer 改用主 LLM（chat_complete）后，旧 {qid}_{mode}.json（LightRAG 内部
    # query() 生成的 answer）必须失效，否则读到错误 answer。
    safe_course = re.sub(r"[^A-Za-z0-9_-]", "_", course_id)
    return config.CACHE_DIR / safe_course / f"{qid}_{mode}_v2.json"


def load_cache(qid: str, mode: str, course_id: str) -> dict | None:
    p = _cache_path(qid, mode, course_id)
    if p.exists():
        try:
            data = json.loads(p.read_text("utf-8"))
            # 向后兼容：旧缓存无 latency 字段，补 0
            data.setdefault("retrieve_ms", 0)
            data.setdefault("query_ms", 0)
            return data
        except Exception:
            return None
    return None


def save_cache(qid: str, mode: str, course_id: str, data: dict) -> None:
    p = _cache_path(qid, mode, course_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------------------
# retrieve_context 拼接字符串 → list[str]（production-parity 路径用）
# ---------------------------------------------------------------------------
def _split_context_string(text: str) -> list[str]:
    """把 retrieve_context() 返回的拼接字符串按证据分隔符拆成 list[str]。

    retrieve_context 用 "\\n\\n---\\n\\n" 连接各证据块（每块带 [证据N｜来源:xxx] 头），
    拆分后保留块文本作为 retrieved_contexts；拆不出来就整体作为单条。
    """
    if not text or not text.strip():
        return []
    parts = [p.strip() for p in text.split(_CONTEXT_SEP) if p.strip()]
    return parts or [text.strip()]


# ---------------------------------------------------------------------------
# LightRAG 查询（统一入口：retrieve 取 contexts + query 取 answer）
# ---------------------------------------------------------------------------
async def _run_lightrag_query(
    course_id: str,
    question: str,
    mode: str,
    *,
    top_k: int | None = None,
    production_parity: bool = False,  # deprecated：contexts 现已恒走生产路径，保留仅为向后兼容
) -> dict[str, Any]:
    """对单条问题查询，返回 {answer, contexts, retrieve_ms, query_ms}。

    复刻生产自适应路由（tool_registry._execute_rag）：contexts 走生产检索方法，answer 走
    主对话 LLM（chat_complete）基于 contexts 生成——与生产一致。不再用 retriever.query()
    （LightRAG 内部 LLM 端到端生成，是生产不走的死路径，会测错 faithfulness）。

    mode 实际是生产 strategy：
      - "relationship" → graph_augmented_retrieve（图谱邻域 + naive 事实去重）
      - 其它（含 "fact"）→ retrieve_context(naive)（纯向量 chunk）

    contexts 与 answer 来源独立（检索 vs 主 LLM），faithfulness 才有意义（不会 answer 自证）。
    """
    del production_parity  # deprecated，不再分支
    from core.llm.llm import chat_complete
    from core.rag import get_retriever
    from settings import get_settings

    retriever = get_retriever("lightrag")
    k = top_k if top_k is not None else config.EVAL_TOP_K

    contexts: list[str] = []
    ctx_str = ""
    answer = ""
    retrieve_ms = 0
    query_ms = 0

    # ---- 1. 取 contexts（生产检索路径，only_need_context=True，0 次 LightRAG 内部 LLM）----
    t0 = time.perf_counter()
    try:
        if mode == "relationship":
            ctx_str = await retriever.graph_augmented_retrieve(
                course_id=course_id, query=question, top_k=k
            )
        else:  # fact（默认）
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
            model=get_settings().llm.text_model,
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
) -> dict[str, Any]:
    """对单条问题用指定模式查询，返回 {answer, contexts, retrieve_ms, query_ms}。"""
    return await _run_lightrag_query(
        course_id, question, mode, top_k=top_k, production_parity=production_parity
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
) -> dict[str, list[dict]]:
    """对全部问题和全部模式运行查询，返回 {mode: [results]}。

    每个 result：{answer, contexts, retrieve_ms, query_ms}
    """
    all_results: dict[str, list[dict]] = {}

    for mode in modes:
        logger.info("=== 模式: %s ===", mode)
        mode_results: list[dict] = []

        for idx, item in enumerate(qa_items):
            qid = item["id"]
            question = item["question"]

            # 检查缓存
            if not no_cache:
                cached = load_cache(qid, mode, course_id)
                if cached is not None:
                    logger.info("[%s] %s (%s) → 缓存命中", mode, qid, question[:30])
                    mode_results.append(cached)
                    continue

            # 执行查询
            logger.info("[%s] %s (%s) → 查询中...", mode, qid, question[:30])
            try:
                result = await run_single_query(
                    course_id, qid, question, mode,
                    top_k=top_k, production_parity=production_parity,
                )
            except Exception as e:
                logger.error("[%s] %s 查询失败: %s", mode, qid, e)
                result = {"answer": "", "contexts": [], "retrieve_ms": 0, "query_ms": 0}

            # 写入缓存
            save_cache(qid, mode, course_id, result)
            mode_results.append(result)

            # 限流延迟
            await asyncio.sleep(config.QUERY_DELAY)

        all_results[mode] = mode_results
        logger.info("模式 %s 完成，%d 条结果", mode, len(mode_results))

    return all_results
