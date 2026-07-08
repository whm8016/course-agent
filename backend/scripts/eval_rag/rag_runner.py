"""RAG Runner —— 对 LightRAG 多检索模式的统一调用封装。

直接调用底层 retriever（不走 HTTP API），隔离评测变量：
  - contexts: 用 retrieve() 纯检索（only_need_context=True，0 LLM 调用）
  - answer:   用 query() 走 LightRAG 端到端生成（only_need_context=False，含内部 LLM）

关键设计：contexts 与 answer 来自两次独立调用，faithfulness 才有意义。
旧实现把 answer 当 context 回填（contexts 为空时 contexts=[answer]），导致 RAGAS
判定"回答中的陈述全都能在 context 找到"，faithfulness 恒为 1.0 —— 已废弃。

--production-parity 模式：contexts 走 retrieve_context()（naive, top_k=5），
精确对齐生产 tool_registry._execute_rag 的检索路径（生产也是这条路拿上下文喂 chat LLM）。

返回统一格式：
  {"answer": str, "contexts": list[str], "retrieve_ms": int, "query_ms": int}
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


# 生产路径对齐参数（tool_registry._execute_rag: retrieve_context 的默认）
_PROD_MODE = "naive"
# retrieve_context 用 "\n\n---\n\n" 连接各证据块（见 lightrag._format_contexts_for_prompt）
_CONTEXT_SEP = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _cache_path(qid: str, mode: str) -> Path:
    return config.CACHE_DIR / f"{qid}_{mode}.json"


def load_cache(qid: str, mode: str) -> dict | None:
    p = _cache_path(qid, mode)
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


def save_cache(qid: str, mode: str, data: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(qid, mode).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


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
    production_parity: bool = False,
) -> dict[str, Any]:
    """对单条问题查询，返回 {answer, contexts, retrieve_ms, query_ms}。

    contexts 与 answer 来自两次独立调用，保证 faithfulness 有效（不会 answer 自证）。
    """
    from core.rag import get_retriever

    retriever = get_retriever("lightrag")
    k = top_k if top_k is not None else config.EVAL_TOP_K

    contexts: list[str] = []
    answer = ""
    retrieve_ms = 0
    query_ms = 0

    # ---- 1. 取 contexts（纯检索，only_need_context=True，不产生 LLM 调用）----
    t0 = time.perf_counter()
    try:
        if production_parity:
            # 对齐生产 _execute_rag：retrieve_context(naive, top_k) → 拼接字符串
            ctx_str = await retriever.retrieve_context(
                course_id=course_id, query=question, top_k=k, mode=_PROD_MODE
            )
            contexts = _split_context_string(ctx_str)
        else:
            # 默认：retrieve() → list[RetrievalResult]，content 天然适配 RAGAS retrieved_contexts
            results = await retriever.retrieve(
                course_id=course_id, query=question, top_k=k, mode=mode
            )
            contexts = [r.content for r in results if getattr(r, "content", "")]
    except Exception as e:
        logger.error("[%s] retrieve contexts 失败: %s", mode, e)
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    # ---- 2. 取 answer（LightRAG 端到端生成，含内部 LLM）----
    t0 = time.perf_counter()
    try:
        qres = await retriever.query(course_id=course_id, message=question, mode=mode)
        if isinstance(qres, dict):
            answer = (
                qres.get("response")
                or qres.get("answer")
                or qres.get("content")
                or ""
            )
    except Exception as e:
        logger.error("[%s] query answer 失败: %s", mode, e)
    query_ms = int((time.perf_counter() - t0) * 1000)

    # 不再把 answer 当 context 回填（旧实现 faithfulness=1.0 根因）
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
                cached = load_cache(qid, mode)
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
            save_cache(qid, mode, result)
            mode_results.append(result)

            # 限流延迟
            await asyncio.sleep(config.QUERY_DELAY)

        all_results[mode] = mode_results
        logger.info("模式 %s 完成，%d 条结果", mode, len(mode_results))

    return all_results
