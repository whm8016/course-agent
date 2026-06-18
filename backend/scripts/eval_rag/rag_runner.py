"""RAG Runner —— 对 4 种 LightRAG 检索模式的统一调用封装。

直接调用底层函数（不走 HTTP API），隔离评测变量：
  - naive  → query_with_lightrag(mode="naive")  朴素向量检索（baseline）
  - local  → query_with_lightrag(mode="local")  实体级局部检索
  - global → query_with_lightrag(mode="global") 关系级全局检索
  - mix    → query_with_lightrag(mode="mix")    local+global 混合检索

返回统一格式：{"answer": str, "contexts": list[str]}
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def _cache_path(qid: str, mode: str) -> Path:
    return config.CACHE_DIR / f"{qid}_{mode}.json"


def load_cache(qid: str, mode: str) -> dict | None:
    p = _cache_path(qid, mode)
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None
    return None


def save_cache(qid: str, mode: str, data: dict) -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(qid, mode).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------------------
# LightRAG 查询（统一入口）
# ---------------------------------------------------------------------------
async def _run_lightrag_query(
    course_id: str, question: str, mode: str
) -> dict[str, Any]:
    """LightRAG 查询（naive / local / global / mix）。"""
    from core.rag.lightrag_engine import query_with_lightrag

    result = await query_with_lightrag(course_id, question, mode=mode)

    # 提取 answer
    if isinstance(result, dict):
        answer = (
            result.get("response")
            or result.get("answer")
            or result.get("content")
            or ""
        )
        raw_contexts = result.get("contexts", [])
    else:
        answer = str(result)
        raw_contexts = []

    # 提取 contexts 文本
    contexts: list[str] = []
    for ctx in raw_contexts:
        text = _extract_context_text(ctx)
        if text:
            contexts.append(text)

    # 如果 contexts 为空但 answer 非空，将 answer 作为单一 context
    if not contexts and answer:
        contexts = [answer]

    return {"answer": answer, "contexts": contexts}


def _extract_context_text(ctx: Any) -> str:
    """从 LightRAG 返回的 context 中提取文本。"""
    if isinstance(ctx, str):
        return ctx.strip()
    if isinstance(ctx, dict):
        for key in ("content", "text", "chunk", "passage"):
            value = ctx.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(ctx).strip() if ctx else ""


# ---------------------------------------------------------------------------
# 单条查询
# ---------------------------------------------------------------------------
async def run_single_query(
    course_id: str, question_id: str, question: str, mode: str
) -> dict[str, Any]:
    """对单条问题用指定模式查询，返回 {"answer": str, "contexts": list[str]}。"""
    return await _run_lightrag_query(course_id, question, mode)


# ---------------------------------------------------------------------------
# 批量查询（含缓存 + 限流）
# ---------------------------------------------------------------------------
async def run_all_modes(
    course_id: str,
    qa_items: list[dict],
    modes: list[str],
    *,
    no_cache: bool = False,
) -> dict[str, list[dict]]:
    """对全部问题和全部模式运行查询，返回 {mode: [results]}。

    每个 result 格式：{"answer": str, "contexts": list[str]}
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
                result = await run_single_query(course_id, qid, question, mode)
            except Exception as e:
                logger.error("[%s] %s 查询失败: %s", mode, qid, e)
                result = {"answer": "", "contexts": []}

            # 写入缓存
            save_cache(qid, mode, result)
            mode_results.append(result)

            # 限流延迟
            await asyncio.sleep(config.QUERY_DELAY)

        all_results[mode] = mode_results
        logger.info("模式 %s 完成，%d 条结果", mode, len(mode_results))

    return all_results
