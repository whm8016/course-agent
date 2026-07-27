"""Config 驱动的混合检索流水线：BM25 + Dense 并行召回 → RRF/linear 融合 → rerank。

每个开关独立控制，方便消融实验（见 retrieval_config.ABLATION_CONFIGS）。ES 不可用
或未配置时，调用方应降级回纯 dense（LightRAG naive）——本模块不在 ES 缺失时兜底，
而是把 es_store=None / dense_search_fn=None 当作"该路关闭"处理。

dense_search_fn / rerank_fn 由调用方注入（适配 LightRAG 向量检索与 gte-rerank-v2）。
"""
from __future__ import annotations

import asyncio
import logging

from .retrieval_config import RetrievalConfig, reciprocal_rank_fusion, linear_fusion

logger = logging.getLogger(__name__)


async def retrieve(
    query: str,
    course_id: str,
    config: RetrievalConfig,
    *,
    es_store=None,
    dense_search_fn=None,
    rerank_fn=None,
) -> list[dict]:
    """Config 驱动的检索流水线。

    Args:
        query: 查询文本。
        course_id: 课程 ID（BM25 按此过滤）。
        config: 检索配置（各开关）。
        es_store: ESChunkStore 实例（bm25_enabled 时用）；None 则跳过 BM25 路。
        dense_search_fn: ``async (query, top_k) -> list[dict]``，每项含
            content/chunk_id/score；None 则跳过 dense 路。
        rerank_fn: 可选 ``async (query, docs: list[dict], top_n) -> list[dict]``；
            None 或 config 关闭则跳过精排。

    Returns:
        排序后的 dict 列表（含 content/chunk_id/score），已融合 + 可选精排。
        任一路召回失败不拖垮整条流水线，仅跳过该路。
    """
    tasks: dict[str, asyncio.Task] = {}
    if config.bm25_enabled and es_store is not None:
        tasks["bm25"] = asyncio.create_task(
            es_store.bm25_search(query, course_id, config.bm25_top_k)
        )
    if config.dense_enabled and dense_search_fn is not None:
        tasks["dense"] = asyncio.create_task(dense_search_fn(query, config.dense_top_k))

    if not tasks:
        return []

    gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
    results: dict[str, list[dict]] = {}
    for key, res in zip(tasks.keys(), gathered):
        if isinstance(res, Exception):
            logger.warning("retrieve 召回路 %s 失败，跳过该路: %s", key, res)
            continue
        results[key] = res or []

    ranked_lists: list[list[dict]] = []
    if "bm25" in results:
        ranked_lists.append(results["bm25"])
    if "dense" in results:
        ranked_lists.append(results["dense"])

    if not ranked_lists:
        return []

    # ---- 融合 ----
    if len(ranked_lists) == 1:
        fused = list(ranked_lists[0])
    elif config.fusion_method == "linear":
        fused = linear_fusion(*ranked_lists, alpha=config.linear_alpha)
    else:
        fused = reciprocal_rank_fusion(*ranked_lists, k=config.rrf_k)

    # ---- 精排 ----
    if config.rerank_enabled and rerank_fn and fused:
        try:
            reranked = await rerank_fn(query, fused, top_n=config.rerank_top_n)
            if reranked:
                fused = reranked
        except Exception as exc:
            logger.warning("rerank 失败，返回融合结果（未精排）: %s", exc)

    return fused


__all__ = ["retrieve"]
