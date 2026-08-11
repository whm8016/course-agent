"""消融实验 runner：遍历 ABLATION_CONFIGS，每个 RetrievalConfig 跑 hybrid retrieve。

对比不同召回路径(BM25/Dense)/融合方式(RRF/linear)/精排(rerank) 组合的检索召回质量。
只算 context_precision/context_recall（纯检索指标，无需生成 answer，省 LLM 成本）——
消融要隔离"检索"这一变量，answer 留空，故跳过依赖 answer 的 faithfulness/answer_relevancy。

dense_search_fn 复用 LightRAGRetriever.dense_search（naive 向量检索），bm25 路走 ES；
ES 未启用时 get_es_store()=None，bm25 配置自动退化为纯 dense（hybrid_retrieve 内部跳过）。

返回 {config_label: [results]}，每个 result：{answer:"", contexts, retrieve_ms, query_ms:0}
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import partial

from . import config

logger = logging.getLogger(__name__)


async def run_ablation(
    course_id: str,
    qa_items: list[dict],
    configs: dict,
    *,
    top_k: int | None = None,
) -> dict[str, list[dict]]:
    """对每个 RetrievalConfig 跑 hybrid retrieve，返回 {config_label: [results]}。

    Args:
        course_id: 课程 ID（BM25 按 course_id 过滤）。
        qa_items: 评测集，每项含 id/question。
        configs: retrieval_config.ABLATION_CONFIGS（或子集）。
        top_k: 取前 K 条 context；None → config.EVAL_TOP_K。
    """
    from core.rag.hybrid_retriever import retrieve as hybrid_retrieve
    from core.rag.es_client import get_es_store
    from core.rag.rerank import build_rerank_fn
    from core.rag import get_retriever

    retriever = get_retriever("lightrag")
    es_store = get_es_store()
    if es_store is None:
        logger.warning(
            "ES 未启用（settings.elasticsearch.enabled=false），BM25 路将退化为纯 dense，"
            "bm25_only / bm25+rerank 配置会返回空结果"
        )
    k = top_k if top_k is not None else config.EVAL_TOP_K
    # partial 把 course_id 绑定为 dense_search 第一参，剩 (query, k) 适配 hybrid_retrieve 签名
    dense_fn = partial(retriever.dense_search, course_id)

    # 精排：force=True 无视生产 RERANK__ENABLED 开关，只看 api_key——否则默认关时
    # hybrid_rrf+rerank 与 hybrid_no_rerank 两组跑出来一样，消融无意义。是否真正调用
    # 由各 cfg.rerank_enabled 决定（hybrid_retriever 逐配置 gate）。
    rerank_fn = build_rerank_fn(force=True)

    all_results: dict[str, list[dict]] = {}
    for cfg_name, cfg in configs.items():
        label = cfg.label()
        logger.info(
            "=== 消融配置: %s (bm25=%s dense=%s fusion=%s rerank=%s) ===",
            label, cfg.bm25_enabled, cfg.dense_enabled, cfg.fusion_method, cfg.rerank_enabled,
        )
        results: list[dict] = []
        for item in qa_items:
            qid = item["id"]
            question = item["question"]
            t0 = time.perf_counter()
            try:
                docs = await hybrid_retrieve(
                    question, course_id, cfg,
                    es_store=es_store,
                    dense_search_fn=dense_fn,
                    rerank_fn=rerank_fn,
                )
                contexts = [d.get("content", "") for d in (docs or [])[:k] if d.get("content")]
            except Exception as e:
                logger.error("[%s] %s retrieve 失败: %s", label, qid, e)
                contexts = []
            retrieve_ms = int((time.perf_counter() - t0) * 1000)
            results.append({
                "answer": "",
                "contexts": contexts,
                "retrieve_ms": retrieve_ms,
                "query_ms": 0,
            })
            logger.info("[%s] %s (%s) → contexts=%d", label, qid, question[:30], len(contexts))
            await asyncio.sleep(config.QUERY_DELAY)
        all_results[label] = results
        logger.info("配置 %s 完成，%d 条结果", label, len(results))

    return all_results


__all__ = ["run_ablation"]
