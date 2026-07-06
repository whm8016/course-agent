"""LightRAG Rerank 适配器。

封装 LightRAG 内置的 ali_rerank（DashScope gte-rerank-v2），提供符合
LightRAG rerank_model_func 签名的可调用对象。

签名（来自 lightrag/api/lightrag_server.py server_rerank_func）：
    async def rerank_func(query, documents, top_n=None, extra_body=None)

只在 RERANK_API_KEY 存在时构建，缺 key 则返回 None，由调用方决定是否跳过。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from settings import get_settings

logger = logging.getLogger(__name__)

# DashScope rerank 专用（检索阶段，非索引）；默认复用 EMBEDDING__API_KEY（同属 DashScope）
RERANK_API_KEY = get_settings().embedding.api_key.get_secret_value()
RERANK_MODEL = os.getenv("LIGHTRAG_RERANK_MODEL", "gte-rerank-v2").strip() or "gte-rerank-v2"


def build_rerank_func() -> Any | None:
    """构建 DashScope ali_rerank 适配函数。

    读取：
        RERANK_API_KEY（settings.embedding.api_key，即 EMBEDDING__API_KEY）
        LIGHTRAG_RERANK_MODEL  可选 env，默认 gte-rerank-v2

    Returns:
        符合 LightRAG rerank_model_func 签名的协程函数，或 None（未配置）
    """
    if not RERANK_API_KEY:
        logger.debug("build_rerank_func: RERANK_API_KEY not set, rerank disabled")
        return None

    try:
        from lightrag.rerank import ali_rerank
    except ImportError:
        logger.warning("build_rerank_func: lightrag.rerank not available, rerank disabled")
        return None

    api_key = RERANK_API_KEY
    model = RERANK_MODEL

    async def _rerank_func(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        extra_body: dict | None = None,
    ) -> list[dict]:
        return await ali_rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            api_key=api_key,
            model=model,
            extra_body=extra_body,
        )

    logger.info("build_rerank_func: DashScope reranker ready model=%s", model)
    return _rerank_func
