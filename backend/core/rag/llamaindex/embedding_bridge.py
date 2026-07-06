"""
embedding_bridge.py
───────────────────
替代 的 get_embedding_client / get_embedding_config。

基于 EMBEDDING__* 凭证独立构造 OpenAI 兼容客户端（不复用主 LLM client——
主 LLM 可能是不提供 /embeddings 的 provider，如 deepseek），
对外暴露与 EmbeddingClient 完全一致的接口：

    await get_embedding_client().embed(list[str])  ->  list[list[float]]

放置路径：rag_llama/embedding_bridge.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Any

# ── Embedding 专用客户端（独立于主 LLM client）──────────────────────────────
# 旧实现 `from core.llm.llm import client` 复用主 LLM 的 OpenAI client。主 LLM
# 切到非 embedding provider（如 deepseek）后，用该 client 调 /embeddings 会 404。
# 这里基于 EMBEDDING__* 凭证独立构造，确保命中 DashScope 的 text-embedding 端点。
from settings import get_settings
from core.llm.provider_factory import get_llm_client

_emb = get_settings().embedding
EMBEDDING_MODEL = _emb.model
EMBEDDING_DIM = _emb.dim
EMBEDDING_BATCH_SIZE = _emb.batch_size
EMBEDDING_API_KEY = _emb.api_key.get_secret_value()
EMBEDDING_BASE_URL = _emb.base_url

_async_openai_client = get_llm_client(
    api_key=EMBEDDING_API_KEY,
    base_url=EMBEDDING_BASE_URL or None,
)

logger = logging.getLogger("EmbeddingBridge")


# ── 配置对象（对应 的 EmbeddingConfig）─────────────────────────────
@dataclass
class EmbeddingConfig:
    model: str
    dim: int
    binding: str = "dashscope_openai_compat"


def get_embedding_config() -> EmbeddingConfig:
    """供 LlamaIndexPipeline._configure_settings() 打日志用。"""
    return EmbeddingConfig(model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)


# ── Embedding 客户端（对应 的 EmbeddingClient）─────────────────────
class DashScopeEmbeddingClient:
    """
    异步 embed()：接收字符串列表，返回向量列表。
    接口与 EmbeddingClient.embed 完全一致，CustomEmbedding 可以直接调用。
    """

    async def embed(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], Any]] = None,
    ) -> List[List[float]]:
        if not texts:
            return []

        all_vecs: List[List[float]] = []
        total_batches = max(1, math.ceil(len(texts) / EMBEDDING_BATCH_SIZE))

        for batch_idx in range(total_batches):
            batch = texts[batch_idx * EMBEDDING_BATCH_SIZE : (batch_idx + 1) * EMBEDDING_BATCH_SIZE]

            logger.debug(
                "Embedding batch %d/%d, %d texts",
                batch_idx + 1, total_batches, len(batch),
            )

            resp = await _async_openai_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )

            # 按 index 排序，保证顺序与输入一致
            ordered = sorted(resp.data, key=lambda d: d.index)
            for item in ordered:
                all_vecs.append(list(item.embedding))

            if progress_callback is not None:
                try:
                    progress_callback(batch_idx + 1, total_batches)
                except Exception:
                    pass

        logger.debug("Embedding done: %d vectors, dim=%d", len(all_vecs), len(all_vecs[0]) if all_vecs else 0)
        return all_vecs


# ── 单例工厂（对应 的 get_embedding_client()）──────────────────────
_client_instance: Optional[DashScopeEmbeddingClient] = None


def get_embedding_client() -> DashScopeEmbeddingClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = DashScopeEmbeddingClient()
    return _client_instance