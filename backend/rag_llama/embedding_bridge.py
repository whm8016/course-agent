"""
embedding_bridge.py
───────────────────
替代 DeepTutor 的 get_embedding_client / get_embedding_config。

使用 core/llm.py 里已有的 AsyncOpenAI 客户端（指向 DashScope），
对外暴露与 DeepTutor EmbeddingClient 完全一致的接口：

    await get_embedding_client().embed(list[str])  ->  list[list[float]]

放置路径：rag_llama/embedding_bridge.py
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Any

# ── 复用项目已有的 OpenAI 客户端和配置 ────────────────────────────────────────
from core.llm import client as _async_openai_client   # AsyncOpenAI 实例
from config import EMBEDDING_MODEL                     # 例如 "text-embedding-v3"

logger = logging.getLogger("EmbeddingBridge")

# DashScope text-embedding-v3 / v2 默认 1024 维，可在 .env 里覆盖
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
# 每批最多发多少条文本（避免单次请求 token 超限）
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))


# ── 配置对象（对应 DeepTutor 的 EmbeddingConfig）─────────────────────────────
@dataclass
class EmbeddingConfig:
    model: str
    dim: int
    binding: str = "dashscope_openai_compat"


def get_embedding_config() -> EmbeddingConfig:
    """供 LlamaIndexPipeline._configure_settings() 打日志用。"""
    return EmbeddingConfig(model=EMBEDDING_MODEL, dim=EMBEDDING_DIM)


# ── Embedding 客户端（对应 DeepTutor 的 EmbeddingClient）─────────────────────
class DashScopeEmbeddingClient:
    """
    异步 embed()：接收字符串列表，返回向量列表。
    接口与 DeepTutor EmbeddingClient.embed 完全一致，CustomEmbedding 可以直接调用。
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


# ── 单例工厂（对应 DeepTutor 的 get_embedding_client()）──────────────────────
_client_instance: Optional[DashScopeEmbeddingClient] = None


def get_embedding_client() -> DashScopeEmbeddingClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = DashScopeEmbeddingClient()
    return _client_instance