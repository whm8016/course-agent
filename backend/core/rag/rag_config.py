"""RAG 子系统统一配置管理。

将散布在 config.py 和 settings/base.py 中的 30+ 个 LIGHTRAG_* / CHUNK_* / EMBEDDING_* 常量
聚合为结构化 dataclass，各模块从此处导入，避免零散引用。

使用方式：
    from core.rag.rag_config import get_chunking_config, get_lightrag_config
    config = get_lightrag_config()
    print(config.top_k, config.query_mode)
"""
from __future__ import annotations

from dataclasses import dataclass


# 从 config shim 读取（config shim 读取自 settings，保持单一事实源）
from config import (
    # LightRAG 核心配置
    LIGHTRAG_ENABLED,
    LIGHTRAG_WORKDIR,
    LIGHTRAG_QUERY_MODE,
    LIGHTRAG_TOP_K,
    LIGHTRAG_TIMEOUT_SEC,
    LIGHTRAG_EMBEDDING_DIM,
    LIGHTRAG_AUTO_INDEX_TTL_SEC,
    LIGHTRAG_STREAM_CONTEXT_LIMIT,
    LIGHTRAG_STREAM_CONTEXT_MAX_CHARS,
    LIGHTRAG_AGENTIC_RAG_MAX_CHARS,
    LIGHTRAG_ENABLE_RERANK,
    LIGHTRAG_SAVE_INGEST_CHUNKS,
    LIGHTRAG_INGEST_CHUNKS_SUBDIR,
    LIGHTRAG_INGEST_CHUNKS_SNAPSHOT,
    LIGHTRAG_INGEST_BATCH_SIZE,
    LIGHTRAG_MAX_ASYNC,
    LIGHTRAG_LRU_CAPACITY,
    # LightRAG 安全阈值
    LIGHTRAG_SAFE_TOP_K,
    LIGHTRAG_CHUNK_TOP_K,
    LIGHTRAG_MAX_TOTAL_TOKENS,
    LIGHTRAG_MAX_ENTITY_TOKENS,
    LIGHTRAG_MAX_RELATION_TOKENS,
    LIGHTRAG_MAX_HISTORY_MESSAGES,
    LIGHTRAG_MAX_HISTORY_CHARS,
    LIGHTRAG_LLM_SYSTEM_MAX_CHARS,
    # Chunk 配置
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    INGEST_CHUNK_OVERLAP,
    # Embedding 配置
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """统一切块配置（唯一事实源）。"""
    size: int = 500
    overlap: int = 80
    # 用于 LightRAG ingestion
    ingest_size: int = 900
    ingest_overlap: int = 180


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding 配置。"""
    model: str = "text-embedding-v3"
    api_key: str = ""
    base_url: str = ""
    embedding_dim: int = 1024


@dataclass(frozen=True)
class LightRAGConfig:
    """LightRAG 后端配置（聚合所有 LIGHTRAG_* 常量）。"""
    enabled: bool = True
    workdir: str = "./lightrag_data"
    query_mode: str = "mix"
    top_k: int = 5
    timeout_sec: int = 60
    embedding_dim: int = 1024
    auto_index_ttl_sec: int = 0
    stream_context_limit: int = 5
    stream_context_max_chars: int = 2000
    agentic_rag_max_chars: int = 8000
    enable_rerank: bool = False

    # Ingestion 配置
    save_ingest_chunks: bool = False
    ingest_chunks_subdir: str = "chunks"
    ingest_chunks_snapshot: bool = False
    ingest_batch_size: int = 20
    max_async: int = 4
    lru_capacity: int = 4

    # 安全阈值（防 API 拒绝）
    safe_top_k: int = 10
    chunk_top_k: int = 10
    max_total_tokens: int = 26000
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 6000
    max_history_messages: int = 10
    max_history_chars: int = 4000
    llm_system_max_chars: int = 2000


# ── 工厂函数（读取当前 config 值）──────────────────────────────────────────────


def get_chunking_config() -> ChunkingConfig:
    """获取统一切块配置。"""
    return ChunkingConfig(
        size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        ingest_size=INGEST_CHUNK_SIZE,
        ingest_overlap=INGEST_CHUNK_OVERLAP,
    )


def get_embedding_config() -> EmbeddingConfig:
    """获取 Embedding 配置。"""
    return EmbeddingConfig(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
        embedding_dim=LIGHTRAG_EMBEDDING_DIM,
    )


def get_lightrag_config() -> LightRAGConfig:
    """获取 LightRAG 配置。"""
    return LightRAGConfig(
        enabled=LIGHTRAG_ENABLED,
        workdir=LIGHTRAG_WORKDIR,
        query_mode=LIGHTRAG_QUERY_MODE,
        top_k=LIGHTRAG_TOP_K,
        timeout_sec=LIGHTRAG_TIMEOUT_SEC,
        embedding_dim=LIGHTRAG_EMBEDDING_DIM,
        auto_index_ttl_sec=LIGHTRAG_AUTO_INDEX_TTL_SEC,
        stream_context_limit=LIGHTRAG_STREAM_CONTEXT_LIMIT,
        stream_context_max_chars=LIGHTRAG_STREAM_CONTEXT_MAX_CHARS,
        agentic_rag_max_chars=LIGHTRAG_AGENTIC_RAG_MAX_CHARS,
        enable_rerank=LIGHTRAG_ENABLE_RERANK,
        save_ingest_chunks=LIGHTRAG_SAVE_INGEST_CHUNKS,
        ingest_chunks_subdir=LIGHTRAG_INGEST_CHUNKS_SUBDIR,
        ingest_chunks_snapshot=LIGHTRAG_INGEST_CHUNKS_SNAPSHOT,
        ingest_batch_size=LIGHTRAG_INGEST_BATCH_SIZE,
        max_async=LIGHTRAG_MAX_ASYNC,
        lru_capacity=LIGHTRAG_LRU_CAPACITY,
        safe_top_k=LIGHTRAG_SAFE_TOP_K,
        chunk_top_k=LIGHTRAG_CHUNK_TOP_K,
        max_total_tokens=LIGHTRAG_MAX_TOTAL_TOKENS,
        max_entity_tokens=LIGHTRAG_MAX_ENTITY_TOKENS,
        max_relation_tokens=LIGHTRAG_MAX_RELATION_TOKENS,
        max_history_messages=LIGHTRAG_MAX_HISTORY_MESSAGES,
        max_history_chars=LIGHTRAG_MAX_HISTORY_CHARS,
        llm_system_max_chars=LIGHTRAG_LLM_SYSTEM_MAX_CHARS,
    )


# ── 计算后的安全值（与 lightrag_engine.py 内部逻辑一致）────────────────────────────


def get_safe_top_k() -> int:
    """计算安全 top_k（防 API 拒绝）。"""
    config = get_lightrag_config()
    return min(config.top_k, config.safe_top_k)


def get_safe_chunk_top_k() -> int:
    """计算安全 chunk_top_k。"""
    config = get_lightrag_config()
    return min(config.chunk_top_k, get_safe_top_k())