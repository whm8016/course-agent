"""PGVectorStore 工厂 + embedding model + course 隔离（llamaindex_pg 后端存储层）。

向量存 Postgres（pgvector HNSW），进程不常驻——这是与旧 ``SimpleVectorStore`` 落 JSON
的核心区别（旧方案暴力扫描无 ANN、每个 worker 各持一份索引常驻内存、索引文件与 PG
数据分家，三点正是它当初被删的原因）。course_id 用 metadata filter 隔离，所有 KB 共享
一张表（``data_kb_chunks``），靠 ``indexed_metadata_keys`` 给 course_id 建索引避免全表扫。

检索绕开 PGVectorStore 自带的 hybrid 模式：它只把 dense/sparse 各取 top_k 后简单去重合并，
没有 RRF、没有重排、连 alpha 加权都被忽略（run-llama/llama_index Discussion #19606）。
改为 dense（DEFAULT 走 HNSW）与 sparse（SPARSE 走 tsvector 全文）各查一次，交项目
``hybrid_retriever`` 做 RRF 融合 + 可选 rerank。两路查同一张表，chunk_id（PG 行 node_id）
天然一致，无需像 LightRAG+ES 那样双写对齐。

embedding 用 LlamaIndex 官方 ``OpenAIEmbedding``，接 ``settings.embedding``（DashScope
text-embedding-v3，OpenAI 兼容）。项目此前无统一 embedding 工厂（LightRAG 直接用 SDK
自带 ``openai_embed``），本模块即是 llamaindex_pg 后端的 embedding 入口。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from settings import get_settings

logger = logging.getLogger(__name__)

# 所有 KB 共享一张向量表（PGVectorStore 自动加 data_ 前缀 → data_kb_chunks）。
# 靠 course_id metadata filter 隔离，避免每 KB 一表的 DDL 膨胀。表名常量供 indexer
# delete（按 course_id 清行）复用，避免与 PGVectorStore 内部命名脱节。
PG_TABLE_NAME = "kb_chunks"

# 进程内单例：建表 / 连接池 / embedding model 加载只付一次。web(gunicorn -wN) 与
# ARQ worker 各自进程隔离，各持一份（PGVectorStore 本身无进程间共享语义）。
_vector_store: Any = None
_embed_model: Any = None


def _pg_connection_strings() -> tuple[str, str]:
    """从 ``settings.db.url`` 派生 (同步连接串, 异步连接串)。

    PGVectorStore 的 ``perform_setup``（建表 / HNSW / tsvector）走同步 SQLAlchemy，
    查询走异步。同步 driver 用 ``postgresql://``（psycopg2，由
    llama-index-vector-stores-postgres 依赖带入），异步沿用项目已在用的
    ``postgresql+asyncpg://``。
    """
    url = get_settings().db.url.get_secret_value()
    sync = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return sync, url


class _OpenAICompatEmbedding(BaseEmbedding):
    """OpenAI 兼容 embedding（直接调 ``/embeddings``，绕过官方 enum 校验）。

    不用官方 ``OpenAIEmbedding``——后者 ``__init__`` 把 model 名校验进
    ``OpenAIEmbeddingModelType`` enum（仅含 OpenAI 官方模型），DashScope 的
    ``text-embedding-v3`` 会 ``ValueError``。本类直调 AsyncOpenAI，支持任意 OpenAI
    兼容 endpoint（DashScope / Azure / 本地 vLLM）。

    sync 版用独立事件循环跑 async：``VectorStoreIndex`` 的 embed 走 sync 路径
    （``_get_text_embeddings``），而 HTTP 调用是 async，需桥接；新循环避免与 worker
    运行中 loop 冲突（HEAD ``embedding_bridge`` 同款做法）。
    """

    _client: Any = PrivateAttr()
    _model: str = PrivateAttr()
    _batch_size: int = PrivateAttr()

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model: str,
        batch_size: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from openai import AsyncOpenAI  # noqa: PLC0415

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = max(1, batch_size)

    @classmethod
    def class_name(cls) -> str:
        return "openai_compat_embedding"

    def _run_sync(self, coro: Any) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = await self._client.embeddings.create(
                model=self._model, input=batch
            )
            # M-25：校验返回长度，provider 超限/截断会静默少返回，错位会污染向量库
            if len(resp.data) != len(batch):
                raise RuntimeError(
                    f"embedding 返回长度不匹配：期望 {len(batch)} 实际 {len(resp.data)}"
                )
            for d in sorted(resp.data, key=lambda x: x.index):
                out.append(list(d.embedding))
        return out

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._embed_batch([query]))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._embed_batch([text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_batch(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._run_sync(self._aget_query_embedding(query))

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._run_sync(self._aget_text_embedding(text))

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._run_sync(self._aget_text_embeddings(texts))


def get_embed_model() -> Any:
    """``_OpenAICompatEmbedding`` 单例，接 ``settings.embedding``。

    ``api_key`` / ``base_url`` 为空时 settings 已在 ``_apply_legacy_and_fallbacks``
    回退到 ``llm.*``。DashScope text-embedding-v3 返回固定 1024 维、不接受 dimensions
    参数，维度由 PGVectorStore.embed_dim 保证建表列宽一致。
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model

    emb = get_settings().embedding
    _embed_model = _OpenAICompatEmbedding(
        api_key=emb.api_key.get_secret_value(),
        base_url=emb.base_url or None,
        model=emb.model,
        batch_size=emb.batch_size,
    )
    logger.info("llamaindex_pg embed_model ready: model=%s dim=%d", emb.model, emb.dim)
    return _embed_model


def get_vector_store() -> Any:
    """``PGVectorStore`` 单例（``perform_setup=True`` 首次调用建 data_kb_chunks + HNSW）。

    ``checkfirst=True`` / ``IF NOT EXISTS`` 保证幂等；单 ARQ worker 首次建表 race 风险低。
    HNSW 参数 m=16 / ef_construction=64 来自 pgvector 实测（50M 向量以内无需专用向量库，
    也是零新增服务的唯一选项）。``text_search_config='simple'``：中英混排语料不做词干化
    （``'english'`` 会把中文当噪声丢掉）。
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    from llama_index.vector_stores.postgres import PGVectorStore

    sync_url, async_url = _pg_connection_strings()
    emb = get_settings().embedding
    _vector_store = PGVectorStore.from_params(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=PG_TABLE_NAME,
        embed_dim=emb.dim,
        hybrid_search=True,  # 建 text_search_tsv 列，SPARSE 模式才可用
        text_search_config="simple",  # 中英混排不做词干化
        use_jsonb=True,  # metadata_ 用 JSONB，filter 走 @> 更快
        perform_setup=True,  # 自动建表 + HNSW（幂等，schema 跟着 llama-index 版本走）
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
        # course_id 提升为带索引的独立列，metadata filter 等值查询走索引而非 JSONB 全扫。
        # PGType 取 "text"（plan 依据）；若该版本 PGType 要求 sqlalchemy 类型对象，
        # 真机验证时改为 sa.Text。
        indexed_metadata_keys={("course_id", "text")},
    )
    logger.info("llamaindex_pg vector_store ready: table=data_%s", PG_TABLE_NAME)
    return _vector_store


def course_filter(course_id: str) -> Any:
    """course 隔离的 metadata filter（MetadataFilter 新版 API，非旧 ExactMatchFilter）。"""
    from llama_index.core.vector_stores.types import (  # noqa: PLC0415
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    return MetadataFilters(
        filters=[
            MetadataFilter(key="course_id", value=course_id, operator=FilterOperator.EQ)
        ]
    )


def is_llamaindex_pg_available() -> tuple[bool, str]:
    """可用性探测：依赖装了 + embedding key 配了。

    建表 / 连接的真正验证推迟到首次 ``get_vector_store()``（懒），避免启动期探测误杀。
    """
    try:
        import llama_index.vector_stores.postgres  # noqa: F401
    except ImportError:
        return False, "llama-index-vector-stores-postgres 未安装"
    emb = get_settings().embedding
    if not emb.api_key.get_secret_value():
        return False, "embedding api_key 未配置"
    return True, ""


__all__ = [
    "PG_TABLE_NAME",
    "get_embed_model",
    "get_vector_store",
    "course_filter",
    "is_llamaindex_pg_available",
]
