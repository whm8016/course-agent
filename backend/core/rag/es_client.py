"""Elasticsearch BM25 chunk 索引管理（ik_smart 中文分词）。

ES 只负责 BM25 关键词检索（ES 最擅长的事）；向量检索继续走 LightRAG chunks_vdb。
两路结果在 hybrid_retriever 用 RRF 融合，chunk_id 是两系统的 join key。

依赖 elasticsearch[async]。未安装 / 未连通时：_ensure() 返回 False，index_chunks /
bm25_search 安全降级（返回 False / []），调用方（hybrid_retriever）据此跳过 BM25 路，
整条流水线退化为纯 dense，不报错。
"""
from __future__ import annotations

import logging
from typing import Any

from settings import get_settings

logger = logging.getLogger(__name__)

_INDEX_SETTINGS: dict[str, Any] = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                    "filter": ["lowercase"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ik_smart_analyzer",
                "search_analyzer": "ik_smart_analyzer",
            },
            "chunk_id": {"type": "keyword"},
            "course_id": {"type": "keyword"},
            "file_path": {"type": "keyword"},
        }
    },
}


class ESChunkStore:
    """Elasticsearch BM25 chunk 索引管理（惰性建连，失败即降级）。"""

    def __init__(self, es_url: str, index_name: str = "rag_chunks"):
        self._es_url = es_url
        self._index_name = index_name
        self._client = None
        self._ready = False

    async def _ensure(self) -> bool:
        """惰性建 client + 索引；任一步失败置 _ready=False 让调用方降级。"""
        if self._ready:
            return True
        try:
            from elasticsearch import AsyncElasticsearch  # noqa: F401
        except ImportError:
            logger.info("elasticsearch 未安装，ES BM25 路径不可用（将降级纯 dense）")
            return False
        try:
            from elasticsearch import AsyncElasticsearch

            self._client = AsyncElasticsearch(self._es_url)
            if not await self._client.indices.exists(index=self._index_name):
                await self._client.indices.create(
                    index=self._index_name, body=_INDEX_SETTINGS
                )
            self._ready = True
            return True
        except Exception as exc:
            logger.warning("ES 连接/建索引失败（BM25 将降级）: %s", exc)
            await self._close()
            return False

    async def _close(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        self._ready = False

    async def index_chunks(self, chunks: list[dict], course_id: str) -> bool:
        """批量写入 chunks。chunks 每项含 chunk_id/content[/file_path]。失败返回 False。"""
        if not await self._ensure():
            return False
        from elasticsearch.helpers import async_bulk

        actions = [
            {
                "_index": self._index_name,
                "_id": c["chunk_id"],
                "_source": {
                    "content": c.get("content", ""),
                    "chunk_id": c["chunk_id"],
                    "course_id": course_id,
                    "file_path": c.get("file_path", ""),
                },
            }
            for c in chunks
            if c.get("chunk_id")
        ]
        if not actions:
            return True
        try:
            await async_bulk(self._client, actions)
            return True
        except Exception as exc:
            logger.warning("ES index_chunks 失败 course=%s: %s", course_id, exc)
            return False

    async def bm25_search(self, query: str, course_id: str, top_k: int = 50) -> list[dict]:
        """BM25 关键词检索，返回 [{chunk_id, content, score, file_path}, ...]。"""
        if not await self._ensure():
            return []
        try:
            resp = await self._client.search(
                index=self._index_name,
                query={
                    "bool": {
                        "must": {"match": {"content": query}},
                        "filter": {"term": {"course_id": course_id}},
                    }
                },
                size=top_k,
            )
            return [
                {
                    "chunk_id": hit["_id"],
                    "content": hit["_source"].get("content", ""),
                    "score": float(hit.get("_score") or 0.0),
                    "file_path": hit["_source"].get("file_path", ""),
                }
                for hit in resp["hits"]["hits"]
            ]
        except Exception as exc:
            logger.warning("ES bm25_search 失败 course=%s: %s", course_id, exc)
            return []


_store: "ESChunkStore | None" = None


def get_es_store() -> "ESChunkStore | None":
    """全局 ES store 单例；未启用（settings.elasticsearch.enabled=False）返回 None。"""
    global _store
    cfg = get_settings().elasticsearch
    if not cfg.enabled:
        return None
    if _store is None:
        _store = ESChunkStore(es_url=cfg.url, index_name=cfg.index_name)
    return _store


__all__ = ["ESChunkStore", "get_es_store"]
