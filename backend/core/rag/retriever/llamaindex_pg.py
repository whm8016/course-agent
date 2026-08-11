"""llamaindex_pg 检索器：dense(DEFAULT) + sparse(SPARSE) → hybrid_retriever RRF 融合。

绕开 PGVectorStore 自带的 hybrid 模式（只去重合并、无 RRF、alpha 被忽略），改为两路
分别查同一张 PG 表：dense 走 HNSW（DEFAULT + query_embedding），sparse 走 tsvector
全文（SPARSE + query_str）。两路的 chunk_id 都来自 PG 行的 node_id，天然一致，RRF 可直接
join——比 LightRAG+ES 的双系统对齐（双写 md5(content)）更简单。

两路结果交项目 ``hybrid_retriever.retrieve`` 做 RRF 融合（``retrieval_config`` 的算法），
course_id 用 metadata filter 隔离。rerank 通过 ``core.rag.rerank.build_rerank_fn`` 注入
（DashScope qwen3-rerank）：``RERANK__ENABLED=false``（默认）或无 ``EMBEDDING__API_KEY`` 时
返回 None、行为与无精排一致；开启后对 RRF 融合结果做 Cross-Encoder 精排（rerank_top_n
跟随调用方 top_k）。精排失败由 hybrid_retriever 降级回融合结果。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from sqlalchemy import text as sa_text

from core.rag.llamaindex.pg_store import PG_TABLE_NAME, TEXT_SEARCH_CONFIG
from core.rag.retriever.base import Retriever
from core.rag.types import ChunkMeta, RetrievalResult

logger = logging.getLogger(__name__)


def _nodes_to_docs(result: Any) -> list[dict]:
    """VectorStoreQueryResult.nodes → hybrid_retriever 期望的 dict 列表。

    每项含 chunk_id/node 内容/score/file_path，chunk_id 用 node_id（dense 与 sparse 同源，
    RRF 融合的 join key）。注意：PGVectorStore 返回的 nodes 是纯 TextNode（无 score 字段），
    余弦相似度单独放在 result.similarities（与 nodes 按下标一一对应），不能从 node 上取。
    """
    docs: list[dict] = []
    similarities = result.similarities or []
    for i, n in enumerate(result.nodes or []):
        content = n.get_content() if hasattr(n, "get_content") else getattr(n, "text", "")
        if not content:
            continue
        score = similarities[i] if i < len(similarities) else 0.0
        docs.append(
            {
                "chunk_id": getattr(n, "node_id", "") or "",
                "content": content,
                "score": float(score or 0.0),
                "file_path": (getattr(n, "metadata", None) or {}).get("file_path", ""),
            }
        )
    return docs


# sparse 路 SQL（常量：表名/配置名都是模块级常量，SQL 文本固定，避免每次 bm25_search 重构 sa_text）。
# 绕开库 SPARSE（base.py:956 to_tsquery 对中文不分词）：zhparser 分词查询取 lexeme，| 连接成 OR
# tsquery -> 召回含任一查询词的 chunk（ts_rank 排序，rerank 兜底）。文档侧 text_search_tsv 由
# alembic 030 改 TEXT_SEARCH_CONFIG 配置，与查询同源 -> token 对齐。
_BM25_SQL = sa_text(f"""
    WITH q AS (
        SELECT to_tsquery('{TEXT_SEARCH_CONFIG}', string_agg(lexeme, ' | ')) AS q
        FROM unnest(to_tsvector('{TEXT_SEARCH_CONFIG}', :q)) AS t(lexeme)
    )
    SELECT d.node_id, d.text,
           d.metadata_ ->> 'file_path' AS file_path,
           ts_rank(d.text_search_tsv, q.q) AS rank
    FROM data_{PG_TABLE_NAME} d, q
    WHERE d.metadata_ @> :cid
      AND d.text_search_tsv @@ q.q
    ORDER BY rank DESC
    LIMIT :k
""")


class _PgSparseStore:
    """适配 ``hybrid_retriever`` 的 es_store 接口：``bm25_search`` 走 PG tsvector 全文。

    duck-type ``ESChunkStore``：只需实现 ``async bm25_search(query, course_id, top_k)``。
    hybrid_retriever 把它当 BM25 路。**不调库的 SPARSE 模式**（llama-index base.py:956 用
    to_tsquery 且其预处理 ``re.sub(r'\\W+',' ',q)`` 对中文不分词，整串一个 token -> 0 命中），
    改为直接 SQL（``_BM25_SQL``）：用 zhparser 把查询分词取 lexeme，``|`` 连接成 OR tsquery，
    ``ts_rank`` 排序。文档侧 ``text_search_tsv`` 与查询同用 ``TEXT_SEARCH_CONFIG`` 配置，token
    对齐。失败（如未装 zhparser/未建列）安全降级返回 []，hybrid_retriever 据此跳过该路，退化纯 dense。
    """

    async def bm25_search(self, query: str, course_id: str, top_k: int = 50) -> list[dict]:
        from core.db.database import engine  # noqa: PLC0415  函数内 import 保测试隔离

        if not query or not query.strip():
            return []
        try:
            async with engine.begin() as conn:
                rows = await conn.execute(
                    _BM25_SQL,
                    {"q": query, "cid": json.dumps({"course_id": course_id}), "k": top_k},
                )
            return [
                {
                    "chunk_id": r.node_id,
                    "content": r.text,
                    "score": float(r.rank),
                    "file_path": r.file_path or "",
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning(
                "PG SPARSE 查询失败（降级跳过 sparse 路）course=%s: %s", course_id, exc
            )
            return []


class LlamaIndexRetriever(Retriever):
    """pgvector 检索器（dense + sparse → RRF 融合，course 隔离）。"""

    async def is_available(self) -> tuple[bool, str]:
        from core.rag.llamaindex.pg_store import is_llamaindex_pg_available  # noqa: PLC0415

        return is_llamaindex_pg_available()

    async def _fuse(self, course_id: str, query: str, top_k: int) -> list[dict]:
        """dense(DEFAULT) + sparse(SPARSE) → hybrid_retriever RRF 融合，返回 list[dict]。"""
        from dataclasses import replace

        from core.rag.hybrid_retriever import retrieve as hybrid_retrieve  # noqa: PLC0415
        from core.rag.llamaindex.pg_store import (  # noqa: PLC0415
            course_filter,
            get_embed_model,
            get_vector_store,
        )
        from core.rag.rerank import build_rerank_fn  # noqa: PLC0415
        from core.rag.retrieval_config import DEFAULT_CONFIG  # noqa: PLC0415
        from llama_index.core.vector_stores.types import (  # noqa: PLC0415
            VectorStoreQuery,
            VectorStoreQueryMode,
        )
        from settings import get_settings  # noqa: PLC0415

        ok, reason = await self.is_available()
        if not ok:
            logger.warning("llamaindex_pg retriever skipped: %s", reason)
            return []

        vs = get_vector_store()
        emb_model = get_embed_model()
        filters = course_filter(course_id)

        async def _dense(query_text: str, k: int) -> list[dict]:
            # dense 路：先 embed query（HNSW 余弦检索需要 query_embedding）
            q_emb = await emb_model._aget_query_embedding(query_text)
            q = VectorStoreQuery(
                query_embedding=q_emb,
                similarity_top_k=k,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=filters,
            )
            result = await vs.aquery(q)
            return _nodes_to_docs(result)

        sparse_store = _PgSparseStore()

        # 精排注入：无 key 或 RERANK__ENABLED=false（默认）时 build_rerank_fn() 返回 None，
        # rerank_enabled 置 False（行为与改动前一致，hybrid_retriever 据此跳过精排）。
        # rerank_top_n 跟随调用方 top_k——DEFAULT_CONFIG.rerank_top_n 硬编码 5，调用方传
        # top_k=10 时不应被砍到 5。
        rerank_fn = build_rerank_fn()
        cfg = replace(
            DEFAULT_CONFIG,
            rerank_enabled=rerank_fn is not None,
            rerank_top_n=top_k,
            min_rerank_score=get_settings().rerank.min_score,
        )

        results = await hybrid_retrieve(
            query,
            course_id,
            cfg,
            es_store=sparse_store,
            dense_search_fn=_dense,
            rerank_fn=rerank_fn,
        )
        return results or []

    async def retrieve(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> list[RetrievalResult]:
        """检索相关文档片段（dense+sparse 融合后取 top_k）。"""
        try:
            t0 = time.perf_counter()
            results = await self._fuse(course_id, query, top_k)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            out: list[RetrievalResult] = []
            for r in results[:top_k]:
                content = r.get("content", "")
                if not content:
                    continue
                out.append(
                    RetrievalResult(
                        content=content,
                        score=float(r.get("score", 0.0)),
                        source_chunk=ChunkMeta(
                            source_path=r.get("file_path", ""),
                            start_char=0,
                            end_char=len(content),
                            chunk_id=r.get("chunk_id", ""),
                        ),
                        metadata={"backend": "llamaindex_pg"},
                    )
                )
            logger.info(
                "llamaindex_pg.retrieve course=%s query=%.60s top_k=%d results=%d elapsed_ms=%d",
                course_id, query[:60], top_k, len(out), elapsed_ms,
            )
            return out
        except Exception as exc:
            logger.error(
                "llamaindex_pg.retrieve failed course=%s: %s",
                course_id, exc, exc_info=True,
            )
            return []

    async def retrieve_context(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
        max_chars: int = 4000,
        **kwargs,
    ) -> str:
        """检索并拼接为上下文字符串（供 LLM prompt）。"""
        try:
            t0 = time.perf_counter()
            results = await self._fuse(course_id, query, top_k)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            if not results:
                return ""

            contexts = [
                {"content": r.get("content", ""), "file_path": r.get("file_path", "")}
                for r in results[:top_k]
            ]
            context_text = _format_contexts(contexts, limit=top_k, max_chars=max_chars)
            logger.info(
                "llamaindex_pg.retrieve_context course=%s query=%.60s top_k=%d chars=%d elapsed_ms=%d",
                course_id, query[:60], top_k, len(context_text), elapsed_ms,
            )
            return context_text
        except Exception as exc:
            logger.error(
                "llamaindex_pg.retrieve_context failed course=%s: %s",
                course_id, exc, exc_info=True,
            )
            return ""


def _format_contexts(
    contexts: list[dict], limit: int, max_chars: int
) -> str:
    """拼接上下文：按 max_chars 截断，超出部分尾部裁剪（与 LightRAG 格式一致：纯文本分段）。"""
    parts: list[str] = []
    total = 0
    for c in contexts[:limit]:
        content = (c.get("content") or "").strip()
        if not content:
            continue
        if total + len(content) > max_chars:
            content = content[: max(0, max_chars - total)]
        parts.append(content)
        total += len(content)
        if total >= max_chars:
            break
    return "\n\n".join(parts)
