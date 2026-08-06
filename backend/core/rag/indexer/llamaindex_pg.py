"""llamaindex_pg 索引器：复用 ingestion.parse_files 的 chunks → TextNode → PGVectorStore。

与 LightRAGIndexer 的根本区别：不做逐 chunk 的 LLM 实体/关系抽取（LightRAG 慢的根因，
几千 chunk 即数小时），只做 embedding 批调用，同样语料分钟级完成。chunks 由
``parse_files`` 统一切好（与 LightRAG 摄入前半段完全一致——同一份解析+切块代码），
本索引器只负责 embed + 入库。

node_id 用 ``md5(course_id|content)`` 确定性生成：同一课程同一内容 → 同一 node_id，
便于将来 upsert 与按内容去重；delete 按 course_id metadata 清行（见 delete）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from core.rag.indexer.base import Indexer
from core.rag.types import IndexResult

logger = logging.getLogger(__name__)


class LlamaIndexIndexer(Indexer):
    """pgvector 索引器（LlamaIndex VectorStoreIndex + PGVectorStore）。"""

    async def index(
        self,
        course_id: str,
        file_paths: list[str],
        **kwargs,
    ) -> IndexResult:
        """索引文档：parse_files 切块 → TextNode(带 course_id) → VectorStoreIndex embed+入库。

        kwargs:
            resume_from_chunk: 断点续传，跳过前 N 个 chunk（与 LightRAG 续传语义一致）。
        """
        from core.rag.ingestion import parse_files  # noqa: PLC0415
        from core.rag.llamaindex.pg_store import (  # noqa: PLC0415
            get_embed_model,
            get_vector_store,
        )

        resume_from_chunk = int(kwargs.get("resume_from_chunk") or 0)

        try:
            # Step 1: 复用摄入前半段——解析 + 切块（CPU 密集，放线程池，与 LightRAG 同源）
            all_chunks, all_sources, _doc_texts = await asyncio.to_thread(
                parse_files, file_paths
            )

            if not all_chunks:
                return IndexResult(
                    course_id=course_id,
                    files_indexed=0,
                    chunks_created=0,
                    status="skipped",
                    error="no_chunks",
                )

            # 断点续传：跳过前 N 个（前 N 个已写入，只 embed/写入剩余）
            start = min(resume_from_chunk, len(all_chunks))
            chunks = all_chunks[start:]
            sources = all_sources[start:]

            if not chunks:
                logger.info(
                    "llamaindex_pg 续传无新 chunk course=%s total=%d resume_from=%d",
                    course_id, len(all_chunks), resume_from_chunk,
                )
                return IndexResult(
                    course_id=course_id,
                    files_indexed=len(file_paths),
                    chunks_created=len(all_chunks),
                    status="success",
                )

            # Step 2: 构造 TextNode（带 course_id metadata 供检索隔离 filter；node_id 确定性）
            from llama_index.core.schema import TextNode  # noqa: PLC0415

            nodes = [
                TextNode(
                    text=text,
                    metadata={"course_id": course_id, "file_path": src},
                    id_=hashlib.md5(f"{course_id}|{text}".encode("utf-8")).hexdigest(),
                )
                for text, src in zip(chunks, sources)
                if text and text.strip()
            ]
            if not nodes:
                return IndexResult(
                    course_id=course_id, files_indexed=len(file_paths), chunks_created=0,
                    status="skipped", error="no_nonempty_chunks",
                )

            # Step 3: VectorStoreIndex embed + 写入 PGVectorStore（同步阻塞，放线程池）
            def _build_and_embed() -> None:
                from llama_index.core import StorageContext, VectorStoreIndex  # noqa: PLC0415

                storage_context = StorageContext.from_defaults(
                    vector_store=get_vector_store()
                )
                # 不保留 index 引用：VectorStoreIndex 此处只作"embed + add"的编排器，
                # 数据已落 PGVectorStore（持久层），后续检索直接查 vector_store。
                VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=get_embed_model(),
                    show_progress=False,
                )

            await asyncio.to_thread(_build_and_embed)

            logger.info(
                "llamaindex_pg indexed course=%s files=%d chunks=%d",
                course_id, len(file_paths), len(nodes),
            )
            return IndexResult(
                course_id=course_id,
                files_indexed=len(file_paths),
                chunks_created=len(nodes),
                status="success",
            )
        except Exception as exc:
            logger.error(
                "LlamaIndexIndexer.index failed course=%s: %s",
                course_id, exc, exc_info=True,
            )
            return IndexResult(
                course_id=course_id,
                files_indexed=0,
                chunks_created=0,
                status="error",
                error=str(exc),
            )

    async def delete(self, course_id: str) -> bool:
        """删除课程所有 chunk（按 course_id metadata filter 清 data_kb_chunks 行）。

        用底层 SQL 而非 PGVectorStore.delete：后者按 ref_doc_id 删，不适合"按 course 清空"
        的批量场景。metadata_ 是 PGVectorStore 内部的 JSONB 列，@> 做 JSON 包含匹配。
        """
        from sqlalchemy import text  # noqa: PLC0415

        from core.db.database import engine  # noqa: PLC0415
        from core.rag.llamaindex.pg_store import PG_TABLE_NAME  # noqa: PLC0415

        table = f"data_{PG_TABLE_NAME}"
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(f"DELETE FROM {table} WHERE metadata_ @> :cid"),
                    {"cid": json.dumps({"course_id": course_id})},
                )
            logger.info("llamaindex_pg deleted course=%s (table=%s)", course_id, table)
            return True
        except Exception as exc:
            logger.error(
                "LlamaIndexIndexer.delete failed course=%s: %s",
                course_id, exc, exc_info=True,
            )
            return False
