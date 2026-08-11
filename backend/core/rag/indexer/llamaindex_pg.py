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
from pathlib import Path

from core.rag.indexer.base import Indexer
from core.rag.types import IndexResult

logger = logging.getLogger(__name__)


def _node_id(course_id: str, text: str) -> str:
    """与 data_kb_chunks.node_id 同源（``md5(course_id|text)``）；审计 JSON 与入库行靠它
    对齐，不可有第二份公式——否则 JSON 里的 node_ids 与实际 PG 主键漂移后这列就是废的。"""
    return hashlib.md5(f"{course_id}|{text}".encode("utf-8")).hexdigest()


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
        from core.rag.ingestion import parse_files, persist_ingest_chunks  # noqa: PLC0415
        from core.rag.llamaindex.pg_store import (  # noqa: PLC0415
            get_embed_model,
            get_vector_store,
        )

        resume_from_chunk = int(kwargs.get("resume_from_chunk") or 0)

        try:
            # Step 1: 复用摄入前半段——解析 + 切块（CPU 密集，放线程池，与 LightRAG 同源）
            all_chunks, all_sources, doc_texts, parse_errors = await asyncio.to_thread(
                parse_files, file_paths
            )

            if not all_chunks:
                # 解析失败的真正原因（如「MinerU 解析失败: 超过 200 页上限」）透传给索引层，
                # 写进 kb_builds.error_msg；无失败原因兜底 no_chunks。
                return IndexResult(
                    course_id=course_id,
                    files_indexed=0,
                    chunks_created=0,
                    status="skipped",
                    error="; ".join(parse_errors) or "no_chunks",
                )

            from settings import get_settings  # noqa: PLC0415

            # Phase 3: Contextual Chunking（与 LightRAG 侧同源，共用 _apply_contextual_enrichment
            # 与同一份 contextual_cache.json）。pgvector 的 BM25 走 zhparser tsvector，contextual
            # 文本能同时改善 dense 与稀疏召回——Anthropic 方案中收益最大的那一半（contextual BM25）
            # 此前在此后端完全缺失。自门控 + 整体降级已下沉到 helper 内，开关默认关 = 零变化。
            from core.rag.ingestion import _apply_contextual_enrichment  # noqa: PLC0415
            all_chunks = await _apply_contextual_enrichment(
                course_id, all_chunks, all_sources, doc_texts,
            )

            _chunk_cfg = get_settings().chunking

            # Phase 4：图片 VLM 描述回填（复用 LightRAG 路径同款管线：image_extractor 的
            # collect_image_candidates + desc_cache，跳过知识图谱写入——pgvector 无图谱）。
            # 与 LightRAG 共用同一个开关 chunking.inline_image_descriptions；两个后端各自
            # 建 course 目录下的 image_desc_cache.json，互不冲突、也不会对同一张图重复调 VLM。
            if _chunk_cfg.inline_image_descriptions:
                img_cache = (
                    Path(get_settings().paths.ingest_chunks_dir)
                    / f"course_{course_id}"
                    / "image_desc_cache.json"
                )
                try:
                    from core.rag.ingestion import _append_image_desc_chunks  # noqa: PLC0415
                    from core.rag.llamaindex.image_extractor import (  # noqa: PLC0415
                        caption_images_from_files,
                    )

                    await caption_images_from_files(
                        file_paths, cache_path=str(img_cache)
                    )
                    added = _append_image_desc_chunks(all_chunks, all_sources, img_cache)
                    if added:
                        logger.info(
                            "图片描述回填 course=%s 追加 %d 条 (llamaindex_pg)",
                            course_id, added,
                        )
                except Exception as exc:
                    logger.warning(
                        "图片描述回填失败（降级跳过）course=%s: %s", course_id, exc,
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
                    id_=_node_id(course_id, text),
                )
                for text, src in zip(chunks, sources)
                if text and text.strip()
            ]
            if not nodes:
                return IndexResult(
                    course_id=course_id, files_indexed=len(file_paths), chunks_created=0,
                    status="skipped", error="no_nonempty_chunks",
                )

            # 落盘摄入切块审计 JSON（先于 embed：embedding 失败也能看到切块）。落盘的是**全量**
            # all_chunks + resume_from_chunk 标记（与 LightRAG 侧对齐，非续传后的切片）；
            # node_ids 与上面 TextNode.id_ 同源（_node_id 单一公式），供审计 JSON 直接 join
            # 回 data_kb_chunks 行。开关 chunking.save_pg_ingest_chunks 默认开。
            await asyncio.to_thread(
                persist_ingest_chunks,
                course_id,
                file_paths,
                all_chunks,
                resume_from_chunk,
                all_sources,
                backend="llamaindex_pg",
                node_ids=[_node_id(course_id, t) for t in all_chunks],
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
