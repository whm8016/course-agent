"""One-off：对比某课程 pg 切块「不开 enrich（基线）」vs「开 contextual enrichment」。

只读 KnowledgeBase/KBFile 拿真实原文件路径，走 parse_files 切块 + 可选 contextual
enrichment，两份结果分别落盘到独立 course_id 目录（``{course_id}_chunkcmp_base`` /
``{course_id}_chunkcmp_enrich``），不复用真实 course_id 目录、不写 PG 向量表/
knowledge_bases/kb_builds，跟线上索引与问答完全隔离，可放心跑。

用法（同 run_chunk_eval.py 约定，-P 避免工作目录遮蔽 regex 触发 ImportError）：
    cd backend
    python -P scripts/chunk_enrich_compare.py <course_id>
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from core.db.database import AsyncSessionLocal, KBFile, KnowledgeBase  # noqa: E402
from core.rag.ingestion import (  # noqa: E402
    _apply_contextual_enrichment,
    parse_files,
    persist_ingest_chunks,
)
from settings import get_settings  # noqa: E402


def _node_id(scope: str, text: str) -> str:
    return hashlib.md5(f"{scope}|{text}".encode("utf-8")).hexdigest()


async def _load_file_paths(course_id: str) -> list[str]:
    async with AsyncSessionLocal() as db:
        kb_result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
        )
        kb = kb_result.scalar_one_or_none()
        if not kb:
            raise SystemExit(f"course_id={course_id} 未找到知识库")
        files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
        return [f.file_path for f in files_result.scalars().all()]


async def run(course_id: str) -> None:
    settings = get_settings()
    file_paths = await _load_file_paths(course_id)
    print(f"course={course_id} files={[Path(p).name for p in file_paths]}")
    print(f"strategy={settings.chunking.strategy} ingest_size={settings.chunking.ingest_size}")

    chunks, sources, doc_texts, parse_errors = parse_files(file_paths)
    print(f"chunks={len(chunks)} parse_errors={parse_errors}")

    base_scope = f"{course_id}_chunkcmp_base"
    out_base = persist_ingest_chunks(
        base_scope, file_paths, chunks, 0, sources,
        backend="llamaindex_pg",
        node_ids=[_node_id(base_scope, t) for t in chunks],
    )
    print(f"baseline（无 enrich）saved: {out_base}")

    enrich_scope = f"{course_id}_chunkcmp_enrich"
    settings.chunking.contextual_enrichment = True  # 仅本进程内存生效，不改 .env/DB
    enriched = await _apply_contextual_enrichment(enrich_scope, chunks, sources, doc_texts)
    out_enrich = persist_ingest_chunks(
        enrich_scope, file_paths, enriched, 0, sources,
        backend="llamaindex_pg",
        node_ids=[_node_id(enrich_scope, t) for t in enriched],
    )
    print(f"enriched（开 enrich）saved: {out_enrich}")


if __name__ == "__main__":
    asyncio.run(run(sys.argv[1]))
