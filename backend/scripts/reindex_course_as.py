"""One-off：把某课程的原文件用当前 chunking/图片描述管线重新索引到**另一个 course_id**下。

不touch 线上 course_id 的 PG 行（LlamaIndexIndexer.index 只按 metadata.course_id 写入新行，
delete 也是按 course_id 隔离），因此可以安全对照「图片描述改进前后」的效果：debug_retrieve.py
指向新 course_id 即可看到新流水线产出的 chunk/embedding，旧 course_id 数据原样保留。

用法（同 chunk_enrich_compare.py 约定，-P 避免工作目录遮蔽 regex 触发 ImportError）：
    cd backend
    python -P scripts/reindex_course_as.py <src_course_id> <new_course_id>

产物：
    - PG data_kb_chunks 里 metadata.course_id=<new_course_id> 的新行（embedding+tsvector 齐全）
    - data/ingest_chunks/course_<new_course_id>/latest.json 切块审计（供人工检查 chunk 质量）
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from core.db.database import AsyncSessionLocal, KBFile, KnowledgeBase  # noqa: E402
from core.rag.indexer.llamaindex_pg import LlamaIndexIndexer  # noqa: E402


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


async def run(src_course_id: str, new_course_id: str) -> None:
    if src_course_id == new_course_id:
        raise SystemExit("new_course_id 必须与 src_course_id 不同，避免覆盖线上索引")

    file_paths = await _load_file_paths(src_course_id)
    print(f"src={src_course_id} new={new_course_id} files={[Path(p).name for p in file_paths]}")

    indexer = LlamaIndexIndexer()
    # 先清一遍 new_course_id 下的旧行（若本脚本之前跑过），保证本次是干净重建
    await indexer.delete(new_course_id)
    result = await indexer.index(new_course_id, file_paths)
    print(f"status={result.status} files_indexed={result.files_indexed} "
          f"chunks_created={result.chunks_created} error={result.error}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("用法: python -P scripts/reindex_course_as.py <src_course_id> <new_course_id>")
    asyncio.run(run(sys.argv[1], sys.argv[2]))
