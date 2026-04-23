"""LlamaIndex 向量库：后台建索引 + 检索（管理员）。"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_admin
from api.courses import invalidate_courses_cache
from config import LLAMA_INDEX_KB_ROOT
from core.database import AsyncSessionLocal, KBFile, KnowledgeBase, get_db
from rag_llama.llamaindex_pipeline import LlamaIndexPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["llama-rag"])

# 进度回调在 worker 线程触发，节流后写入 DB，避免打爆库与缓存
_LLAMA_PROGRESS_THROTTLE: dict[str, float] = {}
_LLAMA_PROGRESS_MIN_INTERVAL = 0.8


async def _get_kb_or_404(db: AsyncSession, course_id: str) -> KnowledgeBase:
    r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.course_id == course_id))
    kb = r.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail=f"知识库 '{course_id}' 不存在")
    return kb


async def _update_llamaindex_progress(
    kb_id: str, batch_num: int, total_batches: int, *, _now: float | None = None
) -> None:
    t = time.monotonic() if _now is None else _now
    last = _LLAMA_PROGRESS_THROTTLE.get(kb_id, 0.0)
    if t - last < _LLAMA_PROGRESS_MIN_INTERVAL and batch_num < total_batches:
        return
    _LLAMA_PROGRESS_THROTTLE[kb_id] = t
    total = max(total_batches, 1)
    pct = min(99, int(100 * min(batch_num, total_batches) / total))
    msg = f"LlamaIndex embedding 批次 {batch_num}/{total_batches}"
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
                kb = r.scalar_one_or_none()
                if kb and kb.status == "indexing":
                    kb.progress = pct
                    kb.progress_msg = msg
                    kb.updated_at = time.time()
    except Exception:
        logger.exception("LlamaIndex 进度写入失败 kb_id=%s", kb_id)


async def _run_llamaindex_build(kb_id: str, course_id: str, file_paths: list[str]) -> None:
    """后台任务：与 DeepTutor `run_initialization_task` 类似，HTTP 先已把状态置为 indexing，这里慢慢 embed + 落盘。"""
    main_loop: asyncio.AbstractEventLoop | None = None
    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    try:
        pipeline = LlamaIndexPipeline(kb_base_dir=str(LLAMA_INDEX_KB_ROOT))

        def _on_progress(batch_num: int, total_batches: int) -> None:
            logger.info(
                "LlamaIndex embedding batches %s/%s (course_id=%s)",
                batch_num,
                total_batches,
                course_id,
            )
            if not main_loop:
                return

            def _schedule() -> None:
                asyncio.create_task(
                    _update_llamaindex_progress(kb_id, batch_num, total_batches)
                )

            main_loop.call_soon_threadsafe(_schedule)

        ok = await pipeline.initialize(
            kb_name=course_id,
            file_paths=file_paths,
            progress_callback=_on_progress,
        )

        _LLAMA_PROGRESS_THROTTLE.pop(kb_id, None)

        async with AsyncSessionLocal() as db:
            async with db.begin():
                r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
                kb = r.scalar_one_or_none()
                if kb:
                    if ok:
                        kb.status = "ready"
                        kb.progress = 100
                        kb.progress_msg = "LlamaIndex 索引已完成"
                        kb.error_msg = ""
                    else:
                        kb.status = "error"
                        kb.error_msg = "LlamaIndex 索引失败"
                    kb.updated_at = time.time()
        await invalidate_courses_cache()
    except Exception as e:
        _LLAMA_PROGRESS_THROTTLE.pop(kb_id, None)
        logger.exception("LlamaIndex 后台建库失败 course_id=%s", course_id)
        try:
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
                    kb = r.scalar_one_or_none()
                    if kb:
                        kb.status = "error"
                        kb.error_msg = str(e)[:500]
                        kb.updated_at = time.time()
            await invalidate_courses_cache()
        except Exception:
            logger.exception("写入 LlamaIndex 失败状态时出错")


@router.post("/kb/{course_id}/llamaindex/build")
async def build_llamaindex_index(
    course_id: str,
    background_tasks: BackgroundTasks,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    接受请求后立即返回；真正建索引在 FastAPI BackgroundTasks 中执行（同 DeepTutor 思路）。
    落盘：{LLAMA_INDEX_KB_ROOT}/{course_id}/llamaindex_storage/
    """
    kb = await _get_kb_or_404(db, course_id)

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    if kb.status == "indexing":
        raise HTTPException(
            status_code=409,
            detail="知识库正在索引中，请稍候完成后再试",
        )

    kb.status = "indexing"
    kb.progress = 0
    kb.error_msg = ""
    kb.progress_msg = "LlamaIndex 向量索引构建中…"
    kb.updated_at = time.time()
    await db.commit()
    await invalidate_courses_cache()

    background_tasks.add_task(_run_llamaindex_build, kb.id, course_id, file_paths)

    return {
        "accepted": True,
        "message": "LlamaIndex 索引任务已在后台启动（与 DeepTutor 后台初始化类似）",
        "course_id": course_id,
        "file_count": len(file_paths),
        "storage_dir": str(Path(LLAMA_INDEX_KB_ROOT) / course_id / "llamaindex_storage"),
    }


class LlamaSearchBody(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


@router.post("/kb/{course_id}/llamaindex/search")
async def search_llamaindex(
    course_id: str,
    body: LlamaSearchBody,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    await _get_kb_or_404(db, course_id)
    pipeline = LlamaIndexPipeline(kb_base_dir=str(LLAMA_INDEX_KB_ROOT))
    return await pipeline.search(
        query=body.query,
        kb_name=course_id,
        top_k=body.top_k,
    )