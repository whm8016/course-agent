"""知识库索引入队编排（admin / teacher 共用）。

把"触发索引"接口里重复的状态校验 / 文件查询 / DB 预置 / ARQ 入队逻辑收口到这里，
admin 与 teacher 入口只保留各自的权限校验（_get_kb_or_404 / _get_owned_kb），其余
全部委托给本模块。

放 api 层（而非 core/rag/）是因为要调 api.courses.invalidate_courses_cache——放 core
会形成 core→api 循环依赖。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.courses import invalidate_courses_cache
from core.db.database import KBFile, KbBuild, KnowledgeBase

logger = logging.getLogger(__name__)

# 后端展示名（409/日志用）
_BACKEND_LABELS = {"llamaindex_pg": "pgvector"}


def _backend_label(backend: str) -> str:
    return _BACKEND_LABELS.get(backend, "LightRAG")


async def get_build(db: AsyncSession, kb_id: str, backend: str) -> KbBuild | None:
    """取 (kb_id, backend) 对应的 kb_builds 行；无则 None。"""
    r = await db.execute(
        select(KbBuild).where(KbBuild.kb_id == kb_id, KbBuild.backend == backend)
    )
    return r.scalar_one_or_none()


async def get_or_create_build(db: AsyncSession, kb_id: str, backend: str) -> KbBuild:
    """取或建 (kb_id, backend) 的 kb_builds 行（status=pending）。在调用方事务内 flush 取 id。"""
    build = await get_build(db, kb_id, backend)
    if build is None:
        build = KbBuild(kb_id=kb_id, backend=backend, status="pending")
        db.add(build)
        await db.flush()
    return build


async def trigger_kb_indexing(
    db: AsyncSession,
    kb: KnowledgeBase,
    course_id: str,
    backend: str = "lightrag",
    force: bool = False,
    resume: bool = False,
) -> dict:
    """触发知识库索引（入队 ARQ run_indexing），按 backend 写 kb_builds 行。

    kb 由调用方按各自权限校验后传入（admin 用 _get_kb_or_404，teacher 用 _get_owned_kb）。
    backend: lightrag | llamaindex_pg——每后端一条 kb_builds（get-or-create），状态/进度隔离，
    一课程可同时构建两后端。封装：状态校验 → 文件查询/存在性 → resume 计算 → DB 预置
    indexing → ARQ 入队 → 返回。

    ARQ/Redis 不可用时直接 503，由 get_db 回滚 build status——不降级 BackgroundTasks，避免
    索引跑到 gunicorn 多进程、跨进程写同一份 lightrag_store 导致重复文档刷屏/损坏/卡死。
    """
    build = await get_or_create_build(db, kb.id, backend)
    if build.status == "indexing" and not force:
        raise HTTPException(
            status_code=409,
            detail=f"{_backend_label(backend)} 正在索引中，请等待完成后再试",
        )

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件，请先上传文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    # 断点续传：error / paused 状态且有进度记录时生效（按本 build 的进度）
    resume_from = 0
    if resume and build.status in ("error", "paused") and build.chunks_done > 0:
        resume_from = build.chunks_done
        logger.info(
            "断点续传 course_id=%s backend=%s 从 chunk %d 继续", course_id, backend, resume_from
        )

    # 提前置本 build 为 indexing：接口返回时 DB 已是 indexing，前端 loadCourses 立即拿到 →
    # hasIndexing 触发轮询 → 刷新页面也不丢"进行中"状态。worker(_run_indexing) 仍会
    # 再写一次，幂等兜底。
    build.status = "indexing"
    build.progress = 0
    build.error_msg = ""
    build.progress_msg = "准备中…" if resume_from == 0 else f"续传中（从第 {resume_from} 块）…"
    if resume_from == 0:
        build.chunks_done = 0
        build.chunks_total = 0
        build.token_estimate = 0
    build.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()

    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("run_indexing", kb.id, course_id, file_paths, resume_from, backend)
    logger.info(
        "ARQ 索引任务已入队 course_id=%s backend=%s files=%d", course_id, backend, len(file_paths)
    )

    return {
        "message": "索引任务已启动" if resume_from == 0 else f"续传任务已启动（从第 {resume_from} 个文本块）",
        "course_id": course_id,
        "backend": backend,
        "file_count": len(file_paths),
        "resume_from_chunk": resume_from,
    }
