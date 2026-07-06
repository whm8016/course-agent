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

from settings import get_settings
from api.courses import invalidate_courses_cache
from core.db.database import KBFile, KnowledgeBase

LLAMA_INDEX_KB_ROOT = get_settings().paths.llama_index_kb_root

logger = logging.getLogger(__name__)


async def trigger_kb_indexing(
    db: AsyncSession,
    kb: KnowledgeBase,
    course_id: str,
    force: bool = False,
    resume: bool = False,
) -> dict:
    """触发知识库 LightRAG 索引（入队 ARQ run_indexing）。

    kb 由调用方按各自权限校验后传入（admin 用 _get_kb_or_404，teacher 用 _get_owned_kb）。
    封装：状态校验 → 文件查询/存在性 → resume 计算 → DB 预置 indexing → ARQ 入队 → 返回。

    ARQ/Redis 不可用时直接 503，由 get_db 回滚 status——不降级 BackgroundTasks，避免
    索引跑到 gunicorn 多进程、跨进程写同一份 lightrag_store 导致重复文档刷屏/损坏/卡死。
    """
    if kb.status == "indexing" and not force:
        raise HTTPException(status_code=409, detail="正在索引中，请等待完成后再试")

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件，请先上传文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    # 断点续传：error / paused 状态且有进度记录时生效
    resume_from = 0
    if resume and kb.status in ("error", "paused") and kb.chunks_done > 0:
        resume_from = kb.chunks_done
        logger.info("断点续传 course_id=%s 从 chunk %d 继续", course_id, resume_from)

    # 提前置 indexing：接口返回时 DB 已是 indexing，前端 loadCourses 立即拿到 →
    # hasIndexing 触发轮询 → 刷新页面也不丢"进行中"状态。worker(_run_indexing) 仍会
    # 再写一次，幂等兜底。
    kb.status = "indexing"
    kb.progress = 0
    kb.error_msg = ""
    kb.progress_msg = "准备中…" if resume_from == 0 else f"续传中（从第 {resume_from} 块）…"
    if resume_from == 0:
        kb.chunks_done = 0
        kb.chunks_total = 0
        kb.token_estimate = 0
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()

    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("run_indexing", kb.id, course_id, file_paths, resume_from)
    logger.info("ARQ 索引任务已入队 course_id=%s files=%d", course_id, len(file_paths))

    return {
        "message": "索引任务已启动" if resume_from == 0 else f"续传任务已启动（从第 {resume_from} 个文本块）",
        "course_id": course_id,
        "file_count": len(file_paths),
        "resume_from_chunk": resume_from,
    }


async def trigger_llamaindex_build(
    db: AsyncSession,
    kb: KnowledgeBase,
    course_id: str,
) -> dict:
    """触发 LlamaIndex 向量索引构建（入队 ARQ run_llamaindex_build）。

    kb 由调用方按各自权限校验后传入（admin 用 _get_kb_or_404，teacher 用 _get_owned_kb）。
    封装：文件查询/存在性 → 409 校验 → DB 预置 indexing → ARQ 入队。

    统一用 flush（非 commit）：ARQ 故障 503 时由 get_db 回滚 status，避免卡 indexing。
    （原 llama_rag 版用 commit，是 BackgroundTasks 时代的残留——后台任务必须在请求事务
    结束前落盘；改 ARQ 后不再需要，commit 反而导致 503 时 status 不回滚。）
    """
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
    await db.flush()
    await invalidate_courses_cache()

    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("run_llamaindex_build", kb.id, course_id, file_paths)
    logger.info("ARQ LlamaIndex 任务已入队 course_id=%s files=%d", course_id, len(file_paths))

    storage_dir = str(Path(LLAMA_INDEX_KB_ROOT) / course_id / "llamaindex_storage")
    return {
        "accepted": True,
        "message": "LlamaIndex 索引任务已在后台启动",
        "course_id": course_id,
        "file_count": len(file_paths),
        "storage_dir": storage_dir,
    }
