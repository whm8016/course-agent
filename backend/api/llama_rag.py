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
from settings import get_settings
LLAMA_INDEX_KB_ROOT = get_settings().paths.llama_index_kb_root
from core.db.database import AsyncSessionLocal, KBFile, KnowledgeBase, get_db
from core.rag.llamaindex.llamaindex_pipeline import LlamaIndexPipeline

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


async def _llama_final_fallback(kb_id: str, course_id: str) -> None:
    """终态兜底：_mark_final 三次重试全失败时强制收尾，避免 status 永久卡 indexing。

    与 worker.run_llamaindex_build 的兜底（无脑改 error）不同，这里按索引产物是否真
    生成判定：docstore.json 在 → ready（embedding 实际成功，仅终态回写 DB 失败）；
    不在 → error。仅在 status 仍为 indexing 时介入，绝不覆盖 _mark_final 已写入的
    ready/paused 等终态。

    放在 _run_llamaindex_build 内部 try/finally 里调用，覆盖 ARQ 与 BackgroundTasks
    两种调用路径（worker 那层只对 ARQ 生效）。worker 的外层兜底作为最后一道保留——
    连本函数都失败（DB 彻底不可用）时由它强制改 error。
    """
    try:
        storage_ok = (
            Path(LLAMA_INDEX_KB_ROOT) / course_id / "llamaindex_storage" / "docstore.json"
        ).exists()
        async with AsyncSessionLocal() as db:
            async with db.begin():
                r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
                kb = r.scalar_one_or_none()
                if kb and kb.status == "indexing":
                    if storage_ok:
                        logger.warning(
                            "LlamaIndex 终态兜底：storage 已生成但终态回写失败，按 ready 处理 kb_id=%s",
                            kb_id,
                        )
                        kb.status = "ready"
                        kb.progress = 100
                        kb.progress_msg = "LlamaIndex 索引已完成"
                        kb.error_msg = ""
                    else:
                        logger.warning(
                            "LlamaIndex 终态兜底：status 仍为 indexing 且 storage 未生成，强制改 error kb_id=%s",
                            kb_id,
                        )
                        kb.status = "error"
                        kb.error_msg = "索引任务已结束但终态回写失败，请重试"
                    kb.updated_at = time.time()
        await invalidate_courses_cache()
    except Exception:
        logger.exception("LlamaIndex 终态兜底失败 kb_id=%s", kb_id)


async def _run_llamaindex_build(kb_id: str, course_id: str, file_paths: list[str]) -> None:
    """后台任务：与 `run_initialization_task` 类似，HTTP 先已把状态置为 indexing，这里慢慢 embed + 落盘。"""
    main_loop: asyncio.AbstractEventLoop | None = None
    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    async def _mark_final(
        *,
        status: str,
        err_msg: str = "",
        progress: int | None = None,
        progress_msg: str | None = None,
    ) -> None:
        """统一终态回写：只在 status 仍为 indexing 时覆盖，避免压掉用户手动 pause/stop。

        带 3 次重试 —— DB 写入偶发失败（连接池超时/锁冲突）时给恢复机会，避免 status
        永久卡 indexing（_run_llamaindex_build 内部全 catch，单次失败会被静默吞掉）。
        调用处仍用 asyncio.shield 包裹，确保任务被取消时回写也能跑完。3 次全失败由
        worker.run_llamaindex_build 的终态兜底强制改 error 兜住。
        """
        for attempt in range(3):
            try:
                async with AsyncSessionLocal() as db:
                    async with db.begin():
                        r = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
                        kb = r.scalar_one_or_none()
                        if not kb:
                            return
                        # 期间若被 pause/stop 接口强制改态（≠indexing），不覆盖（同 _run_indexing）
                        if kb.status != "indexing":
                            logger.info("LlamaIndex 终态被外部干预 kb=%s 当前=%s", kb_id, kb.status)
                            return
                        kb.status = status
                        kb.error_msg = err_msg
                        kb.updated_at = time.time()
                        if progress is not None:
                            kb.progress = progress
                        if progress_msg is not None:
                            kb.progress_msg = progress_msg
                await invalidate_courses_cache()
                return  # 成功则退出
            except Exception:
                logger.warning(
                    "LlamaIndex 终态回写第 %d 次失败 kb_id=%s",
                    attempt + 1, kb_id, exc_info=True,
                )
                if attempt < 2:
                    await asyncio.sleep(1)
        logger.error("LlamaIndex 终态回写 3 次均失败 kb_id=%s（worker 兜底会强制改 error）", kb_id)

    try:
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
            await asyncio.shield(_mark_final(
                status="ready" if ok else "error",
                err_msg="" if ok else "LlamaIndex 索引失败",
                progress=100 if ok else None,
                progress_msg="LlamaIndex 索引已完成" if ok else None,
            ))
        except asyncio.CancelledError:
            # 同 _run_indexing：CancelledError 不被 except Exception 捕获，需单独兜底，
            # 否则 status 永久卡 indexing。uncancel 清除取消计数，让回写 await 能跑完。
            _LLAMA_PROGRESS_THROTTLE.pop(kb_id, None)
            logger.warning("LlamaIndex 任务被取消 course_id=%s", course_id)
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            try:
                await asyncio.shield(_mark_final(
                    status="error",
                    err_msg="LlamaIndex 任务被中断（worker 超时/OOM/重启），可重试",
                ))
            except asyncio.CancelledError:
                logger.warning("LlamaIndex 终态回写收到二次取消 kb_id=%s", kb_id)
        except Exception as e:
            _LLAMA_PROGRESS_THROTTLE.pop(kb_id, None)
            logger.exception("LlamaIndex 后台建库失败 course_id=%s", course_id)
            try:
                await asyncio.shield(_mark_final(status="error", err_msg=str(e)[:500]))
            except asyncio.CancelledError:
                logger.warning("LlamaIndex 终态回写收到取消信号 kb_id=%s", kb_id)
    finally:
        # 终态兜底：无论正常完成/取消/异常，最后都复查一次。_mark_final 已成功写入
        # ready 时 status!=indexing，本调用直接跳过；仅当 _mark_final 三次重试全失败
        # （status 仍 indexing）时才按 storage 真相补改。覆盖 ARQ 与 BackgroundTasks。
        await _llama_final_fallback(kb_id, course_id)


@router.post("/kb/{course_id}/llamaindex/build")
async def build_llamaindex_index(
    course_id: str,
    background_tasks: BackgroundTasks,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    接受请求后立即返回；真正建索引在 FastAPI BackgroundTasks 中执行（同 思路）。
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

    # 必须走 ARQ（与 LightRAG 索引共用 worker 单进程 + 索引锁）；Redis 故障不降级
    # 到 BackgroundTasks，直接报错回滚，避免跨 gunicorn 进程撞存储。
    from core.arq_pool import get_arq_pool
    arq_pool = await get_arq_pool()
    if arq_pool is None:
        raise HTTPException(status_code=503, detail="任务队列（ARQ/Redis）不可用，请稍后重试")
    await arq_pool.enqueue_job("run_llamaindex_build", kb.id, course_id, file_paths)

    return {
        "accepted": True,
        "message": "LlamaIndex 索引任务已在后台启动（与 后台初始化类似）",
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