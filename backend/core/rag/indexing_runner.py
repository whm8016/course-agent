"""后台索引任务执行器（从 api/admin.py 归位的纯业务层，无 HTTP 概念）。

历史：_run_indexing/_run_indexing_llamaindex_pg 原住在 api/admin.py（约 270 行
状态机/进度回调/Redis 控制信号/终态回写守卫），导致 worker.py 反向 import
api 层（全仓唯一 from api）——ARQ job 层不应依赖 HTTP 路由模块。

搬移原则：代码逐行原样（仅去下划线前缀转公开名），配套下沉两件薄依赖：
- get_build/get_or_create_build（kb_builds 行助手，自 api/kb_indexing.py 下沉，
  kb_indexing 仍 re-export 保住 admin/teacher/测试的 import 路径）
- invalidate_courses_cache（自 api/courses.py 下沉到 core/db/cache.py，本模块
  依赖它——它曾是索引业务滞留 api 层的根因）
"""
from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.cache import invalidate_courses_cache
from core.db.database import AsyncSessionLocal, KbBuild, KnowledgeBase
from core.rag.ingestion import IndexingAborted, IndexingControl, ingest_to_lightrag

logger = logging.getLogger(__name__)


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



async def run_indexing(
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
    backend: str = "lightrag",
) -> None:
    """后台任务：LlamaIndex 解析 → LightRAG 摄入（附带进度回调，支持断点续传）。

    状态/进度按 backend 写 kb_builds 行（与 KB 行解耦）；backend 由调用方传入，
    不再从 kb.index_backend 读取。
    """
    # 1. 重置/保留进度，更新状态为 indexing（写本 build 行）
    async with AsyncSessionLocal() as db:
        async with db.begin():
            result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            )
            kb = result.scalar_one_or_none()
            if not kb:
                return
            build = await get_or_create_build(db, kb_id, backend)
            build.status = "indexing"
            build.error_msg = ""
            if resume_from_chunk == 0:
                build.progress = 0
                build.progress_msg = "准备中…"
                build.chunks_done = 0
                build.chunks_total = 0
                build.token_estimate = 0
            else:
                build.progress_msg = f"续传中（从第 {resume_from_chunk} 个文本块继续）…"
            build.updated_at = time.time()

    # 状态从 pending/error/paused → indexing，让前端的「就绪/未就绪」徽章及时变更
    await invalidate_courses_cache()

    # llamaindex_pg 分流：embedding 批调用分钟级完成，走独立的轻量路径（无需 LightRAG 的
    # control / 逐 chunk 进度 / purge），完成后早返回。下方 lightrag 主体零变化。
    if backend == "llamaindex_pg":
        await run_indexing_llamaindex_pg(kb_id, course_id, file_paths, resume_from_chunk, backend)
        await invalidate_courses_cache()
        return

    # 进度回调
    async def _on_progress(
        progress: int,
        msg: str,
        chunks_done: int,
        chunks_total: int,
        token_estimate: int,
    ) -> None:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                build = await get_build(db, kb_id, backend)
                if build:
                    build.progress = progress
                    build.progress_msg = msg
                    build.chunks_done = chunks_done
                    build.chunks_total = chunks_total
                    build.token_estimate = token_estimate
                    build.updated_at = time.time()

    # 控制信号走 Redis，跨 worker 共享；先清掉上次残留
    control = IndexingControl(kb_id)
    await control.clear()

    # 全新索引（非续传）：清空 LightRAG 旧数据，杜绝 ainsert 整批判重
    # （"Duplicate document" → failed entries）。续传绝不能清，会抹掉进度。
    if resume_from_chunk == 0:
        from core.rag.lightrag import purge_course_workspace
        await purge_course_workspace(course_id)
        # ES 双写副本同步清：ES 按 course_id 累积写，不清会让旧 chunk 在混合检索冒出
        # （best-effort，未启用/失败仅告警不阻断索引）
        from core.rag.es_client import delete_course_es_docs
        await delete_course_es_docs(course_id)

    abort_action: str | None = None
    abort_chunks_done = 0
    # 预置"未正常结束"兜底：任何未被显式覆盖的退出路径都至少写成 error，绝不留在 indexing。
    final_status: str = "error"
    final_err: str = "索引任务未正常结束"
    try:
        summary = await ingest_to_lightrag(
            course_id,
            file_paths,
            on_progress=_on_progress,
            resume_from_chunk=resume_from_chunk,
            control=control,
        )
        logger.info("索引完成 kb_id=%s summary=%s", kb_id, summary)
        # 只有真正产出 chunks 的索引才算 ready；0 chunk（解析全失败/空文档）→ error，
        # 把真实原因（summary.parse_errors 透传自 parse_files，如「MinerU 解析失败: 超过 200 页上限」）
        # 写进 error_msg，避免空索引被误判 ready、绿徽章骗用户而按钮却禁用。
        if summary.get("chunks", 0) > 0:
            final_status = "ready"
            final_err = ""
        else:
            final_status = "error"
            final_err = (
                "；".join(summary.get("parse_errors") or [])
                or "索引产出 0 个文本块（解析失败或文件为空），详见日志"
            )
    except IndexingAborted as e:
        abort_action = e.action
        abort_chunks_done = e.chunks_done
        if e.action == "pause":
            final_status = "paused"
            logger.info("索引已暂停 kb_id=%s chunks_done=%d", kb_id, e.chunks_done)
        else:
            final_status = "pending"
            logger.info("索引已终止 kb_id=%s", kb_id)
        final_err = ""
    except asyncio.CancelledError:
        # CancelledError 是 BaseException 子类，上面 except Exception 捕获不到。
        # ARQ worker 超时/OOM/重启会取消任务，若不在本分支兜底，异常直接冒泡 →
        # 下方终态回写不执行 → status 永久卡 indexing（前端一直"构建中"且无救）。
        logger.warning("索引任务被取消 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = "索引任务被中断（worker 超时/OOM/重启），可重试"
        # 捕获后取消计数仍在，后续 await（control.clear / 终态回写）会立即再抛
        # CancelledError。uncancel 清除计数，保证下方清理与回写能真正跑完。
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
    except Exception as e:
        logger.exception("索引失败 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = str(e)[:500]
    finally:
        # 不论结果如何，清掉信号，避免下次启动立即被旧信号中断
        await control.clear()

    # 2. 更新最终状态 —— 用 shield 保护回写协程，避免任务取消时 await 被二次取消；
    #    外层兜底 CancelledError/Exception，确保 _run_indexing 不会因回写失败而抛出，
    #    从而保证 indexing 一定会被改写成终态（ready/error/paused/pending），杜绝卡死。
    async def _apply_final() -> None:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                build = await get_build(db, kb_id, backend)
                if not build:
                    return
                # 期间若被 pause/stop 接口强制改态（≠indexing），说明用户已主动干预，
                # 不覆盖——否则被强制终止的任务跑完会回写 ready/paused，与用户意图冲突。
                if build.status != "indexing":
                    logger.info(
                        "索引终态被外部干预 kb=%s backend=%s 当前=%s，不覆盖为 %s",
                        kb_id, backend, build.status, final_status,
                    )
                    return
                build.status = final_status
                build.error_msg = final_err
                build.updated_at = time.time()
                if abort_action == "pause":
                    build.chunks_done = abort_chunks_done
                    build.progress_msg = (
                        f"已暂停（已完成 {abort_chunks_done}"
                        f"{f'/{build.chunks_total}' if build.chunks_total else ''} 个文本块）"
                    )
                elif abort_action == "stop":
                    build.progress = 0
                    build.progress_msg = "已终止"
                    build.chunks_done = 0
                    build.chunks_total = 0
                    build.token_estimate = 0

    try:
        await asyncio.shield(_apply_final())
    except asyncio.CancelledError:
        logger.warning("终态回写外层收到取消信号 kb_id=%s（shield 内已尽力完成）", kb_id)
    except Exception:
        logger.exception("索引终态回写失败 kb_id=%s course=%s", kb_id, course_id)

    # 索引结束（ready / error / paused / pending），重要：ready 时前端要切到 LightRAG 路径
    await invalidate_courses_cache()

async def run_indexing_llamaindex_pg(
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
    backend: str = "llamaindex_pg",
) -> None:
    """llamaindex_pg 后台索引：调 LlamaIndexIndexer.index，写 kb_builds 终态。

    与 _run_indexing（lightrag）的分工：本函数只处理 pgvector 后端，走 embedding 批调用
    （分钟级），无需 LightRAG 的 IndexingControl（暂停/终止）、逐 chunk 进度回调、
    purge_course_workspace。但复用同款"只在 status==indexing 时覆盖"的终态回写守卫，
    保证 status 不卡 indexing（与 run_indexing._apply_final 同构）。

    全新索引（resume_from_chunk==0）先 delete 旧向量行，避免重复索引产生重复 chunk
    （node_id 虽确定性，但 PGVectorStore.add 不保证按 node_id upsert）。
    """
    from core.rag import get_indexer  # noqa: PLC0415

    final_status = "error"
    final_err = "索引任务未正常结束"
    chunks_created = 0

    try:
        indexer = get_indexer("llamaindex_pg")

        # 全新索引：先清旧向量行（杜绝重复 chunk）；失败仅告警不阻断（最坏多几条重复行）
        if resume_from_chunk == 0:
            try:
                await indexer.delete(course_id)
            except Exception:
                logger.warning(
                    "llamaindex_pg 清旧数据失败 course=%s（继续索引）",
                    course_id, exc_info=True,
                )

        result = await indexer.index(
            course_id, file_paths, resume_from_chunk=resume_from_chunk
        )
        # IndexResult.status: success | skipped | error
        # 只有真正产出 chunks 的 success 才算 ready；skipped(0块)/error/success但0块 → error，
        # 把真实原因（result.error 透传自 parse_errors，如「MinerU 解析失败: 超过 200 页上限」）
        # 写进 error_msg，避免空索引被误判 ready、绿徽章骗用户而按钮却禁用。
        if result.status == "success" and result.chunks_created > 0:
            final_status = "ready"
            final_err = ""
            chunks_created = result.chunks_created
        else:
            final_status = "error"
            final_err = result.error or "索引产出 0 个文本块（解析失败或文件为空）"
    except asyncio.CancelledError:
        # ARQ worker 超时/OOM/重启会取消任务；不兜底则 status 永久卡 indexing
        logger.warning("llamaindex_pg 索引任务被取消 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = "索引任务被中断（worker 超时/OOM/重启），可重试"
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
    except Exception as e:
        logger.exception("llamaindex_pg 索引失败 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = str(e)[:500]

    # 终态回写（与 _run_indexing 同款守卫：只在 status==indexing 时覆盖，避免压掉用户手动干预）
    async def _apply_final() -> None:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                build = await get_build(db, kb_id, backend)
                if not build:
                    return
                if build.status != "indexing":
                    logger.info(
                        "llamaindex_pg 终态被外部干预 kb=%s 当前=%s，不覆盖为 %s",
                        kb_id, build.status, final_status,
                    )
                    return
                build.status = final_status
                build.error_msg = final_err
                build.updated_at = time.time()
                if final_status == "ready":
                    build.progress = 100
                    build.progress_msg = "pgvector 索引完成"
                    build.chunks_total = chunks_created
                    build.chunks_done = chunks_created

    try:
        await asyncio.shield(_apply_final())
    except asyncio.CancelledError:
        logger.warning("llamaindex_pg 终态回写收到取消信号 kb_id=%s（shield 内已尽力）", kb_id)
    except Exception:
        logger.exception("llamaindex_pg 终态回写失败 kb_id=%s course=%s", kb_id, course_id)
