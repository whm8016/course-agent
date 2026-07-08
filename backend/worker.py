"""ARQ worker 入口。

启动方式（Docker / 命令行）：
    python -m arq worker.WorkerSettings

包含后台任务：
1. run_indexing         – LightRAG 知识库索引（替代 BackgroundTasks）
2. run_llamaindex_build – LlamaIndex 向量索引（替代 BackgroundTasks）
3. cron_flush_memory    – Mem0 批量刷新（cron，每 30s 扫描 Redis mem_flush:* key）
4. flush_all_pending_job – Shutdown 时 Flush 所有 pending buffer
"""
from __future__ import annotations

import json
import logging
import sys
import os

# 让 import 能找到同目录下所有模块（与 main.py 一致）
sys.path.insert(0, os.path.dirname(__file__))

# Worker 进程自己配置日志（不经过 main.py），与主进程保持同样的 JSON 格式
from pythonjsonlogger import jsonlogger as _jsonlogger  # noqa: E402
from core.observability.logging import ContextFilter as _ContextFilter  # noqa: E402

_LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    _jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
        json_ensure_ascii=False,
    )
)
_handler.addFilter(_ContextFilter())
logging.basicConfig(level=_LOG_LEVEL, handlers=[_handler])

logger = logging.getLogger(__name__)

_JOB_EVENTS_TTL = 3600  # 进度事件列表在 Redis 中保留 1 小时


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

async def _resolve_redis(ctx) -> tuple:
    """获取 redis 连接：优先用 ARQ 注入的 ctx["redis"]（ARQ 管理其生命周期），
    否则 fallback 自建一个（M-3：旧实现自建后从不 aclose，每次 cron/flush job 都泄漏
    一个连接，worker 长跑会耗尽连接池）。返回 (redis, should_close)：
    should_close=True 表示是本函数自建的、用完必须由调用方 aclose。
    """
    r = ctx.get("redis")
    if r is not None:
        return r, False
    import redis.asyncio as aioredis
    from settings import get_settings
    REDIS_URL = get_settings().db.redis_url.get_secret_value()
    return aioredis.from_url(REDIS_URL, decode_responses=True), True


async def _push_event(redis, job_id: str, event: dict) -> None:
    """向 Redis list 追加一条进度事件（供 WS 端轮询）。"""
    key = f"job:{job_id}:events"
    try:
        await redis.rpush(key, json.dumps(event, ensure_ascii=False))
        await redis.expire(key, _JOB_EVENTS_TTL)
    except Exception:
        logger.debug("_push_event failed job_id=%s", job_id, exc_info=True)


# ---------------------------------------------------------------------------
# 任务 1 & 2：知识库索引（复用 admin.py 中的实现）
# ---------------------------------------------------------------------------

async def run_indexing(
    ctx,
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
) -> None:
    """LightRAG 知识库索引后台任务。"""
    import time
    from core.observability import bind_context, log_flow
    from core.rag.lightrag import acquire_index_dlock, release_index_dlock

    # 分布式索引锁：跨 worker 进程互斥（多容器/多进程也能护住），根治多任务并发
    # ainsert 同一份 lightrag_store 导致的"重复文档"刷屏与卡死。被占 = 已有任务在跑，
    # 直接跳过（DB status 仍是 indexing，前端继续等原任务）。
    lock, renew = await acquire_index_dlock(course_id)
    if lock is None:
        logger.warning(
            "课程 %s 已有索引任务在运行（分布式锁），跳过本次 job_id=%s",
            course_id, ctx.get("job_id"),
        )
        return
    try:
        job_id = str(ctx.get("job_id", kb_id or ""))
        bind_context(job_id=job_id, course_id=course_id)
        t0 = time.perf_counter()
        log_flow("worker.indexing.start", job_id=job_id, course_id=course_id,
                 kb_id=kb_id, files=len(file_paths), resume_from_chunk=resume_from_chunk)
        try:
            from api.admin import _run_indexing
            await _run_indexing(kb_id, course_id, file_paths, resume_from_chunk)
            _el = int((time.perf_counter() - t0) * 1000)
            log_flow("worker.indexing.complete", job_id=job_id, course_id=course_id, elapsed_ms=_el)
            from core.observability.metrics import observe_worker_job
            observe_worker_job("indexing", "ok", _el)
        except Exception as exc:
            _el = int((time.perf_counter() - t0) * 1000)
            log_flow("worker.indexing.error", logger=logger, level=logging.ERROR,
                     job_id=job_id, error=str(exc), elapsed_ms=_el)
            from core.observability.metrics import observe_worker_job
            observe_worker_job("indexing", "error", _el)
            raise
    finally:
        await release_index_dlock(lock, renew)


async def run_llamaindex_build(
    ctx,
    kb_id: str,
    course_id: str,
    file_paths: list[str],
) -> None:
    """LlamaIndex 向量索引后台任务。"""
    import time
    from core.observability import bind_context, log_flow
    from core.rag.lightrag import acquire_index_dlock, release_index_dlock

    # 分布式索引锁：与 run_indexing 共用 course 级锁，同一课程不能同时跑两种索引。
    lock, renew = await acquire_index_dlock(course_id)
    if lock is None:
        logger.warning(
            "课程 %s 已有索引任务在运行（分布式锁），跳过本次 job_id=%s",
            course_id, ctx.get("job_id"),
        )
        return
    try:
        job_id = str(ctx.get("job_id", kb_id or ""))
        bind_context(job_id=job_id, course_id=course_id)
        t0 = time.perf_counter()
        log_flow("worker.llamaindex.start", job_id=job_id, course_id=course_id,
                 kb_id=kb_id, files=len(file_paths))
        try:
            from api.llama_rag import _run_llamaindex_build
            await _run_llamaindex_build(kb_id, course_id, file_paths)
            log_flow("worker.llamaindex.complete", job_id=job_id, course_id=course_id,
                     elapsed_ms=int((time.perf_counter() - t0) * 1000))
        except Exception as exc:
            log_flow("worker.llamaindex.error", logger=logger, level=logging.ERROR,
                     job_id=job_id, error=str(exc),
                     elapsed_ms=int((time.perf_counter() - t0) * 1000))
            raise
        finally:
            # 终态兜底：_run_llamaindex_build 内部静默吞所有异常（_mark_final 自带 try/except），
            # 若 DB 写入 3 次重试仍失败，status 会卡在 indexing、worker 却报 complete。这里在
            # 任务结束后复查 DB，仍为 indexing 才强制改 error（不覆盖 ready/paused/pending 等已落
            # 终态或被用户主动干预的状态），给用户重试入口，杜绝永久卡死。
            try:
                from core.db.database import AsyncSessionLocal, KnowledgeBase
                from sqlalchemy import select
                async with AsyncSessionLocal() as db:
                    async with db.begin():
                        r = await db.execute(
                            select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
                        )
                        kb = r.scalar_one_or_none()
                        if kb and kb.status == "indexing":
                            logger.warning(
                                "LlamaIndex 终态兜底：status 仍为 indexing，强制改为 error kb_id=%s",
                                kb_id,
                            )
                            kb.status = "error"
                            kb.error_msg = "索引任务已结束但终态回写失败，请重试"
                            kb.updated_at = time.time()
                from api.courses import invalidate_courses_cache
                await invalidate_courses_cache()
            except Exception:
                logger.exception("LlamaIndex 终态兜底失败 kb_id=%s", kb_id)
    finally:
        await release_index_dlock(lock, renew)


# ---------------------------------------------------------------------------
# 任务 3 & 4：Mem0 批量刷新（Producer-Consumer 模式）
# ---------------------------------------------------------------------------


async def cron_flush_memory(ctx) -> None:
    """每 30s 扫描 Redis mem_flush:* key，满批或 idle 超时则 flush。"""
    import time
    from settings.base import get_settings

    settings = get_settings()
    max_turns = settings.mem0.flush_max_turns
    idle_timeout = settings.mem0.flush_idle_timeout

    t0 = time.perf_counter()
    logger.debug("[worker] cron_flush_memory start max_turns=%d idle_timeout=%.1fs",
                 max_turns, idle_timeout)

    try:
        from core.memory.flush_manager import scan_and_flush

        # 优先用 ARQ 注入的 ctx["redis"]；fallback 自建的连接用完必须 aclose（M-3）。
        r, should_close = await _resolve_redis(ctx)
        try:
            flushed = await scan_and_flush(r, max_turns, idle_timeout)
        finally:
            if should_close:
                await r.aclose()

        logger.info("[worker] cron_flush_memory complete flushed=%d elapsed_ms=%d",
                    flushed, int((time.perf_counter() - t0) * 1000))

    except Exception as exc:
        logger.warning("[worker] cron_flush_memory error: %s", exc, exc_info=True)


async def flush_all_pending_job(ctx) -> None:
    """Flush 所有 pending buffer（shutdown 时由 main.py enqueue）。"""
    import time
    t0 = time.perf_counter()
    logger.info("[worker] flush_all_pending_job start")

    try:
        from core.memory.flush_manager import flush_all_pending

        # 优先用 ARQ 注入的 ctx["redis"]；fallback 自建的连接用完必须 aclose（M-3）。
        r, should_close = await _resolve_redis(ctx)
        try:
            flushed = await flush_all_pending(r)
        finally:
            if should_close:
                await r.aclose()

        logger.info("[worker] flush_all_pending_job complete flushed=%d elapsed_ms=%d",
                    flushed, int((time.perf_counter() - t0) * 1000))

    except Exception as exc:
        logger.warning("[worker] flush_all_pending_job error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# WorkerSettings
# ---------------------------------------------------------------------------

class WorkerSettings:
    """arq WorkerSettings：`python -m arq worker.WorkerSettings` 读取此配置。"""

    from settings import get_settings
    _redis_url = get_settings().db.redis_url.get_secret_value()
    import urllib.parse as _up

    _parsed = _up.urlparse(_redis_url)

    from arq.connections import RedisSettings

    redis_settings = RedisSettings(
        host=_parsed.hostname or "localhost",
        port=_parsed.port or 6379,
        password=_parsed.password or None,
        database=int((_parsed.path or "/0").lstrip("/") or 0),
    )

    functions = [run_indexing, run_llamaindex_build, flush_all_pending_job]
    max_jobs = 10
    job_timeout = 36000   # 单个任务最长 10 小时
    keep_result = 300    # 任务结果保留 5 分钟

    # Mem0 批量刷新 cron：每 30s 扫描一次 Redis
    from arq import cron
    from settings.base import get_settings
    _settings = get_settings()
    cron_jobs = [
        cron(cron_flush_memory, second=set(range(0, 60, _settings.mem0.flush_scan_interval))),
    ]
