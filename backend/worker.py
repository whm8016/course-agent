"""ARQ worker 入口。

启动方式（Docker / 命令行）：
    python -m arq worker.WorkerSettings

包含后台任务：
1. run_indexing         – LightRAG 知识库索引（替代 BackgroundTasks）
2. cron_flush_memory    – Mem0 批量刷新（cron，每 30s 扫描 Redis mem_flush:* key）
3. flush_all_pending_job – Shutdown 时 Flush 所有 pending buffer
"""
from __future__ import annotations

import json
import logging
import sys
import os
import time

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

# ARQ 任务重试与死信（plan 第三批-2）
_ARQ_MAX_TRIES = 3  # 单任务最大尝试次数（含首次）。索引任务重跑幂等（LightRAG purge 清空
                    # 旧数据），自动重试不产生重复数据；flush 任务内部全 catch 不 re-raise，
                    # max_tries 对它们无实际作用。
_DEADLETTER_KEY = "arq:deadletter"  # 全局死信 list（复用 job:{job_id}:events 的 list 风格）
_DEADLETTER_TTL = 7 * 24 * 3600     # 死信保留 7 天供运维排查/重放


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


async def _push_deadletter_if_terminal(
    ctx, *, function: str, error: BaseException
) -> None:
    """ARQ 任务终态失败（max_tries 用尽）写入 Redis 死信 list，供运维排查/重放。

    复用 _push_event 的 list 风格（rpush + expire）。仅在最后一次尝试失败时写
    （``job_try >= _ARQ_MAX_TRIES``）；中间失败由 ARQ 自动重试，不进死信。死信本身 best-effort，
    写失败只打 debug 日志不抛（观测不应影响主流程，更不能让死信把任务重新标记失败）。
    """
    job_try = ctx.get("job_try", 1)
    if job_try < _ARQ_MAX_TRIES:
        return  # 还有重试机会，交由 ARQ 自动重试
    job_id = str(ctx.get("job_id", ""))
    r, should_close = await _resolve_redis(ctx)
    try:
        payload = json.dumps(
            {
                "job_id": job_id,
                "function": function,
                "job_try": job_try,
                "error": repr(error)[:1000],
                "ts": time.time(),
            },
            ensure_ascii=False,
        )
        await r.rpush(_DEADLETTER_KEY, payload)
        await r.expire(_DEADLETTER_KEY, _DEADLETTER_TTL)
    except Exception:
        logger.debug(
            "deadletter write failed job_id=%s func=%s", job_id, function, exc_info=True
        )
    finally:
        if should_close:
            await r.aclose()


# ---------------------------------------------------------------------------
# 任务 1：知识库索引（复用 admin.py 中的实现）
# ---------------------------------------------------------------------------

async def run_indexing(
    ctx,
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
    backend: str = "lightrag",
) -> None:
    """知识库索引后台任务（lightrag / llamaindex_pg，按 backend 写 kb_builds 行）。"""
    import time
    from core.observability import bind_context, log_flow
    from core.rag.lightrag import acquire_index_dlock, release_index_dlock

    # 分布式索引锁：跨 worker 进程互斥（多容器/多进程也能护住），根治多任务并发
    # ainsert 同一份 lightrag_store 导致的"重复文档"刷屏与卡死。被占 = 已有任务在跑，
    # 直接跳过（DB status 仍是 indexing，前端继续等原任务）。
    lock, renew = await acquire_index_dlock(course_id, backend)
    if lock is None:
        logger.warning(
            "课程 %s 的 %s 索引任务已在运行（分布式锁），跳过本次 job_id=%s",
            course_id, backend, ctx.get("job_id"),
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
            await _run_indexing(kb_id, course_id, file_paths, resume_from_chunk, backend)
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
            await _push_deadletter_if_terminal(ctx, function="run_indexing", error=exc)
            raise
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
# 任务 5：L3 记忆巩固（事件驱动 + cron safety net）
# ---------------------------------------------------------------------------


async def consolidate_memory(ctx, user_id: str, course_id: str = "") -> None:
    """消费某 user(+course) 的 episodic pending → mem0 语义层升格。

    事件驱动触发（main.py _on_capability_complete）：importance 累计超阈值 / quiz 里程碑。
    幂等：mem0.add 内部去重，崩溃重试不产生重复事实。
    """
    import time
    t0 = time.perf_counter()
    try:
        from core.db.database import AsyncSessionLocal
        from core.memory.consolidation import consolidate

        async with AsyncSessionLocal() as db:
            result = await consolidate(db, user_id, course_id)
        if result["claimed"]:
            logger.info(
                "[worker] consolidate_memory user=%s course=%s claimed=%d promoted=%d elapsed_ms=%d",
                user_id, course_id, result["claimed"], result["promoted"],
                int((time.perf_counter() - t0) * 1000),
            )
    except Exception as exc:
        logger.warning(
            "[worker] consolidate_memory error user=%s course=%s: %s",
            user_id, course_id, exc, exc_info=True,
        )
        await _push_deadletter_if_terminal(ctx, function="consolidate_memory", error=exc)
        raise


async def cron_consolidate_memory(ctx) -> None:
    """5min safety net：捞长期 pending + 超时 processing 孤儿 → enqueue consolidate_memory。

    两种孤儿：
    1. pending 超过 _PENDING_STALE_SECONDS（session 空闲等价触发，importance 未攒够也兜底）
    2. processing 超过 _PROCESSING_TIMEOUT_SECONDS（崩溃遗留，用 created_at 近似）→ 回 pending 重领
    """
    import time
    from sqlalchemy import select, update

    from core.arq_pool import get_arq_pool
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from core.memory.consolidation import (
        _PENDING_STALE_SECONDS,
        _PROCESSING_TIMEOUT_SECONDS,
    )

    now = time.time()
    try:
        async with AsyncSessionLocal() as db:
            # 1. 崩溃遗留的 processing → 回 pending（下次 consolidate 重领；mem0 去重保证幂等）
            await db.execute(
                update(MemoryEpisode)
                .where(
                    MemoryEpisode.status == "processing",
                    MemoryEpisode.created_at < now - _PROCESSING_TIMEOUT_SECONDS,
                )
                .values(status="pending")
            )
            await db.commit()
            # 2. 长期 pending 的 (user, course) → 收集去重
            rows = (
                await db.execute(
                    select(MemoryEpisode.user_id, MemoryEpisode.course_id).where(
                        MemoryEpisode.status == "pending",
                        MemoryEpisode.created_at < now - _PENDING_STALE_SECONDS,
                    )
                )
            ).all()

        targets = {(r[0], r[1] or "") for r in rows}
        if not targets:
            return
        pool = await get_arq_pool()
        if pool is None:
            return
        enqueued = 0
        for uid, cid in targets:
            try:
                await pool.enqueue_job("consolidate_memory", user_id=uid, course_id=cid)
                enqueued += 1
            except Exception:
                logger.warning(
                    "[worker] cron_consolidate enqueue failed user=%s", uid, exc_info=True
                )
        logger.info("[worker] cron_consolidate_memory targets=%d enqueued=%d", len(targets), enqueued)
    except Exception as exc:
        logger.warning("[worker] cron_consolidate_memory error: %s", exc, exc_info=True)


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

    functions = [run_indexing, flush_all_pending_job, consolidate_memory]
    max_jobs = 10
    job_timeout = 36000   # 单个任务最长 10 小时
    keep_result = 300    # 任务结果保留 5 分钟
    max_tries = _ARQ_MAX_TRIES  # 失败自动重试：网络抖动/OOM/瞬时 DB 锁的容错（含首次）
    retry_jobs = True           # 显式开启重试（默认即 True，声明便于阅读；幂等性见 _ARQ_MAX_TRIES 注释）

    # Mem0 批量刷新 cron：降级为 5min（Phase 2 后 Redis buffer 不再被喂，主路径已是
    # episodic + consolidate_memory；保留以排干 Phase 2 之前的残留 buffer）。
    from arq import cron
    cron_jobs = [
        cron(cron_flush_memory, minute=set(range(0, 60, 5))),
        # 5min episodic safety net：捞长期 pending + 超时 processing 孤儿 → enqueue consolidate
        cron(cron_consolidate_memory, minute=set(range(0, 60, 5))),
    ]
