import sys
import os
import logging
import asyncio

sys.path.insert(0, os.path.dirname(__file__))
from pythonjsonlogger import jsonlogger

_LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
        json_ensure_ascii=False,
    )
)
from core.observability.logging import ContextFilter  # noqa: E402
_log_handler.addFilter(ContextFilter())
logging.basicConfig(level=_LOG_LEVEL, handlers=[_log_handler])

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from api.run import router as run_router
from api.question import router as question_router
from api.question_notebook import router as question_notebook_router
from api.transcribe import router as transcribe_router
from api.admin import router as admin_router
from api.teacher import router as teacher_router
from api.auth import router as auth_router
from api.chat import router as chat_router
from api.courses import router as courses_router
from api.lightrag import router as lightrag_router
from api.memory import router as memory_router
from api.upload import router as upload_router
from api.sessions import router as sessions_router
from api.skills import router as skills_router
from api.skill_knowledge import router as skill_knowledge_router
from api.mcp import router as mcp_router
from api.bot import router as bot_router
from api.llm import router as llm_router
from api.search_config import router as search_config_router
from api.user_llm import router as user_llm_router
from settings import get_settings
UPLOAD_DIR = get_settings().paths.upload_dir
ALLOWED_ORIGINS = get_settings().security.allowed_origins
REDIS_URL = get_settings().db.redis_url.get_secret_value()
KB_STORE_DIR = get_settings().paths.kb_store_dir
TUTORBOT_ENABLED = get_settings().tutorbot.enabled
# 生产关 Swagger/OpenAPI（docs/redoc/openapi.json 三处一起，只关 docs 不算完；
# 8002 只绑宿主机回环后，这是第二道防线）
IS_PRODUCTION = get_settings().is_production
from core.db.database import init_db, close_db
from core.db.limiter import limiter
from core.arq_pool import get_arq_pool, close_arq_pool
# C-1/C-3：关停标志统一来自 leader 模块（lifespan shutdown 第一行 mark_shutting_down，
# 是「即将死亡」最早信号）。start_singleton_services 在每个 await 后读 is_shutting_down。
from core.leader import mark_shutting_down, is_shutting_down

logger = logging.getLogger(__name__)


# ---- EventBus 订阅者已归位 core/memory/lifecycle_subscribers.py ----


# ---- Leader 单例服务：启停解耦为运行时可调用（竞选接管时动态拉起/停止）----
_singletons_started = False


async def _safe_call(coro_fn, label: str) -> None:
    """调一个单例停止方法，吞异常（回滚/停止路径不能因单个失败阻断后续清理）。"""
    try:
        await coro_fn()
    except Exception:
        logger.warning("singleton stop failed (non-fatal): %s", label, exc_info=True)


async def start_singleton_services() -> None:
    """拉起 leader 专属单例服务（TutorBot / Cron / MCP）。幂等：已启则跳过。

    被 leader 回调调用：首次当选 + 运行中竞选接管。各服务自身亦幂等（双重保险）。

    C-1/C-3 竞态防护：关停标志统一来自 ``leader.is_shutting_down()``（lifespan shutdown
    第一行经 ``mark_shutting_down()`` 置位，最早）。典型竞态——worker 刚当选、本函数
    正在 await 某个 manager.start 时 Gunicorn SIGTERM 到达 → shutdown 跑 stop，但此时
    start 尚未置位 ``_singletons_started``，旧实现的 stop 早退什么都不停；start 恢复后
    又无条件 ``_singletons_started = True`` → Cron/Bot/MCP 在即将死亡的 worker 上重启、
    无人停止。修法：入口 + 每个 await 后都检查 ``is_shutting_down()``，发现已关停则
    **回滚已启动的服务**（逆序），且**绝不置位** ``_singletons_started``。
    """
    global _singletons_started
    if _singletons_started:
        return
    # 入口即查：关停流程已开始，一个服务都不启动，不置位。
    if is_shutting_down():
        logger.info("Singleton startup skipped: application is shutting down")
        return

    # 已启动的服务（逆序回滚用）。每项是 (stop_coro_factory, label)。
    started: list = []

    if TUTORBOT_ENABLED:
        try:
            from core.bot.manager import get_bot_manager
            bot = get_bot_manager()
            await bot.auto_start_bots()
            # C-1/C-3 检查点：await 恢复后立即查关停标志。
            if is_shutting_down():
                logger.info("Shutdown signaled during TutorBot start; rolling back")
                await _safe_call(bot.stop_all, "bot.stop_all")
                return
            started.append((bot.stop_all, "bot.stop_all"))
            logger.info("TutorBot manager initialized (auto-started configured bots)")
        except Exception as e:
            logger.warning(f"TutorBot startup failed (non-fatal): {e}")

    try:
        from services.cron.service import get_cron_service
        cron = get_cron_service()
        await cron.start()
        if is_shutting_down():
            logger.info("Shutdown signaled during Cron start; rolling back bot+cron")
            await _safe_call(cron.stop, "cron.stop")
            for fn, label in reversed(started):
                await _safe_call(fn, label)
            return
        started.append((cron.stop, "cron.stop"))
        logger.info("Cron service started")
    except Exception as e:
        logger.warning(f"Cron service startup failed (non-fatal): {e}")

    # MCP 不在此处：它不是"全局只需一份"的后台任务（如 Cron/Bot），而是每个 worker
    # 处理各自对话时都要用到的工具连接——leader-only 会导致非 leader worker 的
    # get_mcp_manager() 永远拿不到任何工具。改为每个 worker 在 lifespan() 里独立
    # ensure_started()（见下方 MCP 小节 + lifespan 内对应调用）。

    _singletons_started = True


async def stop_singleton_services() -> None:
    """停止 leader 专属单例服务。幂等：未启则跳过。

    被 leader 回调调用：丢失 leader（锁被别人抢）+ 应用 shutdown。
    """
    global _singletons_started
    if not _singletons_started:
        return

    # 与 start 逆序停止：Cron → Bot（MCP 生命周期已独立于 leader，见 lifespan）
    try:
        from services.cron.service import get_cron_service
        await get_cron_service().stop()
    except Exception:
        pass

    if TUTORBOT_ENABLED:
        try:
            from core.bot.manager import get_bot_manager
            await get_bot_manager().stop_all()
        except Exception:
            pass

    _singletons_started = False


# ---- MCP：每个 worker 独立连接（不走 leader-only）----
# MCP 工具要被"每个 worker 各自处理的对话请求"调用到，性质上更接近 DB 连接池
# （每个 worker 都要有一份能用的连接），而不是 Cron/Bot 那种"全局跑一份就够"的
# 后台任务。放进 leader-only 的 start_singleton_services 会导致：单容器
# gunicorn -w N 下，只有当选 leader 的那个 worker 建立了连接，其余 N-1 个
# worker 的 get_mcp_manager() 永远是空的——落在它们身上的对话请求会静默看不到
# 任何 MCP 工具（dynamic_tools.resolve() 的 pool 为空，不报错）。
_mcp_reload_task: asyncio.Task | None = None


async def _mcp_reload_loop() -> None:
    """轮询 data/mcp.json 签名并同步连接。

    admin 的 POST/DELETE /api/mcp/servers/{name} 只会在受理该 HTTP 请求的那个
    worker 里调用一次 reload()——其余 worker 需要靠这个轮询才能跟上配置变化。
    reload() 内部按 connection_signature() 比对，未变化的 server 不会被重连，
    轮询开销可忽略。
    """
    from core.mcp.manager import get_mcp_manager

    mgr = get_mcp_manager()
    interval = get_settings().mcp.reload_interval_s
    while True:
        await asyncio.sleep(interval)
        try:
            await mgr.reload()
        except Exception:
            logger.warning("MCP periodic reload failed (non-fatal)", exc_info=True)


# ---- 资源水位采样：DB 连接池 + LightRAG 实例池 → Prometheus Gauge（压测/运维观测）----
_resource_sampler_task: asyncio.Task | None = None


async def _sample_resource_gauges() -> None:
    """每 5s 采样 DB 连接池 checkedout 数 + LightRAG 实例池水位 + 整卷磁盘水位，写 Prometheus。

    仅观测用，任何采样失败都静默（SQLite StaticPool/NullPool 无 checkedout、实例池未
    初始化等），绝不影响主链路。worker label 用 pid 区分（多容器/多 worker 各自独立
    pool + 实例池）。
    """
    from core.observability.metrics import (
        DB_POOL_CHECKEDOUT,
        DISK_USED_BYTES,
        LIGHTRAG_INSTANCES,
        LIGHTRAG_IN_USE,
    )
    from core.db.database import engine
    from core.rag.lightrag.instance_pool import _instances, _in_use
    import shutil

    worker_id = os.getenv("BACKEND_WORKER_ID") or str(os.getpid())
    # 整卷水位锚点：取 LightRAG workdir 所在卷（业务数据盘）。resolve 一次，循环里只 disk_usage。
    disk_anchor = str(get_settings().paths.lightrag_workdir)
    while True:
        try:
            # AsyncEngine.sync_engine.pool 是底层 Pool；QueuePool 有 checkedout()，
            # SQLite 的 StaticPool/NullPool 没有该方法 → except 静默跳过。
            DB_POOL_CHECKEDOUT.labels(worker=worker_id).set(
                engine.sync_engine.pool.checkedout()
            )
        except Exception:
            pass
        try:
            LIGHTRAG_INSTANCES.set(len(_instances))
            LIGHTRAG_IN_USE.set(len(_in_use))
        except Exception:
            pass
        try:
            # 廉价 syscall（statfs）；写整卷水位，供磁盘告警观测。
            DISK_USED_BYTES.labels(target="volume").set(shutil.disk_usage(disk_anchor).used)
        except Exception:
            pass
        await asyncio.sleep(5)


# ---- 上下文窗口探测：启动预热（best-effort，不阻塞启动）----
# 对标 DeepTutor 的「配置期探测」分工：探测在请求热路径之外跑（GET /models），结果落
# data/context_window_cache.json，热路径 resolve_effective_window 只同步读缓存、零网络。
# 失败只告警不阻塞启动；缓存已有未过期条目则跳过（避免每重启都打 /models）。
_window_probe_task: asyncio.Task | None = None


async def _warmup_context_window_probe() -> None:
    """启动预热：探测 active profile 的 text+fast 模型真实窗口，写入探测缓存。

    best-effort：任何异常只告警。缓存命中（未过期）则跳过该模型。
    """
    try:
        from core.agentic.window_probe import warmup_probe

        results = await warmup_probe()
        for r in results:
            if r.get("detected"):
                logger.info(
                    "context window probe: model=%s window=%d (cached=%s)",
                    r["model"], r["detected"], r.get("skipped_cached", False),
                )
            elif not r.get("skipped_cached"):
                # 探测失败：供应商 /models 未暴露窗口或不可达，退回 _MODEL_WINDOWS 表（属正常降级）
                logger.debug(
                    "context window probe: model=%s returned no window (will fall back to table/heuristic)",
                    r["model"],
                )
    except Exception:
        logger.warning("context window probe warmup failed (non-fatal)", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing database tables")
    await init_db()

    # Mem0 预热：get_memory() 内部 AsyncMemory.from_config 是同步调用，首次执行时
    # 建连 + CREATE TABLE IF NOT EXISTS memories（mem0 Issue #2755）。不预热则该开销
    # 摊到第一条对话/第一次 GET /memory 上；这里启动期做掉，顺带保证 memories 表
    # 一定存在，不会出现「表有时查得到有时查不到」。best-effort，不阻塞启动。
    try:
        from core.memory.mem0_client import get_memory
        get_memory()
        logger.info("Mem0 warmed up (memories table ensured)")
    except Exception:
        logger.warning("Mem0 warmup failed (non-fatal)", exc_info=True)

    # P4：从 DB 回填当日 Redis 配额键（防 Redis 失联后配额静默重置）。best-effort，不阻塞启动。
    try:
        from core.quota.cost_quota import reconcile_quota_from_db
        n = await reconcile_quota_from_db()
        if n:
            logger.info("cost_quota: reconciled %d keys from DB", n)
    except Exception:
        logger.debug("cost_quota reconcile on startup failed", exc_info=True)

    # 注册 EventBus 处理器（turn 完成后记忆更新；订阅者实现在 core/memory）
    from events.event_bus import get_event_bus
    from core.memory.lifecycle_subscribers import subscribe_memory_handlers
    subscribe_memory_handlers(get_event_bus())
    logger.info("EventBus: registered capability_complete handlers")

    # 尝试初始化 ARQ 任务队列连接池
    await get_arq_pool()

    # M-24：校验 BACKEND_WORKERS 与真实 worker 进程数是否一致（不一致会令 DB 池 /
    # 熔断器 / LRU 缩放公式算偏）。读 WEB_CONCURRENCY 约定；无注入线索（dev/单进程）时跳过。
    from settings import get_settings as _get_settings
    _get_settings().validate_runtime_workers()

    # Leader election: 注册状态翻转回调；leader（含运行中竞选接管）启单例服务。
    from core.leader import try_become_leader, shutdown_leader, register_leader_callbacks
    register_leader_callbacks(
        on_gain=start_singleton_services,
        on_lose=stop_singleton_services,
    )
    # 当选则内部走 on_gain 拉单例；未当选则起竞选 loop，锁可被接管时再走 on_gain
    await try_become_leader()

    # MCP：每个 worker 独立连接 + 启动周期性 reload（与 leader 选举无关，见上方说明）
    global _mcp_reload_task
    try:
        from core.mcp.manager import get_mcp_manager
        await get_mcp_manager().ensure_started()
        logger.info("MCP manager initialized (per-worker)")
    except Exception as e:
        logger.warning(f"MCP manager startup failed (non-fatal): {e}")
    _mcp_reload_task = asyncio.create_task(_mcp_reload_loop())

    # 资源水位采样 task（DB pool / LightRAG 实例池 → Prometheus，压测/运维观测）
    global _resource_sampler_task
    _resource_sampler_task = asyncio.create_task(_sample_resource_gauges())

    # 上下文窗口探测启动预热（best-effort，不阻塞启动；详见 _warmup_context_window_probe）
    global _window_probe_task
    _window_probe_task = asyncio.create_task(_warmup_context_window_probe())

    yield

    # C-1/C-3：shutdown 第一行置位关停标志（最早信号）。此后即便 leader 回调触发
    # start_singleton_services 也会因 is_shutting_down() 早退/回滚，单例不会在死亡
    # worker 上重启。
    mark_shutting_down()

    # Shutdown: final memory flush 已停用（flush cron 生产者死透，巩固走 episodes outbox）
    # （原 flush_all_pending_job enqueue 移除；worker 不再注册该 job）

    # 停止单例服务（leader 经 on_lose 回调）+ 清理 leader 选举资源（cancel loop、释放锁）
    await shutdown_leader()

    # 停止本 worker 的 MCP 连接 + 周期性 reload task
    if _mcp_reload_task is not None:
        _mcp_reload_task.cancel()
        try:
            await _mcp_reload_task
        except (asyncio.CancelledError, Exception):
            pass
        _mcp_reload_task = None
    try:
        from core.mcp.manager import get_mcp_manager
        await get_mcp_manager().shutdown()
    except Exception:
        pass

    # 取消资源水位采样 task（须在 close_db 前，否则采样读到已释放的 engine 报错）。
    # _resource_sampler_task 在函数顶部 startup 段已 global 声明（line 297），整个
    # lifespan 作用域生效，此处无需重复声明（重复 global 且前面已赋值 → SyntaxError）。
    if _resource_sampler_task is not None:
        _resource_sampler_task.cancel()
        try:
            await _resource_sampler_task
        except (asyncio.CancelledError, Exception):
            pass
        _resource_sampler_task = None

    # 取消上下文窗口探测预热 task（best-effort；探测自带超时，cancel 仅作干净退出）。
    if _window_probe_task is not None:
        _window_probe_task.cancel()
        try:
            await _window_probe_task
        except (asyncio.CancelledError, Exception):
            pass
        _window_probe_task = None

    # 停止 LightRAG idle reaper（若本 worker 因检索/索引懒启动过它）。须在 close_db 前，
    # 避免 reaper 在 DB 已释放后还去 finalize 实例。
    try:
        from core.rag.lightrag.instance_pool import stop_idle_reaper
        await stop_idle_reaper()
    except Exception:
        logger.warning("stop_idle_reaper failed (non-fatal)", exc_info=True)

    logger.info("Application shutdown – closing database pool")
    await close_db()
    await close_arq_pool()


app = FastAPI(
    title="课程学习Agent",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
    # 生产环境不暴露 API 文档端点（开发保留）
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json",
)


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(title=app.title, version="1.0.0", routes=app.routes)
    schema.setdefault("components", {})["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer"}
    }
    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    # 错误信封（HTTP 侧）：detail 保持字符串（前端 readErrorMessage 只读 detail/message），
    # 带学生可报障的编号（request_id 前 8 位——Sentry 500 页渲染 event_id 同款模式）
    from core.observability.context import get_context_fields

    request_id = get_context_fields().get("request_id") or ""
    ref = request_id[:8]
    detail = f"服务器内部错误（编号 {ref}）" if ref else "服务器内部错误"
    return JSONResponse(
        status_code=500,
        content={"detail": detail, "code": "internal", "ref": ref},
        headers={"X-Request-Id": request_id} if request_id else None,
    )


app.add_middleware(SlowAPIMiddleware)

from core.observability.middleware import ObservabilityMiddleware  # noqa: E402
app.add_middleware(ObservabilityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # X-Request-Id 需显式放行，跨域前端才能读到（学生报障编号的回传通道）
    expose_headers=["X-Request-Id"],
)

# gunicorn 多 worker：worker 退出时清理本进程的 multiproc 文件，防 'all' 模式的 pid-labeled
# Gauge（leader/pool）留下幽灵样本。仅 multiprocess 模式生效；Counter/Histogram 是累积值，
# prometheus_client 本就保留已退出进程的文件（代表已完成的真实工作量），无需清理。
def _mark_prometheus_process_dead() -> None:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        try:
            from prometheus_client import multiprocess

            multiprocess.mark_process_dead(os.getpid())
        except Exception:  # noqa: BLE001 — best-effort 清理，失败不影响进程退出
            pass


import atexit

atexit.register(_mark_prometheus_process_dead)

# Instrumentator 仅 instrument（埋 HTTP 指标到 registry），不再用内置 expose：gunicorn -w 4 下
# 内置 /metrics 只反映接到请求的那个 worker（~1/4 流量）。改为自定义 /metrics：multiprocess
# 模式用 MultiProcessCollector 聚合所有 worker 写入 multiproc 目录的文件，单进程用默认 REGISTRY
# （与旧 expose 行为等价）。excluded_handlers 不含 /metrics（本端点自己处理，避免自指计数）。
Instrumentator(
    excluded_handlers=["/api/health", "/metrics"],
).instrument(app)


@app.get("/metrics", include_in_schema=False)
async def _prometheus_metrics() -> Response:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        CollectorRegistry,
        generate_latest,
        multiprocess,
    )

    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        registry: object = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY  # 单进程：默认 registry（行为等价旧 Instrumentator.expose）
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
app.include_router(question_router, prefix="/api")
app.include_router(question_notebook_router, prefix="/api")
app.include_router(transcribe_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(teacher_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(lightrag_router, prefix="/api")
app.include_router(courses_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(memory_router, prefix="/api")
app.include_router(run_router, prefix="/api")
app.include_router(skills_router, prefix="/api")
app.include_router(skill_knowledge_router, prefix="/api")
app.include_router(mcp_router, prefix="/api")
app.include_router(bot_router, prefix="/api")
app.include_router(llm_router, prefix="/api")
app.include_router(search_config_router, prefix="/api")
app.include_router(user_llm_router, prefix="/api")
from api.ops import router as ops_router
app.include_router(ops_router, prefix="/api")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_STORE_DIR, exist_ok=True)


