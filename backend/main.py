import sys
import os
import logging

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
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from api.run import router as run_router
from api.question import router as question_router
from api.question_notebook import router as question_notebook_router
from api.llama_rag import router as llama_rag_router
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
from config import UPLOAD_DIR, ALLOWED_ORIGINS, REDIS_URL, KB_STORE_DIR, TUTORBOT_ENABLED
from core.db.database import init_db, close_db
from core.db.limiter import limiter
from core.arq_pool import get_arq_pool, close_arq_pool
from core.rag.cache import init_rag_cache, get_cache_stats as get_rag_cache_stats
from core.rag.rag import set_rag_cache
from core.llm.llm import get_llm_circuit_state, reset_llm_circuit_breaker

logger = logging.getLogger(__name__)


# ---- EventBus：turn 完成后的记忆更新订阅者 ----
async def _on_capability_complete(event) -> None:
    """CAPABILITY_COMPLETE：批量 enqueue 到 flush_manager（debounce + 批量刷新）+ L2 摘要压缩。"""
    try:
        from core.memory.flush_manager import get_flush_manager

        flush_mgr = get_flush_manager()

        # 从 event 中提取 session_id
        session_id = getattr(event, "session_id", "") or event.metadata.get("session_id", "") or ""

        await flush_mgr.enqueue(
            user_id=event.user_id,
            session_id=session_id,
            course_id=event.course_id,
            user_msg=event.user_message,
            assistant_msg=event.agent_output,
        )
    except Exception:
        logger.warning("EventBus: memory enqueue failed", exc_info=True)

    # L2: 触发 session summary 增量压缩（异步，不阻塞主链路）
    session_id = getattr(event, "session_id", "") or event.metadata.get("session_id", "") or ""
    if session_id:
        import asyncio
        asyncio.create_task(_maybe_compress_summary(session_id))


async def _maybe_compress_summary(session_id: str) -> None:
    """异步执行 L2 摘要压缩（不阻塞主链路）。"""
    try:
        from core.memory.session_summary import get_summary_manager
        from core.db.database import AsyncSessionLocal

        summary_mgr = get_summary_manager()
        async with AsyncSessionLocal() as db:
            compressed = await summary_mgr.maybe_compress(db, session_id)
            if compressed:
                logger.info("[L2] session summary compressed session=%s", session_id)
    except Exception:
        logger.warning("[L2] session summary compress failed session=%s", session_id, exc_info=True)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing database tables")
    await init_db()

    # 注册 EventBus 处理器（turn 完成后记忆更新）
    from events.event_bus import get_event_bus, EventType
    get_event_bus().subscribe(EventType.CAPABILITY_COMPLETE, _on_capability_complete)
    logger.info("EventBus: registered capability_complete handlers")

    # 尝试初始化 ARQ 任务队列连接池（Redis 不可用时跳过，降级为 BackgroundTasks）
    await get_arq_pool()
    # 初始化 RAG 缓存并注入到 rag.py 检索路径
    try:
        rag_cache = await init_rag_cache(REDIS_URL, ttl_seconds=3600)
        set_rag_cache(rag_cache)
        logger.info("RAG cache initialized and wired to retrieve path")
    except Exception as e:
        logger.warning(f"RAG cache initialization failed: {e}")

    # Leader election: 只有 leader worker 启动单例服务 (Cron/Bot/MCP)
    from core.leader import try_become_leader, shutdown_leader
    _is_leader = await try_become_leader()

    # 仅 leader worker 启 TutorBot
    if _is_leader and TUTORBOT_ENABLED:
        try:
            from core.bot.manager import get_bot_manager
            bot_mgr = get_bot_manager()
            await bot_mgr.auto_start_bots()
            logger.info("TutorBot manager initialized (auto-started configured bots)")
        except Exception as e:
            logger.warning(f"TutorBot startup failed (non-fatal): {e}")

    # 仅 leader worker 启 Cron 调度服务
    if _is_leader:
        try:
            from services.cron.service import get_cron_service
            await get_cron_service().start()
            logger.info("Cron service started")
        except Exception as e:
            logger.warning(f"Cron service startup failed (non-fatal): {e}")

    # 仅 leader worker 启 MCP 连接管理器
    if _is_leader:
        try:
            from core.mcp.manager import get_mcp_manager
            await get_mcp_manager().ensure_started()
            logger.info("MCP manager initialized")
        except Exception as e:
            logger.warning(f"MCP manager startup failed (non-fatal): {e}")

    yield

    # Shutdown: 通知 ARQ worker 做 final memory flush（Producer-Consumer 模式）
    try:
        pool = await get_arq_pool()
        if pool is not None:
            await pool.enqueue_job("flush_all_pending_job")
            logger.info("Enqueued flush_all_pending_job to ARQ worker")
    except Exception as e:
        logger.warning(f"Enqueue flush_all_pending_job failed: {e}")

    # 仅 leader worker 停止单例服务
    if _is_leader:
        # 关闭 MCP 连接
        try:
            from core.mcp.manager import get_mcp_manager
            await get_mcp_manager().shutdown()
        except Exception:
            pass

        # 停止 Cron 服务
        try:
            from services.cron.service import get_cron_service
            await get_cron_service().stop()
        except Exception:
            pass

        # Stop TutorBot if running
        if TUTORBOT_ENABLED:
            try:
                from core.bot.manager import get_bot_manager
                await get_bot_manager().stop_all()
            except Exception:
                pass

        # 清理 leader 选举资源
        await shutdown_leader()

    logger.info("Application shutdown – closing database pool")
    await close_db()
    await close_arq_pool()


app = FastAPI(
    title="课程学习Agent",
    lifespan=lifespan,
    swagger_ui_parameters={"persistAuthorization": True},
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
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
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
)

Instrumentator(
    excluded_handlers=["/api/health", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
app.include_router(question_router, prefix="/api")
app.include_router(question_notebook_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(teacher_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(llama_rag_router, prefix="/api")
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
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_STORE_DIR, exist_ok=True)


@app.get("/api/health")
async def health():
    checks: dict[str, str] = {}

    # DB check
    try:
        from core.db.database import engine
        from sqlalchemy import text as sa_text
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Redis check
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # LLM check：仅验证 API key 已配置；不做实际调用（避免消耗 token / 拖慢探针）
    try:
        from config import DASHSCOPE_API_KEY
        if DASHSCOPE_API_KEY:
            checks["llm"] = "ok (api_key configured)"
        else:
            checks["llm"] = "error: DASHSCOPE_API_KEY not set"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    all_ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )


@app.get("/api/health/detailed")
async def health_detailed():
    """
    详细健康检查（包含熔断器和缓存状态）

    用于运维监控和调试
    """
    checks: dict[str, str] = {}
    details: dict = {}

    # DB check
    try:
        from core.db.database import engine
        from sqlalchemy import text as sa_text
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Redis check
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # LLM check
    try:
        from config import DASHSCOPE_API_KEY
        if DASHSCOPE_API_KEY:
            checks["llm"] = "ok (api_key configured)"
            details["llm_circuit_breaker"] = get_llm_circuit_state()
        else:
            checks["llm"] = "error: DASHSCOPE_API_KEY not set"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    # RAG 缓存状态
    details["rag_cache"] = get_rag_cache_stats().to_dict()

    all_ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
            "details": details,
        },
    )


@app.post("/api/admin/circuit-breaker/reset")
async def reset_circuit_breaker():
    """
    重置 LLM 熔断器（运维操作）

    当熔断器长时间处于 OPEN 状态时，可以手动重置
    """
    reset_llm_circuit_breaker()
    return {"message": "LLM circuit breaker reset successfully", "state": get_llm_circuit_state()}
