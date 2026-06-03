import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))  
from pythonjsonlogger import jsonlogger

_log_handler = logging.StreamHandler()
_log_handler.setFormatter(
    jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
        json_ensure_ascii=False,
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from api.deep_research import router as deep_research_router
from api.deep_solve import router as deep_solve_router
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
from api.sse import router as sse_router
from api.skills import router as skills_router
from config import UPLOAD_DIR, ALLOWED_ORIGINS, REDIS_URL, KB_STORE_DIR
from core.db.database import init_db, close_db
from core.db.limiter import limiter
from core.arq_pool import get_arq_pool, close_arq_pool
from core.rag.cache import init_rag_cache, get_cache_stats as get_rag_cache_stats
from core.llm.llm import get_llm_circuit_state, reset_llm_circuit_breaker

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing database tables")
    await init_db()
    # 尝试初始化 ARQ 任务队列连接池（Redis 不可用时跳过，降级为 BackgroundTasks）
    await get_arq_pool()
    # 初始化 RAG 缓存
    try:
        await init_rag_cache(REDIS_URL, ttl_seconds=3600)
        logger.info("RAG cache initialized")
    except Exception as e:
        logger.warning(f"RAG cache initialization failed: {e}")
    yield
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
app.include_router(sse_router, prefix="/api")
app.include_router(memory_router, prefix="/api")
app.include_router(deep_research_router, prefix="/api")
app.include_router(deep_solve_router, prefix="/api")
app.include_router(skills_router, prefix="/api")
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
