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
    )
)
logging.basicConfig(level=logging.INFO, handlers=[_log_handler])

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.admin import router as admin_router
from api.auth import router as auth_router
from api.chat import router as chat_router
from api.courses import router as courses_router
from api.lightrag import router as lightrag_router
from api.upload import router as upload_router
from api.sessions import router as sessions_router
from api.sse import router as sse_router
from config import UPLOAD_DIR, ALLOWED_ORIGINS, REDIS_URL, KB_STORE_DIR
from core.database import init_db, close_db
from core.limiter import limiter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing database tables")
    await init_db()
    yield
    logger.info("Application shutdown – closing database pool")
    await close_db()


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

app.include_router(admin_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(lightrag_router, prefix="/api")
app.include_router(courses_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(sse_router, prefix="/api")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_STORE_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/api/health")
async def health():
    checks: dict[str, str] = {}

    # DB check
    try:
        from core.database import engine
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

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )
