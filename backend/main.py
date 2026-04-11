import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.courses import router as courses_router
from api.upload import router as upload_router
from api.sessions import router as sessions_router
from config import UPLOAD_DIR, ALLOWED_ORIGINS, REDIS_URL
from core.database import init_db, close_db

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup – initializing database tables")
    await init_db()
    yield
    logger.info("Application shutdown – closing database pool")
    await close_db()


app = FastAPI(title="课程学习Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(courses_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")

os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
