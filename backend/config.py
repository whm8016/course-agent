import os
import sys

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# RAG backend selection
# ---------------------------------------------------------------------------
_raw_rag = os.getenv("RAG_BACKEND", "").strip().lower()
if _raw_rag in ("chroma", "fs"):
    RAG_BACKEND: str = _raw_rag
else:
    RAG_BACKEND = "fs" if sys.platform == "win32" else "chroma"

# ---------------------------------------------------------------------------
# LLM / DashScope
# ---------------------------------------------------------------------------
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen-max")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-max")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ---------------------------------------------------------------------------
# RAG tuning
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
KNOWLEDGE_DIR = os.getenv("KNOWLEDGE_DIR", os.path.join(BASE_DIR, "knowledge"))
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))
DB_PATH = os.getenv("DB_PATH", os.path.join(BASE_DIR, "data", "sessions.db"))

# ---------------------------------------------------------------------------
# PostgreSQL + Redis (high-concurrency stack)
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/course_agent",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if _origins_raw.strip() == "*":
    ALLOWED_ORIGINS: list[str] = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_raw.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------
LIGHTRAG_ENABLED = os.getenv("LIGHTRAG_ENABLED", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LIGHTRAG_WORKDIR = os.getenv("LIGHTRAG_WORKDIR", os.path.join(BASE_DIR, "lightrag_store"))
LIGHTRAG_QUERY_MODE = os.getenv("LIGHTRAG_QUERY_MODE", "mix")
LIGHTRAG_TOP_K = int(os.getenv("LIGHTRAG_TOP_K", "20"))
LIGHTRAG_TIMEOUT_SEC = int(os.getenv("LIGHTRAG_TIMEOUT_SEC", "120"))
LIGHTRAG_EMBEDDING_DIM = int(os.getenv("LIGHTRAG_EMBEDDING_DIM", "1024"))
LIGHTRAG_AUTO_INDEX_TTL_SEC = int(os.getenv("LIGHTRAG_AUTO_INDEX_TTL_SEC", "120"))
LIGHTRAG_STREAM_CONTEXT_LIMIT = int(os.getenv("LIGHTRAG_STREAM_CONTEXT_LIMIT", "4"))
LIGHTRAG_STREAM_CONTEXT_MAX_CHARS = int(os.getenv("LIGHTRAG_STREAM_CONTEXT_MAX_CHARS", "800"))
# LightRAG 默认会开 rerank；未配置 rerank 模型时会告警且可能长时间阻塞，故默认关闭
LIGHTRAG_ENABLE_RERANK = os.getenv("LIGHTRAG_ENABLE_RERANK", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
