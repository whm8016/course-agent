import os
import sys

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))


def _sync_langsmith_env() -> None:
    """Align with langsmith.utils.tracing_is_enabled(): value must be exactly 'true' (lowercase)."""
    truthy = ("1", "true", "yes", "on")
    for key in (
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
    ):
        raw = os.getenv(key)
        if raw is None or str(raw).strip() == "":
            continue
        if str(raw).strip().lower() in truthy:
            os.environ[key] = "true"
    # Many docs use LANGSMITH_TRACING; SDK also checks *_TRACING_V2 first.
    if os.getenv("LANGSMITH_TRACING") == "true" or os.getenv("LANGCHAIN_TRACING") == "true":
        os.environ.setdefault("LANGSMITH_TRACING_V2", "true")
    ls_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    lc_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if ls_key and not lc_key:
        os.environ["LANGCHAIN_API_KEY"] = ls_key
    if lc_key and not ls_key:
        os.environ["LANGSMITH_API_KEY"] = lc_key
    try:
        from langsmith.utils import get_env_var  # type: ignore[import-untyped]

        get_env_var.cache_clear()
    except Exception:
        pass


_sync_langsmith_env()

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

TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen-plus")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen-vl-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ---------------------------------------------------------------------------
# RAG tuning
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "4"))
INGEST_CHUNK_SIZE = int(os.getenv("INGEST_CHUNK_SIZE", "900"))
INGEST_CHUNK_OVERLAP = int(os.getenv("INGEST_CHUNK_OVERLAP", "60"))

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
# FAQ 高频问题
# ---------------------------------------------------------------------------
# 问题被问到达此次数后，开始缓存答案（0 = 不缓存）
FAQ_CACHE_THRESHOLD = int(os.getenv("FAQ_CACHE_THRESHOLD", "3"))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
_JWT_DEFAULT = "dev-secret-change-in-production"
JWT_SECRET = os.getenv("JWT_SECRET", _JWT_DEFAULT)
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "72"))

if JWT_SECRET == _JWT_DEFAULT:
    import warnings
    warnings.warn(
        "JWT_SECRET is using the insecure default value! "
        "Set a strong JWT_SECRET environment variable before deploying to production.",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
_origins_raw = os.getenv("ALLOWED_ORIGINS", "").strip()
if not _origins_raw or _origins_raw == "*":
    ALLOWED_ORIGINS: list[str] = ["*"]
    if _origins_raw != "*" and not _origins_raw:
        import warnings
        warnings.warn(
            "ALLOWED_ORIGINS is not set — defaulting to '*' (allow all). "
            "Set explicit origins (e.g. 'https://example.com') for production.",
            stacklevel=1,
        )
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
# agentic_pipeline._run_rag：aquery 返回文本写入 tool trace 前的最大字符数（过长会截断）
LIGHTRAG_AGENTIC_RAG_MAX_CHARS = int(os.getenv("LIGHTRAG_AGENTIC_RAG_MAX_CHARS", "10000"))
# LightRAG 默认会开 rerank；未配置 rerank 模型时会告警且可能长时间阻塞，故默认关闭
LIGHTRAG_ENABLE_RERANK = os.getenv("LIGHTRAG_ENABLE_RERANK", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# 管理端 LightRAG 摄入时，将 SentenceSplitter 后的文本块落盘（与 LightRAG workspace 并列子目录）
LIGHTRAG_SAVE_INGEST_CHUNKS = os.getenv("LIGHTRAG_SAVE_INGEST_CHUNKS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
LIGHTRAG_INGEST_CHUNKS_SUBDIR = os.getenv("LIGHTRAG_INGEST_CHUNKS_SUBDIR", "ingest_chunks")
LIGHTRAG_INGEST_CHUNKS_SNAPSHOT = os.getenv("LIGHTRAG_INGEST_CHUNKS_SNAPSHOT", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Admin / Knowledge Base Store  /lightrag的
# ---------------------------------------------------------------------------
KB_STORE_DIR = os.getenv("KB_STORE_DIR", os.path.join(BASE_DIR, "kb_store"))
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
MAX_KB_UPLOAD_MB = int(os.getenv("MAX_KB_UPLOAD_MB", "50"))

# ---------------------------------------------------------------------------
# Question coordinator (AgentCoordinator)
# ---------------------------------------------------------------------------
QUESTION_LOG_DIR = os.getenv(
    "QUESTION_LOG_DIR",
    os.path.join(BASE_DIR, "logs", "question"),
)
QUESTION_TOOL_WEB_SEARCH = os.getenv("QUESTION_TOOL_WEB_SEARCH", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QUESTION_TOOL_RAG = os.getenv("QUESTION_TOOL_RAG", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QUESTION_TOOL_CODE_EXECUTION = os.getenv("QUESTION_TOOL_CODE_EXECUTION", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
QUESTION_DEFAULT_TOOL_FLAGS: dict[str, bool] = {
    "web_search": QUESTION_TOOL_WEB_SEARCH,
    "rag": QUESTION_TOOL_RAG,
    "code_execution": QUESTION_TOOL_CODE_EXECUTION,
}

# ---------------------------------------------------------------------------
# LlamaParse（图像 PDF / 扫描件解析）
# ---------------------------------------------------------------------------
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY", "")

# LlamaIndex 向量库根目录（每个 course 一个子目录，其下 llamaindex_storage/）
LLAMA_INDEX_KB_ROOT = os.getenv(
    "LLAMA_INDEX_KB_ROOT",
    os.path.join(BASE_DIR, "data", "knowledge_bases")
)
# POST /api/chat/lightrag → agentic_pipeline 里「知识检索」用哪套引擎（与 body tools 里的 rag/llamaindex_rag 对齐）
#   lightrag   — 使用 LightRAG（默认）
#   llamaindex — 使用 LlamaIndex 向量库（需已对该 course_id 建 llamaindex 索引）
# 兼容旧变量：AGENTIC_KB_TOOL=llamaindex_rag 等价于 AGENTIC_RAG_BACKEND=llamaindex
_arg_backend = os.getenv("AGENTIC_RAG_BACKEND", "").strip().lower()
_legacy_kb = os.getenv("AGENTIC_KB_TOOL", "").strip().lower()
if _arg_backend in ("lightrag", "llamaindex"):
    AGENTIC_RAG_BACKEND: str = _arg_backend
elif _legacy_kb == "llamaindex_rag":
    AGENTIC_RAG_BACKEND = "llamaindex"
else:
    AGENTIC_RAG_BACKEND = "lightrag"
QUESTION_USE_LLAMAINDEX = os.getenv("QUESTION_USE_LLAMAINDEX", "True").strip().lower() in (
    "1", "true", "yes", "on",
)

LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING", "False").strip().lower() in (
    "1", "true", "yes", "on",
)
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT", "")
