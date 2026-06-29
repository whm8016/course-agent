"""配置兼容 shim —— 全部导出名保留，读取自 settings（单一事实源）。

本文件已从「平铺全局变量 + os.getenv」降级为薄 shim：真实配置在 backend/settings/
（pydantic-settings，含类型校验 / SecretStr / catalog 拍平 / prod 安全门）。

存量 `from config import XXX` 无需任何改动；新增配置请去 settings/base.py 加字段。
历史名（DASHSCOPE_* 等）作别名保留以维持向后兼容。
"""
from __future__ import annotations

import os as _os
from settings import BASE_DIR, get_settings

_s = get_settings()


# ---------------------------------------------------------------------------
# 日志 / 环境
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _s.log_level
ENVIRONMENT: str = _s.environment
LLM_TIMEOUT_SEC: int = _s.llm_timeout_sec
FAQ_CACHE_THRESHOLD: int = _s.faq_cache_threshold

# ---------------------------------------------------------------------------
# RAG backend
# ---------------------------------------------------------------------------
RAG_BACKEND: str = _s.rag_backend

# ---------------------------------------------------------------------------
# LLM / 多供应商（catalog 优先 → env 兜底）
# ---------------------------------------------------------------------------
LLM_BINDING: str = _s.llm_binding
DASHSCOPE_API_KEY: str = _s.dashscope_api_key.get_secret_value()
DASHSCOPE_BASE_URL: str = _s.dashscope_base_url
LLM_API_VERSION: str = _s.llm_api_version
TEXT_MODEL: str = _s.text_model
FAST_MODEL: str = _s.fast_model
VISION_MODEL: str = _s.vision_model
EMBEDDING_MODEL: str = _s.embedding_model
EMBEDDING_API_KEY: str = _s.embedding_api_key.get_secret_value()
EMBEDDING_BASE_URL: str = _s.embedding_base_url
FALLBACK_API_KEY: str = _s.fallback_api_key.get_secret_value()
FALLBACK_BASE_URL: str = _s.fallback_base_url
FALLBACK_MODEL: str = _s.fallback_model

# ---------------------------------------------------------------------------
# LightRAG 分角色 LLM（1.5.4+）
# ---------------------------------------------------------------------------
EXTRACT_MODEL: str = _s.extract_model
KEYWORD_MODEL: str = _s.keyword_model

# ---------------------------------------------------------------------------
# RAG tuning
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = _s.chunk_size
CHUNK_OVERLAP: int = _s.chunk_overlap
TOP_K: int = _s.top_k
INGEST_CHUNK_SIZE: int = _s.ingest_chunk_size
INGEST_CHUNK_OVERLAP: int = _s.ingest_chunk_over_lap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR: str = _s.upload_dir
KNOWLEDGE_DIR: str = _s.knowledge_dir
VECTORSTORE_DIR: str = _s.vectorstore_dir
DB_PATH: str = _s.db_path

# ---------------------------------------------------------------------------
# PostgreSQL + Redis
# ---------------------------------------------------------------------------
DATABASE_URL: str = _s.database_url.get_secret_value()
REDIS_URL: str = _s.redis_url.get_secret_value()
DB_POOL_SIZE: int = _s.db_pool_size
DB_MAX_OVERFLOW: int = _s.db_max_overflow

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
JWT_SECRET: str = _s.jwt_secret.get_secret_value()
JWT_EXPIRE_HOURS: int = _s.jwt_expire_hours
ALLOWED_ORIGINS: list[str] = _s.allowed_origins
ADMIN_USERNAME: str = _s.admin_username

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = _s.max_upload_mb

# ---------------------------------------------------------------------------
# LightRAG
# ---------------------------------------------------------------------------
LIGHTRAG_ENABLED: bool = _s.lightrag_enabled
LIGHTRAG_WORKDIR: str = _s.lightrag_workdir
LIGHTRAG_QUERY_MODE: str = _s.lightrag_query_mode
LIGHTRAG_TOP_K: int = _s.lightrag_top_k
LIGHTRAG_TIMEOUT_SEC: int = _s.lightrag_timeout_sec
LIGHTRAG_EMBEDDING_DIM: int = _s.lightrag_embedding_dim
LIGHTRAG_AUTO_INDEX_TTL_SEC: int = _s.lightrag_auto_index_ttl_sec
LIGHTRAG_STREAM_CONTEXT_LIMIT: int = _s.lightrag_stream_context_limit
LIGHTRAG_STREAM_CONTEXT_MAX_CHARS: int = _s.lightrag_stream_context_max_chars
LIGHTRAG_AGENTIC_RAG_MAX_CHARS: int = _s.lightrag_agentic_rag_max_chars
LIGHTRAG_ENABLE_RERANK: bool = _s.lightrag_enable_rerank
LIGHTRAG_SAVE_INGEST_CHUNKS: bool = _s.lightrag_save_ingest_chunks
LIGHTRAG_INGEST_CHUNKS_SUBDIR: str = _s.lightrag_ingest_chunks_subdir
LIGHTRAG_INGEST_CHUNKS_SNAPSHOT: bool = _s.lightrag_ingest_chunks_snapshot
LIGHTRAG_INGEST_BATCH_SIZE: int = _s.lightrag_ingest_batch_size
LIGHTRAG_MAX_ASYNC: int = _s.lightrag_max_async
# 多 worker 下调整 LRU 缓存容量: 总容量 / worker 数
_WORKERS = int(_os.getenv("BACKEND_WORKERS", "4"))
LIGHTRAG_LRU_CAPACITY: int = max(2, _s.lightrag_lru_capacity // _WORKERS)

# ---------------------------------------------------------------------------
# Admin / Knowledge Base Store
# ---------------------------------------------------------------------------
KB_STORE_DIR: str = _s.kb_store_dir
MAX_KB_UPLOAD_MB: int = _s.max_kb_upload_mb

# ---------------------------------------------------------------------------
# Question coordinator (AgentCoordinator)
# ---------------------------------------------------------------------------
QUESTION_LOG_DIR: str = _s.question_log_dir
QUESTION_TOOL_WEB_SEARCH: bool = _s.question_tool_web_search
QUESTION_TOOL_RAG: bool = _s.question_tool_rag
QUESTION_TOOL_CODE_EXECUTION: bool = _s.question_tool_code_execution
QUESTION_DEFAULT_TOOL_FLAGS: dict[str, bool] = {
    "web_search": QUESTION_TOOL_WEB_SEARCH,
    "rag": QUESTION_TOOL_RAG,
    "code_execution": QUESTION_TOOL_CODE_EXECUTION,
}

# ---------------------------------------------------------------------------
# LlamaParse（图像 PDF / 扫描件解析）
# ---------------------------------------------------------------------------
LLAMA_CLOUD_API_KEY: str = _s.llama_cloud_api_key.get_secret_value()
LLAMA_INDEX_KB_ROOT: str = _s.llama_index_kb_root
AGENTIC_RAG_BACKEND: str = _s.agentic_rag_backend
QUESTION_USE_LLAMAINDEX: bool = _s.question_use_llamaindex

LANGSMITH_TRACING: bool = _s.langsmith_tracing
LANGSMITH_API_KEY: str = _s.langsmith_api_key.get_secret_value()
LANGSMITH_PROJECT: str = _s.langsmith_project

# ---------------------------------------------------------------------------
# TutorBot 社交平台集成
# ---------------------------------------------------------------------------
TUTORBOT_ENABLED: bool = _s.tutorbot_enabled
TUTORBOT_WORKSPACE_DIR: str = _s.tutorbot_workspace_dir

# QQ Bot (botpy SDK)
QQ_BOT_ENABLED: bool = _s.qq_bot_enabled
QQ_APP_ID: str = _s.qq_app_id.get_secret_value()
QQ_SECRET: str = _s.qq_secret.get_secret_value()
QQ_ALLOW_FROM: str = _s.qq_allow_from

# Feishu Bot (lark-oapi SDK, WebSocket)
FEISHU_BOT_ENABLED: bool = _s.feishu_bot_enabled
FEISHU_APP_ID: str = _s.feishu_app_id.get_secret_value()
FEISHU_APP_SECRET: str = _s.feishu_app_secret.get_secret_value()
FEISHU_ENCRYPT_KEY: str = _s.feishu_encrypt_key.get_secret_value()
FEISHU_VERIFICATION_TOKEN: str = _s.feishu_verification_token.get_secret_value()
FEISHU_ALLOW_FROM: str = _s.feishu_allow_from

# Heartbeat
TUTORBOT_HEARTBEAT_ENABLED: bool = _s.tutorbot_heartbeat_enabled
TUTORBOT_HEARTBEAT_INTERVAL_SEC: int = _s.tutorbot_heartbeat_interval_sec

# ---------------------------------------------------------------------------
# Search（web search 服务）
# ---------------------------------------------------------------------------
SEARCH_ENABLED: bool = _s.search_enabled
SEARCH_PROVIDER: str = _s.search_provider
SEARCH_API_KEY: str = _s.search_api_key.get_secret_value()
SEARCH_BASE_URL: str = _s.search_base_url
SEARCH_MAX_RESULTS: int = _s.search_max_results
SEARCH_PROXY: str = _s.search_proxy
SEARCH_CONFIG_PATH: str = _s.search_config_path

# ---------------------------------------------------------------------------
# MCP（Model Context Protocol）
# ---------------------------------------------------------------------------
MCP_CONFIG_PATH: str = _s.mcp_config_path
MCP_SESSIONS_DIR: str = _s.mcp_sessions_dir

# ---------------------------------------------------------------------------
# Image ingestion（LLamaIndex 图片提取）
# ---------------------------------------------------------------------------
IMAGE_INGEST_MIN_PX: int = _s.image_ingest_min_px
IMAGE_INGEST_MIN_AREA: int = _s.image_ingest_min_area
IMAGE_INGEST_MAX_PER_FILE: int = _s.image_ingest_max_per_file
IMAGE_INGEST_SEMAPHORE: int = _s.image_ingest_semaphore
IMAGE_INGEST_WMF_MIN_BLOB: int = _s.image_ingest_wmf_min_blob

# ---------------------------------------------------------------------------
# Embedding bridge
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = _s.embedding_dim
EMBEDDING_BATCH_SIZE: int = _s.embedding_batch_size

# ---------------------------------------------------------------------------
# LightRAG SAFE_* 参数
# ---------------------------------------------------------------------------
LIGHTRAG_SAFE_TOP_K: int = _s.lightrag_safe_top_k
LIGHTRAG_CHUNK_TOP_K: int = _s.lightrag_chunk_top_k
LIGHTRAG_MAX_TOTAL_TOKENS: int = _s.lightrag_max_total_tokens
LIGHTRAG_MAX_ENTITY_TOKENS: int = _s.lightrag_max_entity_tokens
LIGHTRAG_MAX_RELATION_TOKENS: int = _s.lightrag_max_relation_tokens
LIGHTRAG_MAX_HISTORY_MESSAGES: int = _s.lightrag_max_history_messages
LIGHTRAG_MAX_HISTORY_CHARS: int = _s.lightrag_max_history_chars
LIGHTRAG_LLM_SYSTEM_MAX_CHARS: int = _s.lightrag_llm_system_max_chars

# ---------------------------------------------------------------------------
# Output cards
# ---------------------------------------------------------------------------
OUTPUT_CARDS_PATH: str = _s.output_cards_path

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
TESTING: bool = _s.testing

# ---------------------------------------------------------------------------
# BASE_DIR（向后兼容：旧 config.py 有此导出）
# ---------------------------------------------------------------------------
BASE_DIR = BASE_DIR  # noqa: F811  (从 settings 导入的同一对象)
