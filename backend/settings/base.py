"""集中式配置（pydantic-settings）—— 全项目单一事实源。

设计要点：
- 字段名采用 snake_case，对齐历史 env 名（pydantic-settings 大小写不敏感，
  dashscope_api_key 自动匹配 DASHSCOPE_API_KEY），零行为变化。
- secrets 用 SecretStr，print(get_settings()) 不泄密；shim 层 get_secret_value() 还原 str。
- bool 统一经 _truthy（复刻旧 config.py 的 ("1","true","yes","on") 约定）。
- model_catalog 拍平（catalog 优先 → env 兜底）、LangSmith env 同步、JWT/CORS prod 校验、
  RAG_BACKEND 平台判断、AGENTIC_RAG_BACKEND legacy alias —— 全部原样移植自旧 config.py。
- 实例化即校验 → fail-fast（错类型/坏 prod 配置当场 ValidationError/RuntimeError）。

get_settings() 为 lru_cache 单例；config.py 已降级为读取本模块的零破坏 shim。
"""
from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from typing import Annotated, Any

from pydantic import BeforeValidator, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 公共类型：bool 复刻旧 config.py 的 truthy 约定；str secret 去首尾空白
# ---------------------------------------------------------------------------
def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _strip(v: Any) -> Any:
    return v.strip() if isinstance(v, str) else v


def _split_origins(v: Any) -> Any:
    """逗号分隔 → list（复刻旧 config.py：空/* → ["*"]）。NoDecode 防止 JSON 解析。"""
    if isinstance(v, str):
        raw = v.strip()
        if not raw or raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]
    return v


TruthyBool = Annotated[bool, BeforeValidator(_truthy)]
StrippedSecret = Annotated[SecretStr, BeforeValidator(_strip)]
OriginsList = Annotated[list[str], NoDecode, BeforeValidator(_split_origins)]


# ---------------------------------------------------------------------------
# LangSmith env 同步（原样移植 config._sync_langsmith_env）
#  必须在 Settings 读取 LANGSMITH_* 前跑：归一化 truthy、互填 *_TRACING_V2、
#   互填 LANGSMITH_API_KEY/LANGCHAIN_API_KEY，并清 langsmith.utils 缓存。
# ---------------------------------------------------------------------------
def _sync_langsmith_env() -> None:
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


# 先把 .env 载入 os.environ（供 _sync_langsmith_env 及仍读 os.getenv 的存量代码），
# 再同步 LangSmith；Settings 读取时既看 os.environ 也看 env_file。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BASE_DIR, ".env"))
_sync_langsmith_env()


# ---------------------------------------------------------------------------
# model_catalog 拍平（原样移植 config._load_model_catalog）
# ---------------------------------------------------------------------------
def _load_model_catalog() -> dict[str, str]:
    """读取 data/model_catalog.json 的 active profile，扁平化为 LLM 配置字典。

    空字符串表示「未配置，使用 .env 兜底」。返回字段见 Settings._apply_catalog。
    """
    import json as _json

    catalog_path = os.path.join(BASE_DIR, "data", "model_catalog.json")
    try:
        with open(catalog_path, encoding="utf-8") as f:
            data = _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return {}

    active_id = data.get("active_profile", "default")
    profile = next(
        (p for p in data.get("profiles", []) if p.get("id") == active_id),
        None,
    )
    if not profile:
        return {}

    models = profile.get("models", {})
    fallback = profile.get("fallback", {})
    emb = models.get("embedding", {})

    return {
        "binding": profile.get("binding", ""),
        "api_key": profile.get("api_key", ""),
        "base_url": profile.get("base_url", ""),
        "api_version": profile.get("api_version", ""),
        "text_model": models.get("text", {}).get("model", ""),
        "fast_model": models.get("fast", {}).get("model", ""),
        "vision_model": models.get("vision", {}).get("model", ""),
        "embedding_model": emb.get("model", ""),
        "embedding_api_key": emb.get("api_key", ""),
        "embedding_base_url": emb.get("base_url", ""),
        "fallback_api_key": fallback.get("api_key", ""),
        "fallback_base_url": fallback.get("base_url", ""),
        "fallback_model": fallback.get("model", ""),
    }


class Settings(BaseSettings):
    """全项目配置单例。字段分组按原 config.py 的注释段落排列。"""

    model_config = SettingsConfigDict(
        env_file=(os.path.join(BASE_DIR, ".env"),),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 日志 / 环境 ────────────────────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"
    llm_timeout_sec: int = 120
    faq_cache_threshold: int = 3

    # ── RAG backend 选择（平台相关 + legacy）──────────────────────────────
    rag_backend: str = ""
    agentic_rag_backend: str = ""
    agentic_kb_tool: str = ""  # legacy alias，仅 validator 读

    # ── LLM / 多供应商（catalog 优先 → env 兜底，见 _apply_catalog）────────
    llm_binding: str = "dashscope"
    dashscope_api_key: StrippedSecret = SecretStr("")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_version: str = ""
    text_model: str = "qwen-plus"
    fast_model: str = "qwen-turbo"
    vision_model: str = "qwen-vl-plus"
    embedding_model: str = "text-embedding-v3"
    embedding_api_key: StrippedSecret = SecretStr("")  # 空 → 回退 dashscope key
    embedding_base_url: str = ""  # 空 → 回退 dashscope base_url
    fallback_api_key: StrippedSecret = SecretStr("")
    fallback_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    fallback_model: str = "qwen-plus"

    # ── LightRAG 分角色 LLM（1.5.4+）────────────────────────────────────────
    extract_model: str = "qwen-turbo"  # EXTRACT 角色：实体提取，用低成本模型
    keyword_model: str = "qwen-turbo"  # KEYWORD 角色：关键词提取，用低成本模型

    # ── LLM 可靠性 / 生成参数（收编原硬编码：llm.py RetryConfig、loop.py 字面量）──
    llm_retry_max: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 30.0
    llm_retry_exponential_base: float = 2.0
    llm_circuit_failure_threshold: int = 5
    llm_circuit_success_threshold: int = 2
    llm_circuit_open_timeout: float = 30.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192
    agent_max_iterations: int = 10
    agent_token_chunk_size: int = 8

    # ── 用户级 provider 主加密密钥（Fernet；prod 必填，见 _check_prod）─────
    provider_encryption_key: SecretStr = SecretStr("")

    # ── RAG tuning ─────────────────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 4
    ingest_chunk_size: int = 900
    ingest_chunk_over_lap: int = 60

    # ── Paths ──────────────────────────────────────────────────────────────
    upload_dir: str = ""
    knowledge_dir: str = ""
    vectorstore_dir: str = ""
    db_path: str = ""
    question_log_dir: str = ""
    llama_index_kb_root: str = ""
    lightrag_workdir: str = ""
    kb_store_dir: str = ""
    tutorbot_workspace_dir: str = ""

    # ── PostgreSQL + Redis ─────────────────────────────────────────────────
    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/course_agent"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    db_pool_size: int = 10
    db_max_overflow: int = 15

    # ── Security ───────────────────────────────────────────────────────────
    jwt_secret: SecretStr = SecretStr("dev-secret-change-in-production")
    jwt_expire_hours: int = 72
    allowed_origins: OriginsList = ["*"]
    admin_username: str = "admin"

    # ── Upload limits ──────────────────────────────────────────────────────
    max_upload_mb: int = 10
    max_kb_upload_mb: int = 50

    # ── LightRAG ───────────────────────────────────────────────────────────
    lightrag_enabled: TruthyBool = False
    lightrag_query_mode: str = "mix"
    lightrag_top_k: int = 20
    lightrag_timeout_sec: int = 120
    lightrag_embedding_dim: int = 1024
    lightrag_auto_index_ttl_sec: int = 120
    lightrag_stream_context_limit: int = 4
    lightrag_stream_context_max_chars: int = 800
    lightrag_agentic_rag_max_chars: int = 10000
    lightrag_enable_rerank: TruthyBool = False
    lightrag_save_ingest_chunks: TruthyBool = True
    lightrag_ingest_chunks_subdir: str = "ingest_chunks"
    lightrag_ingest_chunks_snapshot: TruthyBool = False
    lightrag_ingest_batch_size: int = 16
    lightrag_max_async: int = 8
    lightrag_lru_capacity: int = 10

    # ── LlamaParse / LlamaIndex ────────────────────────────────────────────
    llama_cloud_api_key: SecretStr = SecretStr("")  # 兼容 LLAMAPARSE_API_KEY 别名
    llamaparse_api_key: SecretStr = SecretStr("")  # 仅 validator 读，作 fallback
    question_use_llamaindex: TruthyBool = True

    # ── Question coordinator ───────────────────────────────────────────────
    question_tool_web_search: TruthyBool = True
    question_tool_rag: TruthyBool = True
    question_tool_code_execution: TruthyBool = True

    # ── LangSmith / observability ──────────────────────────────────────────
    langsmith_tracing: TruthyBool = False
    langsmith_api_key: SecretStr = SecretStr("")
    langsmith_project: str = ""

    # ── TutorBot 社交平台集成 ──────────────────────────────────────────────
    tutorbot_enabled: TruthyBool = False
    tutorbot_heartbeat_enabled: TruthyBool = True
    tutorbot_heartbeat_interval_sec: int = 1800

    # ── QQ Bot ─────────────────────────────────────────────────────────────
    qq_bot_enabled: TruthyBool = False
    qq_app_id: SecretStr = SecretStr("")
    qq_secret: SecretStr = SecretStr("")
    qq_allow_from: str = "*"

    # ── Feishu Bot ─────────────────────────────────────────────────────────
    feishu_bot_enabled: TruthyBool = False
    feishu_app_id: SecretStr = SecretStr("")
    feishu_app_secret: SecretStr = SecretStr("")
    feishu_encrypt_key: SecretStr = SecretStr("")
    feishu_verification_token: SecretStr = SecretStr("")
    feishu_allow_from: str = "*"

    # ── Search（web search 服务）────────────────────────────────────────────
    search_enabled: TruthyBool = True
    search_provider: str = "duckduckgo"
    search_api_key: SecretStr = SecretStr("")
    search_base_url: str = ""
    search_max_results: int = 5
    search_proxy: str = ""
    search_config_path: str = ""  # data/search_config.json 路径

    # ── MCP（Model Context Protocol）──────────────────────────────────────────
    mcp_config_path: str = ""  # data/mcp.json 路径
    mcp_sessions_dir: str = ""  # MCP sessions 存储目录

    # ── Image ingestion（LLamaIndex 图片提取）────────────────────────────────
    image_ingest_min_px: int = 80
    image_ingest_min_area: int = 15000
    image_ingest_max_per_file: int = 120
    image_ingest_semaphore: int = 5
    image_ingest_wmf_min_blob: int = 2000

    # ── Embedding bridge───────────────────────────────────────────────────────
    embedding_dim: int = 1024
    embedding_batch_size: int = 16

    # ── LightRAG SAFE_* 参数（运行时限制，防止过载）─────────────────────────
    lightrag_safe_top_k: int = 10
    lightrag_chunk_top_k: int = 8
    lightrag_max_total_tokens: int = 22000
    lightrag_max_entity_tokens: int = 4000
    lightrag_max_relation_tokens: int = 4000
    lightrag_max_history_messages: int = 8
    lightrag_max_history_chars: int = 8000
    lightrag_llm_system_max_chars: int = 24000

    # ── Output cards（技能输出卡片存储）──────────────────────────────────────
    output_cards_path: str = ""  # 默认在 validator 中设置

    # ── Testing flag───────────────────────────────────────────────────────────
    testing: TruthyBool = False

    # ── Mem0 时间衰减评分 ──────────────────────────────────────────────────────
    mem0_time_decay_enabled: TruthyBool = False  # 开关（默认关闭=原版行为）
    mem0_time_decay_lambda: float = 0.005  # 衰减系数（半衰期约 139 天）
    mem0_conflict_detect_enabled: TruthyBool = False  # 矛盾检测开关
    mem0_conflict_similarity_threshold: float = 0.85  # 文本相似度阈值
    mem0_conflict_min_days_gap: int = 7  # 最小时间差（天）
    mem0_add_skip_patterns: str = "好的,谢谢,嗯,ok,明白了,收到,好,知道了,继续"  # 跳过 add 的模式（逗号分隔）
    mem0_add_min_length: int = 10  # 用户消息最小长度

    # ── Mem0 批量刷新 ──────────────────────────────────────────────────────────
    mem0_flush_max_turns: int = 3  # 累积 N 轮后 flush
    mem0_flush_idle_timeout: float = 120.0  # 静默 T 秒后 flush
    mem0_flush_scan_interval: int = 30  # ARQ cron 扫描间隔（秒）

    # ── Session L2 摘要 ─────────────────────────────────────────────────────────
    summary_window_size: int = 10  # L1 窗口大小（轮）
    summary_buffer_size: int = 2  # 超出窗口多少条才触发压缩
    summary_compress_interval: int = 5  # 每隔 N 轮才重新压缩

    # ------------------------------------------------------------------
    # validators
    # ------------------------------------------------------------------
    @field_validator("log_level", mode="after")
    @classmethod
    def _upper_log(cls, v: str) -> str:
        return v.upper()

    @field_validator("environment", mode="after")
    @classmethod
    def _lower_env(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("rag_backend", mode="after")
    @classmethod
    def _resolve_rag_backend(cls, v: str) -> str:
        raw = (v or "").strip().lower()
        if raw in ("chroma", "fs"):
            return raw
        return "fs" if sys.platform == "win32" else "chroma"

    @model_validator(mode="after")
    def _apply_legacy_and_fallbacks(self) -> "Settings":
        # AGENTIC_RAG_BACKEND legacy alias（AGENTIC_KB_TOOL=llamaindex_rag → llamaindex）
        arg = (self.agentic_rag_backend or "").strip().lower()
        legacy = (self.agentic_kb_tool or "").strip().lower()
        if arg in ("lightrag", "llamaindex"):
            self.agentic_rag_backend = arg
        elif legacy == "llamaindex_rag":
            self.agentic_rag_backend = "llamaindex"
        else:
            self.agentic_rag_backend = "lightrag"

        # LLAMA_CLOUD_API_KEY fallback to LLAMAPARSE_API_KEY
        if not self.llama_cloud_api_key.get_secret_value() and self.llamaparse_api_key.get_secret_value():
            self.llama_cloud_api_key = self.llamaparse_api_key

        # EMBEDDING_* 回退主 LLM 凭证
        if not self.embedding_api_key.get_secret_value():
            self.embedding_api_key = self.dashscope_api_key
        if not self.embedding_base_url:
            self.embedding_base_url = self.dashscope_base_url

        # 路径默认值（依赖 BASE_DIR，在此组装）
        self.upload_dir = self.upload_dir or os.path.join(BASE_DIR, "uploads")
        self.knowledge_dir = self.knowledge_dir or os.path.join(BASE_DIR, "knowledge")
        self.vectorstore_dir = self.vectorstore_dir or os.path.join(BASE_DIR, "vectorstore")
        self.db_path = self.db_path or os.path.join(BASE_DIR, "data", "sessions.db")
        self.question_log_dir = self.question_log_dir or os.path.join(BASE_DIR, "logs", "question")
        self.llama_index_kb_root = self.llama_index_kb_root or os.path.join(BASE_DIR, "data", "knowledge_bases")
        self.lightrag_workdir = self.lightrag_workdir or os.path.join(BASE_DIR, "lightrag_store")
        self.kb_store_dir = self.kb_store_dir or os.path.join(BASE_DIR, "kb_store")
        self.tutorbot_workspace_dir = self.tutorbot_workspace_dir or os.path.join(BASE_DIR, "data", "tutorbot")
        self.search_config_path = self.search_config_path or os.path.join(BASE_DIR, "data", "search_config.json")
        self.mcp_config_path = self.mcp_config_path or os.path.join(BASE_DIR, "data", "mcp.json")
        self.mcp_sessions_dir = self.mcp_sessions_dir or os.path.join(BASE_DIR, "data", "sessions")
        self.output_cards_path = self.output_cards_path or os.path.join(BASE_DIR, "data", "output_cards.json")
        return self

    @model_validator(mode="after")
    def _apply_catalog(self) -> "Settings":
        """catalog 优先 → 已解析的 env 默认值兜底（行为等同旧 config.py）。"""
        cat = _load_model_catalog()

        def or_env(key: str, current: Any) -> Any:
            val = cat.get(key, "")
            return val if val not in ("", None) else current

        self.llm_binding = or_env("binding", self.llm_binding) or "dashscope"
        if cat.get("api_key"):
            self.dashscope_api_key = SecretStr(str(cat["api_key"]).strip())
        self.dashscope_base_url = or_env("base_url", self.dashscope_base_url)
        self.llm_api_version = or_env("api_version", self.llm_api_version)
        self.text_model = or_env("text_model", self.text_model)
        self.fast_model = or_env("fast_model", self.fast_model)
        self.vision_model = or_env("vision_model", self.vision_model)
        self.embedding_model = or_env("embedding_model", self.embedding_model)
        if cat.get("embedding_api_key"):
            self.embedding_api_key = SecretStr(str(cat["embedding_api_key"]).strip())
        if cat.get("embedding_base_url"):
            self.embedding_base_url = str(cat["embedding_base_url"])
        if cat.get("fallback_api_key"):
            self.fallback_api_key = SecretStr(str(cat["fallback_api_key"]).strip())
        self.fallback_base_url = or_env("fallback_base_url", self.fallback_base_url)
        self.fallback_model = or_env("fallback_model", self.fallback_model)

        # ── 告警：catalog 有真实 key 但 .env 无对应值 ───────────────────────────
        # 提示用户迁移到 .env（catalog 将在 .gitignore 中）
        env_dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if cat.get("api_key") and not env_dashscope_key:
            warnings.warn(
                "model_catalog.json contains API key but DASHSCOPE_API_KEY is not set in .env. "
                "Consider migrating the key to .env for security (catalog is gitignored).",
                stacklevel=1,
            )

        return self

    @model_validator(mode="after")
    def _check_prod(self) -> "Settings":
        """prod 安全门 + dev 告警（原样移植旧 config.py 的 JWT/CORS 校验）。"""
        is_prod = self.environment == "production"

        if is_prod and self.jwt_secret.get_secret_value() == "dev-secret-change-in-production":
            raise RuntimeError(
                "FATAL: JWT_SECRET must be set to a strong value in production. "
                "Set the JWT_SECRET environment variable."
            )
        elif self.jwt_secret.get_secret_value() == "dev-secret-change-in-production":
            warnings.warn(
                "JWT_SECRET is using the insecure default value! "
                "Set a strong JWT_SECRET environment variable before deploying to production.",
                stacklevel=1,
            )

        if is_prod and (not self.allowed_origins or self.allowed_origins == ["*"]):
            raise RuntimeError(
                "FATAL: ALLOWED_ORIGINS must be set to specific origins in production. "
                "Example: ALLOWED_ORIGINS=https://yourdomain.com"
            )

        # 用户级 provider 主密钥：prod 必填（否则用户 key 无法安全加密）
        if is_prod and not self.provider_encryption_key.get_secret_value():
            raise RuntimeError(
                "FATAL: PROVIDER_ENCRYPTION_KEY must be set in production to encrypt "
                "user-level LLM API keys. Generate one: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )

        return self

    # ------------------------------------------------------------------
    # 便捷分组视图（供新代码 settings.llm.* 之类访问；shim 走扁平字段）
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的全局 Settings 单例（首次实例化即 fail-fast 校验）。"""
    return Settings()


__all__ = ["Settings", "get_settings", "BASE_DIR"]
