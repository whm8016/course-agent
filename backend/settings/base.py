"""集中式配置（pydantic-settings 组合模式）—— 全项目单一事实源。

架构（业界标准组合式）：
- 顶层 ``Settings(BaseSettings)`` + ``env_nested_delimiter="__"``。
- 各职责拆为嵌套 ``BaseModel`` 子组（LLM/Vision/Embedding/Fallback/DB/
  Security/Paths/Chunking/LightRAG/Rag/LlamaParse/TutorBot/QQBot/Feishu/
  Search/ImageIngest/Mem0/Summary/Question）。
- ``.env`` 用分隔式注入名：``LLM__API_KEY`` → ``settings.llm.api_key``、
  ``DB__URL`` → ``settings.db.url``、``SECURITY__JWT_SECRET`` →
  ``settings.security.jwt_secret`` …（严格新名，不兼容旧扁平名）。

例外：``langsmith_*`` 与 ``environment``/``log_level``/``testing``/
``backend_workers``/``max_upload_mb``/``max_kb_upload_mb`` 保留顶层扁平字段。
其中 langsmith 因 langchain SDK 硬性读 ``os.environ["LANGSMITH_API_KEY"]``
扁平名（时序难控），保持扁平 + 提供 ``settings.langsmith`` property 视图。

设计要点（继承自旧扁平 Settings）：
- secrets 用 SecretStr，print 不泄密；shim/消费方 ``.get_secret_value()`` 还原。
- bool 统一经 ``_truthy``；origins 逗号分隔。
- model_catalog 拍平（catalog 优先 → env 兜底）、LangSmith env 同步、
  JWT/CORS prod 校验、RAG backend 平台判断、跨组 fallback
  （embedding←llm、vision←embedding）—— 全部保留。
- 实例化即校验 → fail-fast。

``get_settings()`` 为 lru_cache 单例；``config.py`` shim 读取本模块转发。
"""
from __future__ import annotations

import os
import sys
import warnings
from functools import lru_cache
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 公共类型：bool 复刻旧 truthy 约定；str secret 去首尾空白；origins 逗号分隔
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
    """逗号分隔 → list（空/* → ["*"]）。NoDecode 防止 JSON 解析。"""
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
# LangSmith env 同步（原样保留：模块级，读/写扁平 os.environ 供 langchain SDK）
#   归一化 truthy、互填 *_TRACING_V2、互填 LANGSMITH_API_KEY/LANGCHAIN_API_KEY，
#   并清 langsmith.utils 缓存。langsmith_* 仍是顶层扁平字段（见模块 docstring 例外）。
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
# model_catalog 拍平（原样保留：读 active profile 扁平化为 LLM 配置字典）
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


# ===========================================================================
# 嵌套子组 BaseModel
# ===========================================================================


class LLMConfig(BaseModel):
    """主 LLM 供应商 + 生成参数 + 可靠性（retry / circuit breaker）。"""

    binding: str = "dashscope"
    api_key: StrippedSecret = SecretStr("")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_version: str = ""
    text_model: str = "qwen-plus"
    fast_model: str = "qwen-turbo"
    # LightRAG 分角色 LLM（1.5.4+）：低成本模型
    extract_model: str = "qwen-turbo"  # EXTRACT 角色：实体提取
    keyword_model: str = "qwen-turbo"  # KEYWORD 角色：关键词提取
    # 生成参数 / 超时（收编原硬编码）
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout_sec: int = 120
    # retry
    retry_max: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_exponential_base: float = 2.0
    # circuit breaker
    circuit_failure_threshold: int = 5
    circuit_success_threshold: int = 2
    circuit_open_timeout: float = 30.0
    # agent
    agent_max_iterations: int = 10
    agent_token_chunk_size: int = 8


class VisionConfig(BaseModel):
    """两阶段视觉模型（qwen-vl 系列）。凭证独立于 llm，空 → 回退 embedding。"""

    model: str = "qwen-vl-plus"  # 对话图像查询默认
    index_model: str = "qwen-vl-plus"  # 索引知识库提图（凭证与对话共用）
    api_key: StrippedSecret = SecretStr("")
    base_url: str = ""


class EmbeddingConfig(BaseModel):
    """Embedding（单一默认，不分裂）。"""

    model: str = "text-embedding-v3"
    api_key: StrippedSecret = SecretStr("")  # 空 → 回退 llm.api_key
    base_url: str = ""  # 空 → 回退 llm.base_url
    dim: int = 1024
    batch_size: int = 16


class FallbackConfig(BaseModel):
    """fallback provider（主 provider 失败时切换）。"""

    api_key: StrippedSecret = SecretStr("")
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"


class DatabaseConfig(BaseModel):
    """PostgreSQL + Redis。"""

    url: SecretStr = SecretStr(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/course_agent"
    )
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    pool_size: int = 10
    max_overflow: int = 15


class SecurityConfig(BaseModel):
    """安全：JWT / CORS / 管理员 / 用户 provider 加密密钥。"""

    jwt_secret: SecretStr = SecretStr("dev-secret-change-in-production")
    jwt_expire_hours: int = 72
    allowed_origins: OriginsList = ["*"]
    admin_username: str = "admin"
    provider_encryption_key: SecretStr = SecretStr("")


class PathsConfig(BaseModel):
    """运行期路径（默认空，Settings validator 用 BASE_DIR 组装）。"""

    upload_dir: str = ""
    knowledge_dir: str = ""
    vectorstore_dir: str = ""
    db_path: str = ""
    question_log_dir: str = ""
    llama_index_kb_root: str = ""
    lightrag_workdir: str = ""
    kb_store_dir: str = ""
    tutorbot_workspace_dir: str = ""
    search_config_path: str = ""
    mcp_config_path: str = ""
    mcp_sessions_dir: str = ""
    output_cards_path: str = ""


class ChunkingConfig(BaseModel):
    """RAG 切块调参。"""

    size: int = 500
    overlap: int = 80
    top_k: int = 4
    ingest_size: int = 900
    ingest_overlap: int = 60


class LightRAGConfig(BaseModel):
    """LightRAG 后端配置 + 安全阈值 + 计算方法。

    llm_model/api_key/base_url 是 LightRAG 索引 LLM 的专属 provider，完全独立于
    llm.*——_apply_catalog() 用 model_catalog.json 覆盖 llm.*/embedding.*/fallback.*
    时不会 touch self.lightrag.*，故这三者永远只来自 env，不受前端切 provider 影响。
    也没有任何回退：未配置即未配置，is_lightrag_available() 直接报错拒绝启用，
    不会静默套用其它 provider 凭证（否则管理员切 catalog 会把 LightRAG 索引打挂）。
    """

    # 索引 LLM 专属 provider（与对话主 LLM 解耦，零回退；未配则 is_lightrag_available 报错）
    llm_model: str = ""
    api_key: StrippedSecret = SecretStr("")
    base_url: str = ""
    enabled: TruthyBool = False
    query_mode: str = "mix"
    top_k: int = 20
    timeout_sec: int = 120
    embedding_dim: int = 1024
    auto_index_ttl_sec: int = 120
    stream_context_limit: int = 4
    stream_context_max_chars: int = 800
    agentic_rag_max_chars: int = 10000
    enable_rerank: TruthyBool = True
    # ingestion
    save_ingest_chunks: TruthyBool = True
    ingest_chunks_subdir: str = "ingest_chunks"
    ingest_chunks_snapshot: TruthyBool = False
    ingest_batch_size: int = 16
    max_async: int = 8
    lru_capacity: int = 10  # 同时驻留内存的最大课程数（按 worker 缩放，见 Settings.lightrag_lru_capacity_scaled）
    # 安全阈值（防 API 拒绝）
    safe_top_k: int = 6
    chunk_top_k: int = 5
    max_total_tokens: int = 14000
    max_entity_tokens: int = 3000
    max_relation_tokens: int = 3000
    max_history_messages: int = 8
    max_history_chars: int = 8000
    llm_system_max_chars: int = 24000

    # ── 计算方法（原 core/rag/rag_config.py 的 get_safe_top_k 等，内聚至此）──
    def safe_top_k_value(self) -> int:
        """min(top_k, safe_top_k) —— aquery 的 top_k 安全上限。"""
        return min(self.top_k, self.safe_top_k)

    def chunk_top_k_value(self) -> int:
        """min(chunk_top_k, safe_top_k_value())。"""
        return min(self.chunk_top_k, self.safe_top_k_value())

    def max_tokens_config(self) -> dict[str, int]:
        """对应 LightRAG QueryParam 的 max tokens 字典。"""
        return {
            "total": self.max_total_tokens,
            "entity": self.max_entity_tokens,
            "relation": self.max_relation_tokens,
        }


class RagConfig(BaseModel):
    """RAG backend 选择（平台相关 + legacy alias）。

    backend 的平台解析在 Settings._apply_legacy_and_fallbacks 里做（嵌套
    BaseModel 的 field_validator 对默认值不触发，故用 model_validator）。
    """

    backend: str = ""
    agentic_backend: str = ""
    agentic_kb_tool: str = ""  # legacy alias，仅 Settings validator 读


class LlamaParseConfig(BaseModel):
    """LlamaParse / LlamaIndex（图像 PDF / 扫描件解析）。"""

    cloud_api_key: SecretStr = SecretStr("")  # 兼容 LLAMA_CLOUD_API_KEY 语义
    parse_api_key: SecretStr = SecretStr("")  # 作 cloud_api_key 的 fallback
    question_use_llamaindex: TruthyBool = True


class TutorBotConfig(BaseModel):
    """TutorBot 社交平台集成。"""

    enabled: TruthyBool = False
    heartbeat_enabled: TruthyBool = True
    heartbeat_interval_sec: int = 1800


class QQBotConfig(BaseModel):
    """QQ Bot (botpy SDK)。"""

    enabled: TruthyBool = False
    app_id: SecretStr = SecretStr("")
    secret: SecretStr = SecretStr("")
    allow_from: str = "*"


class FeishuConfig(BaseModel):
    """飞书 Bot (lark-oapi SDK, WebSocket)。"""

    enabled: TruthyBool = False
    app_id: SecretStr = SecretStr("")
    app_secret: SecretStr = SecretStr("")
    encrypt_key: SecretStr = SecretStr("")
    verification_token: SecretStr = SecretStr("")
    allow_from: str = "*"


class SearchConfig(BaseModel):
    """web search 服务。"""

    enabled: TruthyBool = True
    provider: str = "duckduckgo"
    api_key: SecretStr = SecretStr("")
    base_url: str = ""
    max_results: int = 5
    proxy: str = ""


class ImageIngestConfig(BaseModel):
    """LlamaIndex 图片提取参数。"""

    min_px: int = 80
    min_area: int = 15000
    max_per_file: int = 120
    semaphore: int = 5
    wmf_min_blob: int = 2000


class Mem0Config(BaseModel):
    """Mem0 记忆：时间衰减 / 矛盾检测 / 跳过模式 / 批量刷新。"""

    time_decay_enabled: TruthyBool = False
    time_decay_lambda: float = 0.005  # 衰减系数（半衰期约 139 天）
    conflict_detect_enabled: TruthyBool = True
    conflict_similarity_threshold: float = 0.85
    conflict_min_days_gap: int = 7
    add_skip_patterns: str = "好的,谢谢,嗯,ok,明白了,收到,好,知道了,继续"
    add_min_length: int = 10
    flush_max_turns: int = 3
    flush_idle_timeout: float = 120.0
    flush_scan_interval: int = 30


class SummaryConfig(BaseModel):
    """Session L2 摘要。"""

    window_size: int = 5
    buffer_size: int = 2
    compress_interval: int = 3


class QuestionConfig(BaseModel):
    """Question coordinator (AgentCoordinator)。"""

    tool_web_search: TruthyBool = True
    tool_rag: TruthyBool = True
    tool_code_execution: TruthyBool = True
    faq_cache_threshold: int = 3

    @property
    def default_tool_flags(self) -> dict[str, bool]:
        return {
            "web_search": self.tool_web_search,
            "rag": self.tool_rag,
            "code_execution": self.tool_code_execution,
        }


class LangSmithView(BaseModel):
    """langsmith 分组只读视图（字段仍为顶层扁平，见模块 docstring 例外）。"""

    tracing: bool = False
    api_key: SecretStr = SecretStr("")
    project: str = ""


# ===========================================================================
# 顶层 Settings
# ===========================================================================
class Settings(BaseSettings):
    """全项目配置单例（组合式嵌套）。"""

    model_config = SettingsConfigDict(
        env_file=(os.path.join(BASE_DIR, ".env"),),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 顶层扁平字段（env 不嵌套）──────────────────────────────────────────
    log_level: str = "INFO"
    environment: str = "development"
    testing: TruthyBool = False
    backend_workers: int = 4  # 承接原 _os.getenv("BACKEND_WORKERS")，供 LRU 缩放
    max_upload_mb: int = 10
    max_kb_upload_mb: int = 50
    # 对话输入上限：chat（HTTP /api/chat）与 run（WS /run/{cap}）共用 message 长度；
    # history 两入口取值不同（HTTP 短、流式长），故分别配置——勿盲目合并。
    chat_message_max_length: int = 2000
    chat_history_max_length: int = 10
    run_history_max_length: int = 20
    # langsmith 扁平字段（langchain SDK 读 os.environ 扁平名，例外）
    langsmith_tracing: TruthyBool = False
    langsmith_api_key: StrippedSecret = SecretStr("")
    langsmith_project: str = ""

    # ── 嵌套子组 ──────────────────────────────────────────────────────────
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    lightrag: LightRAGConfig = Field(default_factory=LightRAGConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    llamaparse: LlamaParseConfig = Field(default_factory=LlamaParseConfig)
    tutorbot: TutorBotConfig = Field(default_factory=TutorBotConfig)
    qq_bot: QQBotConfig = Field(default_factory=QQBotConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    image_ingest: ImageIngestConfig = Field(default_factory=ImageIngestConfig)
    mem0: Mem0Config = Field(default_factory=Mem0Config)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    question: QuestionConfig = Field(default_factory=QuestionConfig)

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

    @model_validator(mode="after")
    def _apply_legacy_and_fallbacks(self) -> "Settings":
        # RAG backend 平台解析（chroma/fs，否则按 sys.platform）
        raw_be = (self.rag.backend or "").strip().lower()
        if raw_be in ("chroma", "fs"):
            self.rag.backend = raw_be
        else:
            self.rag.backend = "fs" if sys.platform == "win32" else "chroma"

        # AGENTIC_RAG_BACKEND legacy alias（AGENTIC_KB_TOOL=llamaindex_rag → llamaindex）
        arg = (self.rag.agentic_backend or "").strip().lower()
        legacy = (self.rag.agentic_kb_tool or "").strip().lower()
        if arg in ("lightrag", "llamaindex"):
            self.rag.agentic_backend = arg
        elif legacy == "llamaindex_rag":
            self.rag.agentic_backend = "llamaindex"
        else:
            self.rag.agentic_backend = "lightrag"

        # LLAMA_CLOUD_API_KEY fallback to LLAMAPARSE_API_KEY
        if not self.llamaparse.cloud_api_key.get_secret_value() and self.llamaparse.parse_api_key.get_secret_value():
            self.llamaparse.cloud_api_key = self.llamaparse.parse_api_key

        # EMBEDDING_* 回退主 LLM 凭证
        if not self.embedding.api_key.get_secret_value():
            self.embedding.api_key = self.llm.api_key
        if not self.embedding.base_url:
            self.embedding.base_url = self.llm.base_url

        # VISION_* 回退 EMBEDDING_*：qwen-vl 系列与 text-embedding 同属阿里 dashscope，默认
        # 复用 embedding 凭证。注意主 LLM 可能是 deepseek/openai 等不支持 vision 的供应商，
        # 不能作 vision 回退。索引提图 + 问答全局回退共用；可在 .env 配独立覆盖。
        if not self.vision.api_key.get_secret_value():
            self.vision.api_key = self.embedding.api_key
        if not self.vision.base_url:
            self.vision.base_url = self.embedding.base_url

        # 路径默认值（依赖 BASE_DIR，在此组装）
        self.paths.upload_dir = self.paths.upload_dir or os.path.join(BASE_DIR, "uploads")
        self.paths.knowledge_dir = self.paths.knowledge_dir or os.path.join(BASE_DIR, "knowledge")
        self.paths.vectorstore_dir = self.paths.vectorstore_dir or os.path.join(BASE_DIR, "vectorstore")
        self.paths.db_path = self.paths.db_path or os.path.join(BASE_DIR, "data", "sessions.db")
        self.paths.question_log_dir = self.paths.question_log_dir or os.path.join(BASE_DIR, "logs", "question")
        self.paths.llama_index_kb_root = self.paths.llama_index_kb_root or os.path.join(BASE_DIR, "data", "knowledge_bases")
        self.paths.lightrag_workdir = self.paths.lightrag_workdir or os.path.join(BASE_DIR, "lightrag_store")
        self.paths.kb_store_dir = self.paths.kb_store_dir or os.path.join(BASE_DIR, "kb_store")
        self.paths.tutorbot_workspace_dir = self.paths.tutorbot_workspace_dir or os.path.join(BASE_DIR, "data", "tutorbot")
        self.paths.search_config_path = self.paths.search_config_path or os.path.join(BASE_DIR, "data", "search_config.json")
        self.paths.mcp_config_path = self.paths.mcp_config_path or os.path.join(BASE_DIR, "data", "mcp.json")
        self.paths.mcp_sessions_dir = self.paths.mcp_sessions_dir or os.path.join(BASE_DIR, "data", "sessions")
        self.paths.output_cards_path = self.paths.output_cards_path or os.path.join(BASE_DIR, "data", "output_cards.json")
        return self

    @model_validator(mode="after")
    def _apply_catalog(self) -> "Settings":
        """catalog 优先 → 已解析的 env 默认值兜底（行为等同旧 config.py）。"""
        cat = _load_model_catalog()

        def or_env(key: str, current: Any) -> Any:
            val = cat.get(key, "")
            return val if val not in ("", None) else current

        self.llm.binding = or_env("binding", self.llm.binding) or "dashscope"
        if cat.get("api_key"):
            self.llm.api_key = SecretStr(str(cat["api_key"]).strip())
        self.llm.base_url = or_env("base_url", self.llm.base_url)
        self.llm.api_version = or_env("api_version", self.llm.api_version)
        self.llm.text_model = or_env("text_model", self.llm.text_model)
        self.llm.fast_model = or_env("fast_model", self.llm.fast_model)
        # vision 不从 catalog 覆盖：config.VISION_MODEL 是「全局独立视觉描述模型」
        # （走 VISION_API_KEY/BASE_URL，默认回退 EMBEDDING_*；供 ingestion image_extractor +
        # chat 两阶段全局回退共用），与 catalog 各 profile 的 vision 模型语义不同。
        self.embedding.model = or_env("embedding_model", self.embedding.model)
        if cat.get("embedding_api_key"):
            self.embedding.api_key = SecretStr(str(cat["embedding_api_key"]).strip())
        if cat.get("embedding_base_url"):
            self.embedding.base_url = str(cat["embedding_base_url"])
        if cat.get("fallback_api_key"):
            self.fallback.api_key = SecretStr(str(cat["fallback_api_key"]).strip())
        self.fallback.base_url = or_env("fallback_base_url", self.fallback.base_url)
        self.fallback.model = or_env("fallback_model", self.fallback.model)

        # ── 告警：catalog 有真实 key 但 .env 无对应值 ───────────────────────
        env_llm_key = os.getenv("LLM__API_KEY", "").strip()
        if cat.get("api_key") and not env_llm_key:
            warnings.warn(
                "model_catalog.json contains API key but LLM__API_KEY is not set in .env. "
                "Consider migrating the key to .env for security (catalog is gitignored).",
                stacklevel=1,
            )

        return self

    @model_validator(mode="after")
    def _check_prod(self) -> "Settings":
        """prod 安全门 + dev 告警（JWT/CORS/PROVIDER_ENCRYPTION_KEY 校验）。"""
        is_prod = self.environment == "production"

        if is_prod and self.security.jwt_secret.get_secret_value() == "dev-secret-change-in-production":
            raise RuntimeError(
                "FATAL: SECURITY__JWT_SECRET must be set to a strong value in production. "
                "Set the SECURITY__JWT_SECRET environment variable."
            )
        elif self.security.jwt_secret.get_secret_value() == "dev-secret-change-in-production":
            warnings.warn(
                "JWT_SECRET is using the insecure default value! "
                "Set a strong SECURITY__JWT_SECRET environment variable before deploying to production.",
                stacklevel=1,
            )

        if is_prod and (not self.security.allowed_origins or self.security.allowed_origins == ["*"]):
            raise RuntimeError(
                "FATAL: SECURITY__ALLOWED_ORIGINS must be set to specific origins in production. "
                "Example: SECURITY__ALLOWED_ORIGINS=https://yourdomain.com"
            )

        if is_prod and not self.security.provider_encryption_key.get_secret_value():
            raise RuntimeError(
                "FATAL: SECURITY__PROVIDER_ENCRYPTION_KEY must be set in production to encrypt "
                "user-level LLM API keys. Generate one: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )

        return self

    # ------------------------------------------------------------------
    # 便捷分组视图 / 计算属性
    # ------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def langsmith(self) -> LangSmithView:
        """langsmith 分组视图（字段为顶层扁平，SDK 兼容）。"""
        return LangSmithView(
            tracing=self.langsmith_tracing,
            api_key=self.langsmith_api_key,
            project=self.langsmith_project,
        )

    @property
    def lightrag_lru_capacity_scaled(self) -> int:
        """LRU 容量按 worker 数缩放（多 worker 下避免超额驻留）。"""
        return max(2, self.lightrag.lru_capacity // max(1, self.backend_workers))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的全局 Settings 单例（首次实例化即 fail-fast 校验）。"""
    return Settings()


__all__ = ["Settings", "get_settings", "BASE_DIR"]
