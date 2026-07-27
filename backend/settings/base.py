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
    # 摄入切块大小（摄入流水线真正生效的切块参数，.env: CHUNKING__INGEST_SIZE/OVERLAP）。
    # 默认 1200/120 与原硬编码 LLAMA_INDEX_CHUNK_SIZE 对齐（行为零变化），可调小做评测。
    ingest_size: int = 1200
    ingest_overlap: int = 120
    # 切块策略（可插拔扩展点）。默认 sentence_splitter = 现有 LlamaIndex SentenceSplitter。
    # ragflow_manual_docx = DOCX 走移植自 RAGFlow Manual 的结构化切块（标题层级栈 +
    #   表格原子化），非 DOCX 仍走 SentenceSplitter。.env: CHUNKING__STRATEGY=ragflow_manual_docx
    strategy: str = "sentence_splitter"
    # Phase 2: Contextual Retrieval（Anthropic）。默认关，需 .env 显式开启：
    #   CHUNKING__CONTEXTUAL_ENRICHMENT=true。开启后摄入时给每个 chunk 注入文档
    #   背景前缀（fast_model 生成），提升检索命中率，代价是每 chunk 一次 LLM 调用。
    contextual_enrichment: TruthyBool = False
    contextual_model: str = ""  # 空=用 llm.fast_model
    # Phase 4: 图片描述回填。开启时把 VLM 图片描述作为独立文本 chunk 追加进索引，
    # 让纯向量检索(fact mode)也能召回图片内容。复用 image_extractor 的 desc_cache，
    # 不重花 VLM。默认关，行为零变化。.env: CHUNKING__INLINE_IMAGE_DESCRIPTIONS=true
    inline_image_descriptions: TruthyBool = False


class ContextPolicyConfig(BaseModel):
    """轮内上下文预算策略（第二批）。默认全部关闭=行为与旧 _snip_tool_results 等价，便于消融对照。

    调研依据 arXiv:2508.21433（The Complexity Trap, JetBrains）：Observation Masking 相对
    Raw 成本减半、解题率持平；纯摘要引发 trajectory elongation。最优是 hybrid（掩码为主、
    摘要为最后手段）。M 取小值（max_iterations=10，远小于 SWE-agent 的 250 轮）。
    .env 前缀 CONTEXT_POLICY__*。
    """

    # 总开关。关=loop 仍走旧 _snip_tool_results（按全局字符>80000 从最早 tool 替换，行为零变化）；
    # 开=走 context_policy.apply 三段式（cap 单条 + 掩码窗口化 + 可选 hybrid 摘要）。
    enabled: TruthyBool = False
    keep_recent_turns: int = 3      # 保留最近 M 轮 role=tool 原文，更早掩码（mask_old_observations）
    budget_chars: int = 80_000      # 总字符触发点，与旧 _snip_tool_results 一致
    tool_result_max_chars: int = 6000  # 单条 tool 结果头尾保留上限（cap_tool_result）；0=不限
    summary_enabled: TruthyBool = False  # hybrid 兜底摘要（调 fast LLM，默认关）
    summary_threshold: int = 4      # 被掩码轮数 >= 此值才触发摘要


class CostQuotaConfig(BaseModel):
    """每用户/课程/自然日 的 LLM 成本配额（第四批）。默认全关=行为零变化，便于灰度。

    设计要点：
    - **只降级不拒绝**：超预算时把本轮对话模型从 text_model 降到便宜档 fast_model
      （profile 配置），而非返回 4xx 阻断用户——避免「上一轮刚好超预算，下一轮被锁死」。
    - **软预算**：成本在 loop 结束后按 estimate_cost 累加（loop.py），故「推过预算的那一轮」
      本身不降级，下一轮才降级——这是 quota 的固有滞后，符合业界软限流惯例。
    - **best-effort**：Redis 不可用时 check_quota 直接放行、accrue_cost 静默跳过，绝不阻塞业务。
    - 计数键 ca:costquota:{user_id}:{course_id}:{YYYYMMDD}，按日滚动（TTL 2 天自清理）。
    .env 前缀 COST_QUOTA__*。
    """

    enabled: TruthyBool = False        # 总开关。关=check/accrue 均短路，零行为变化
    daily_budget_usd: float = 1.0      # 每用户/课程/自然日 的 USD 预算上限
    degrade_model: TruthyBool = True   # 超预算→降级到 fast_model（False=仅记录不降级）


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
    # 分角色模型（LightRAG 1.5.4+ role_llm_configs）：空=回退 llm_model（行为不变）；
    # 非空=该角色索引时改用此模型，凭证复用上面的 api_key/base_url（须同 provider 可见）。
    extract_model: str = ""   # EXTRACT 角色：实体/关系提取（索引 token 大头，建议便宜档）
    keyword_model: str = ""   # KEYWORD 角色：关键词提取
    enabled: TruthyBool = False
    query_mode: str = "mix"
    top_k: int = 6
    timeout_sec: int = 120
    embedding_dim: int = 1024
    auto_index_ttl_sec: int = 120
    stream_context_limit: int = 4
    stream_context_max_chars: int = 800
    agentic_rag_max_chars: int = 10000
    # relationship 路每块上下文字符上限（graph_augmented_retrieve 用，max_chars=None 时读它）。
    # 默认 4000=现状（行为不变）；改 .env LIGHTRAG__GRAPH_AUGMENTED_MAX_CHARS 放宽，
    # 减少 LightRAG local 实体图被硬切。fact 路不受影响（retrieve_context 仍 4000）。
    graph_augmented_max_chars: int = 4000
    enable_rerank: TruthyBool = True
    # 相关性阈值过滤（LightRAG 1.5.4 原生 min_rerank_score，见 lightrag/utils.py 过滤逻辑：
    # 低于阈值的 chunk 在 rerank 后被丢弃，全滤掉则返回 []→触发无命中路径）。
    # 默认 0.0 = 不过滤 = 行为完全不变；阈值口径是 DashScope gte-rerank-v2 的 relevance_score
    #（已接 rerank_adapter.py），与裸余弦相似度分布不同，不能照搬 0.5。生效前置条件：
    # enable_rerank=True（默认）且 rerank_model_func 已挂载（需 EMBEDDING__API_KEY），
    # 缺 key 时阈值静默失效，但无命中哨兵修复不受影响。
    min_rerank_score: float = 0.0
    # ingestion
    save_ingest_chunks: TruthyBool = True
    ingest_chunks_subdir: str = "ingest_chunks"
    ingest_chunks_snapshot: TruthyBool = False
    ingest_batch_size: int = 16
    max_async: int = 8
    lru_capacity: int = 10  # 同时驻留内存的最大课程数（按 worker 缩放，见 Settings.lightrag_lru_capacity_scaled）
    chunk_top_k: int = 5
    max_total_tokens: int = 14000
    max_entity_tokens: int = 3000
    max_relation_tokens: int = 3000
    max_history_messages: int = 8
    max_history_chars: int = 8000
    llm_system_max_chars: int = 24000

    # ── 计算方法 ──
    def chunk_top_k_value(self) -> int:
        """min(chunk_top_k, top_k) —— 最终塞进 prompt 的 chunk 数不超过种子实体数。"""
        return min(self.chunk_top_k, self.top_k)

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


class PdfConfig(BaseModel):
    """PDF 摄入解析配置（backend 二选一，无降级）。

    PDF 解析与 DOCX 同构（extract_pdf_sections → section 列表 → sentence_splitter），
    backend 决定走哪个解析器；切块统一 sentence_splitter，不再有 pdf_structured 策略。
    .env: PDF__BACKEND / PDF__DO_OCR / PDF__OCR_PROVIDER。

    - docling（默认）：Docling 单引擎全包——版面/表格/标题/页码 + OCR（RapidOCR 后端）。
      需 pip install docling rapidocr-onnxruntime（首次运行模型下到 ~/.cache/docling）。
    - mupdf：PyMuPDF 轻量纯文本 + 章节(get_toc) + 页码，不装 torch；无 OCR/表格原子化。
    """

    backend: str = "docling"  # docling(默认) | mupdf；二选一，选定失败则跳过该文件，不降级
    do_ocr: TruthyBool = True  # 仅 docling：扫描件 OCR
    ocr_provider: str = "rapid"  # 仅 docling：rapid | easyocr


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
    # 检索相关性过滤阈值（P0-C）：mem0 V3 search 原生支持 threshold，>0 时过滤低相关噪声。
    # 默认 0.0=不过滤=行为完全不变。当前部署 mem0 版本对 threshold 关键字的兼容性未实测
    # （Docker 起不来），build_memory_context 已加 TypeError 自适应降级兜底。
    # recency_decay_lambda 是否真实生效同理待 Docker 验证。
    search_threshold: float = 0.0
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


class ElasticsearchConfig(BaseModel):
    """Elasticsearch BM25（Phase 3 hybrid search）。

    enabled=False（默认）时 get_es_store() 返回 None，检索降级为纯 dense（LightRAG
    naive），ingestion 也跳过 ES 双写。需先 ``pip install elasticsearch[async]`` 并
    启动 ES（见 docker-compose.yml 的 elasticsearch 服务 + ik 分词插件）。
    """

    enabled: TruthyBool = False
    url: str = "http://localhost:9200"
    index_name: str = "rag_chunks"


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
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    tutorbot: TutorBotConfig = Field(default_factory=TutorBotConfig)
    qq_bot: QQBotConfig = Field(default_factory=QQBotConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    image_ingest: ImageIngestConfig = Field(default_factory=ImageIngestConfig)
    mem0: Mem0Config = Field(default_factory=Mem0Config)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    question: QuestionConfig = Field(default_factory=QuestionConfig)
    elasticsearch: ElasticsearchConfig = Field(default_factory=ElasticsearchConfig)
    context_policy: ContextPolicyConfig = Field(default_factory=ContextPolicyConfig)
    cost_quota: CostQuotaConfig = Field(default_factory=CostQuotaConfig)

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
        """LRU 容量按 worker 数缩放（多 worker 下避免超额驻留）。

        M-23 说明：本公式针对 **web worker 进程**（处理并发检索，需多实例驻留）。
        ARQ worker（python -m arq，跑后台索引）是独立进程，拥有各自的模块级 _instances
        池，与本进程隔离；ARQ 一次只跑一个 indexing job，容量需求实际为 1，但本公式
        同样会给它分一份（轻微多驻留 1 个实例，属可接受的内存开销，不造成数据问题）。
        若未来 ARQ 并发多任务，需单独引入 arq 容量配置项。
        """
        return max(2, self.lightrag.lru_capacity // max(1, self.backend_workers))

    def validate_runtime_workers(
        self, known_workers: int | None = None
    ) -> bool:
        """运行时校验 ``backend_workers`` 与真实 worker 进程数是否一致（M-24）。

        ``backend_workers`` 驱动 DB 连接池缩放（database.py）、LLM 熔断阈值缩放
        （reliability.py）、LightRAG LRU 容量缩放（本类 property）——三者都假设
        它等于 gunicorn/uvicorn 实际拉起的 ``-w`` 进程数。但 ``-w`` 数写在部署
        脚本里（Dockerfile/compose），与 ``.env`` 的 ``BACKEND_WORKERS`` 是两套
        手动维护的值，一旦不同步，缩放公式就会算偏（例如真 8 worker 但 env 写 4，
        实际连接数会翻倍打爆 Postgres）。本方法在进程启动期（lifespan）跑一次，
        把不一致暴露为显式告警。

        真实 worker 数来源优先级：
        1. 调用方显式传入的 ``known_workers``（测试 / 自定义编排注入，最准）；
        2. 环境变量 ``WEB_CONCURRENCY``（gunicorn/uvicorn/容器平台常用约定）；
        3. 其余环境（无 ``-w`` 注入线索，如 pytest、dev 单进程 uvicorn）→ 跳过校验，
           返回 ``True``（无法判定时不制造噪音）。

        不抛异常（缩放偏只是性能/容量问题，不致数据错误），仅 ``warnings.warn``。
        返回 ``True`` 表示一致或无法判定，``False`` 表示检测到不一致。
        """
        if known_workers is None:
            raw = os.getenv("WEB_CONCURRENCY", "").strip()
            if not raw:
                return True  # 无注入线索（如测试/dev），不校验
            try:
                known_workers = int(raw)
            except ValueError:
                return True  # 非法值，交由调用方日志处理，不阻断启动
        if known_workers <= 0:
            return True
        if known_workers != self.backend_workers:
            warnings.warn(
                f"BACKEND_WORKERS ({self.backend_workers}) does not match actual worker "
                f"count ({known_workers}). DB pool / circuit-breaker / LRU scaling "
                f"formulas assume these are equal — update .env BACKEND_WORKERS or the "
                f"gunicorn/uvicorn -w flag to match.",
                stacklevel=2,
            )
            return False
        return True



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的全局 Settings 单例（首次实例化即 fail-fast 校验）。"""
    return Settings()


__all__ = ["Settings", "get_settings", "BASE_DIR"]
