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
        # 模型上下文窗口（显式配置，优先于模型名模式与 heuristic 兜底；见 context_window.resolve_effective_window）
        "context_window": profile.get("context_window", ""),
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
    # 模型标称上下文窗口（token）。显式配置优先于模型名模式匹配与 heuristic 兜底（见
    # core.agentic.context_window.resolve_effective_window 三级解析）。None=走三级解析的后两级。
    context_window: int | None = None


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
    db_path: str = ""
    question_log_dir: str = ""
    lightrag_workdir: str = ""
    kb_store_dir: str = ""
    tutorbot_workspace_dir: str = ""
    search_config_path: str = ""
    mcp_config_path: str = ""
    mcp_sessions_dir: str = ""
    output_cards_path: str = ""
    parse_cache_dir: str = ""  # 解析结果内容寻址缓存（空→BASE_DIR/data/parse_cache）
    # llamaindex_pg 摄入切块审计 JSON（独立根，与 parse_cache 同级；空→BASE_DIR/data/ingest_chunks）。
    # 与 lightrag_workdir 分家：lightrag_store 内有永不删的 graphml，pg 审计产物进独立目录
    # 才能让"整目录删除是否安全"这个运维问题保持简单（详见 ingestion.persist_ingest_chunks）。
    ingest_chunks_dir: str = ""


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
    # llamaindex_pg 摄入切块是否落盘审计 JSON（仅排查/审计，不参与检索）。
    # 与 lightrag.save_ingest_chunks 两个后端各自独立；此处特意带 pg_ 前缀避免读串。
    # JSON 记全量 chunks + node_ids（= data_kb_chunks 主键），便于把审计文本 join 回 PG 行。
    # .env: CHUNKING__SAVE_PG_INGEST_CHUNKS=true（默认开）
    save_pg_ingest_chunks: TruthyBool = True


class ContextPolicyConfig(BaseModel):
    """轮内上下文预算策略（第二批）。

    调研依据 arXiv:2508.21433（The Complexity Trap, JetBrains）：Observation Masking 相对
    Raw 成本减半、解题率持平；纯摘要引发 trajectory elongation。最优是 hybrid（掩码为主、
    摘要为最后手段）。M 取小值（max_iterations=10，远小于 SWE-agent 的 250 轮）。
    .env 前缀 CONTEXT_POLICY__*。
    """

    # 总开关（dormant）：上下文管理重构后 loop 轮内主路径统一走 context_budget.enforce 三级级联，
    # 不再按此开关分支。保留供 eval_turn_budget 旧臂位 patch 与 context_policy.apply 路径回退。
    enabled: TruthyBool = True
    keep_recent_turns: int = 3      # 保留最近 M 轮 role=tool 原文，更早掩码（mask_old_observations，评测臂用）
    budget_chars: int = 80_000      # dormant：掩码触发点已改 token 口径（compute_budgets soft_trigger），此字段仅旧评测回退
    tool_result_max_chars: int = 6000  # 单条 tool 结果头尾保留上限（cap_tool_result）；0=不限
    summary_enabled: TruthyBool = False  # hybrid 兜底摘要（调 fast LLM，多花一次调用，默认关）
    summary_threshold: int = 4      # 被掩码轮数 >= 此值才触发摘要


class ContextBudgetConfig(BaseModel):
    """通用上下文管理：绝对阈值 + 减法留白 + 三级级联 + 反应式兜底。

    调研依据：Claude Code 五级压缩级联（microcompact->collapse->auto-compact->reactive->
    熔断）、Anthropic Context Management API 参数语义（compact 150k / tool clearing 100k
    keep=3 + exclude_tools 白名单）、context rot 论文（arXiv:2606.29718：compaction+trimming
    组合最优、阈值越低 rot 越轻但成本越高，64k 为成本可接受上界）、When Refusals Fail
    （arXiv:2512.02445：标称窗口不可信，1M 窗口模型 100k 即掉 50%+ 性能）。

    双阈值（硬天花板减法留白，预留量是绝对量不随窗口缩放；软阈值=窗口比例线与绝对上限三取 min）：
      硬天花板 = effective_window - min(max_output, output_reserve_tokens) - safety_margin_tokens
      软阈值   = min(rot_threshold_tokens, effective_window * quality_ratio, 硬天花板)
    软阈值驱动主动压缩（L1 清旧 tool 结果->L2 LLM 摘要->L3 丢最旧 20% 消息组），硬天花板
    驱动反应式兜底（413 超限紧急 L3 + 熔断）。env 前缀 CONTEXT_BUDGET__*。
    """

    token_accounting_enabled: TruthyBool = True   # system prompt 告警按 token 口径 + 逐切片分解
    cache_control_enabled: TruthyBool = False      # Anthropic：T1 稳定前缀 + 末工具加 ephemeral 断点
    system_prompt_warn_tokens: int = 2000          # system prompt token 告警阈值（≈旧 8000 字符口径）

    # -- 双阈值（绝对 token）--
    rot_threshold_tokens: int = 128000         # 软阈值上限：比例线算出过大时钳制到此（rot 论文 64k 为成本可接受上界，1M 窗口放宽到 128k）
    quality_ratio: float = 0.5                 # 软阈值比例线：effective_window × 此值（RULER 实测有效长度多为窗口 0.25-0.5，取 0.5 使 128k 模型算出 64000 与旧行为逐字相同）
    output_reserve_tokens: int = 20000         # 输出预留上限：reserve = min(max_output, 此值)
    safety_margin_tokens: int = 4096           # 安全余量：tokenizer 差异/多模态/工具结果突增缓冲

    # -- 三级级联 --
    keep_recent_turns: int = 3                 # L1 保留最近 N 轮 tool 原文（对齐 Anthropic keep=3、论文 keep-latest）
    exclude_tools: list[str] = Field(default_factory=lambda: ["ask_user"])  # L1 白名单：永不清理的 tool（不可重建输入）
    max_consecutive_failures: int = 3          # 反应式兜底连续失败熔断阈值（Claude Code MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES=3）

    # -- 窗口探测（对标 DeepTutor detect_context_window：GET /models 递归扫字段）--
    # 探测在请求热路径之外跑（启动预热 / admin 重探），结果落 data/context_window_cache.json，
    # 热路径 resolve_effective_window 只同步读缓存、零网络（见 core.agentic.window_probe）。
    probe_enabled: TruthyBool = True            # 总开关。关=启动不预热、admin 端点短路
    probe_timeout_s: float = 12.0               # GET /models 超时（对齐 DeepTutor；供应商不可达时尽快放弃）
    probe_cache_ttl_s: int = 604800             # 探测缓存 TTL（7 天；模型窗口变更频率低）

    # -- 兼容字段（保留供 eval_turn_budget 旧臂位读取；新级联统一 L1->L2->L3 不再分支）--
    coordinator_enabled: TruthyBool = False        # 回合前 plan_turn 反应式裁剪开关（默认关=旧 resolve_budget 20% 裁历史）
    carry_forward_location: str = "system_prompt"  # history_prefix=超软阈值时摘要前插历史续接
    eviction_strategy: str = "cascade"              # dormant：新级联恒走 L1->L2->L3，不再按此分支


class KbSeedConfig(BaseModel):
    """chat 进 loop 前的知识库预检索（对标 DeepTutor ``_retrieve_kb_seed_block``）。

    一次用户回合进 agent loop 前，用原问题预检索一次课程知识库，命中则把证据作为
    ``[知识库预检索]`` 块注入首轮消息——材料够时模型第 1 轮直接作答，省下「盲调多次 rag +
    多轮 LLM 反刍」的开销（LatentRAG arXiv:2605.06285 实测 thought+subquery 占 agentic RAG
    约 90% 延迟，瓶颈是轮数而非检索引擎）。只挂载在 ChatPipeline，不污染 research/quiz 共享内核。

    默认开（``KB_SEED__ENABLED=false`` 回退）；超时/失败一律降级为空串，绝不拖慢主链路。
    .env 前缀 ``KB_SEED__*``。
    """

    enabled: TruthyBool = True      # 默认开；出问题 KB_SEED__ENABLED=false 一行回退
    max_chars: int = 4000           # 对标 DeepTutor KB_SEED_CHARS_PER_KB，单次预检索注入上限
    timeout_s: float = 8.0          # asyncio.wait_for 超时降级为空串，绝不拖慢


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


class UsageTrackingConfig(BaseModel):
    """LLM 用量统计落库配置（明细 + 日汇总两级存储）。

    设计要点：
    - **默认开启**：与 cost_quota（默认关）不同，用量统计是无副作用只读账单，写库 best-effort
      （每轮一行 insert，对齐 record_learning_event），开着不损主链路性能，故默认 True。
    - **detail_retention_days**：明细行保留期（默认 90 天）；过期由 cron purge 清，日汇总永久
      保留作账单历史。
    - **expose_cost_to_student**：学生端是否显示美元成本（默认 False——成本是价目表估算而非真实
      账单，且教育场景不宜把费用推给学生；token 数始终可见，不受此开关影响）。
    .env 前缀 USAGE_TRACKING__*。
    """

    enabled: TruthyBool = True              # 总开关。关=loop 不写明细，done 不带 usage
    detail_retention_days: int = 90         # 明细保留天数（日汇总不受限，永久保留）
    expose_cost_to_student: TruthyBool = False  # 学生端气泡是否显示美元成本（token 数始终显示）


class StorageGcConfig(BaseModel):
    """磁盘派生数据保留策略（offline cron GC，对标 Bazel 磁盘缓存 GC）。

    设计要点：
    - **offline cron 而非写时记账**：Bazel 在 issue #5139 论证了「写时强制不超限」跨平台 +
      多进程共享 + 不损性能三者难兼得，改为服务空闲期后台 GC。我们多 worker 共享同一卷、
      不能让索引变慢，约束同构，照搬同一形态（每日 04:17 ARQ cron，单容器天然单例）。
    - **max_age + max_size 双口径**：max_age 删过期（ingest_chunks/parse_cache 旧条目），
      max_size 按 mtime LRU 裁剪到上限（Bazel 口径）。两者独立，先 age 后 size。
    - **dry-run 默认**：admin 端点 POST /api/admin/storage/gc?dry_run=true 默认只报不删。
    - **kb_store 只监控不清理**：删 raw 会导致无法重索引（用户明确决定）。GC 只统计其体积。
    .env 前缀 STORAGE_GC__*。
    """

    enabled: TruthyBool = True              # 总开关。关=cron 与 admin 触发均短路
    # parse_cache（内容寻址解析缓存）：双口径硬上限
    parse_cache_max_gb: float = 4.0
    parse_cache_max_age_days: int = 30
    # lightrag_store（graphml + ingest_chunks 合计）：按体积裁剪 + ingest_chunks 按年龄清
    lightrag_store_max_gb: float = 2.0
    ingest_chunks_max_age_days: int = 7  # 审计 JSON 保留天数（与后端无关，lightrag/pg 两路共用）
    # llamaindex_pg 审计 JSON（data/ingest_chunks 独立根）体积上限；年龄口径复用上面的
    # ingest_chunks_max_age_days（语义本就是「审计 JSON 保留天数」）。该目录无 graphml，
    # 故 GC 不做孤儿回收（课程删后最多残留 0 字节空目录）。
    pg_ingest_chunks_max_gb: float = 1.0
    # uploads（学生聊天附件）：体积裁剪孤儿；按年龄清默认关（删旧附件会让历史会话图片失效）
    uploads_max_gb: float = 2.0
    uploads_max_age_days: int = 0           # 0=关闭按年龄清，只清孤儿 + 体积裁剪
    orphan_sweep_enabled: TruthyBool = True # 回收无 DB 归属的 lightrag_store/uploads 残留目录
    disk_warn_pct: int = 75                 # 整卷水位告警阈值（%）；采样 task 据此打 warning


class ResearchConfig(BaseModel):
    """深度研究「前置澄清 + 大纲确认」配置（均在同 turn 内用共享 waiter 暂停问学生）。

    - clarify_enabled：rephrase 是否挂 ask_user（仅 WS 入口有效；HTTP/IM 无 waiter 不挂）。
    - clarify_wait_timeout_s：ask_user 等待学生回复的硬超时（秒）。超时不挂死，走 "User skipped."
      续跑（见 turn_runtime._wait_for_user_reply + loop._format_reply）。该超时是 turn_runtime
      共享 waiter 的全局上限，chat/quiz 的 ask_user 与下方大纲确认同样受益（不无限挂住 task/WS）。
      clarify_enabled=False 时取 0=不超时（保留旧「无限等」行为，便于关澄清做对照）。
    - clarify_max_questions：单次澄清最多问几个（提示词层约束，见 pipeline.yaml rephrase.system）。
    - outline_confirm_enabled：decompose 后是否出大纲确认卡（仅 WS 入口有效；HTTP 无 waiter 不出），
      学生过目/增删改子主题后再执行 research/reporting。默认 True（采纳 deeptutor 行为）。
      超时复用 clarify_wait_timeout_s，到点走「用原大纲续跑」不挂死。
    .env 前缀 RESEARCH__*。
    """

    clarify_enabled: TruthyBool = True
    clarify_wait_timeout_s: int = 120
    clarify_max_questions: int = 3
    outline_confirm_enabled: TruthyBool = True


class MCPConfig(BaseModel):
    """MCP server 行为配置（部署级）。

    server 进程的连接/配置见 core/mcp/config.py（data/mcp.json，运行时可改）；本组只放
    不宜 per-server 的部署级开关。.env 前缀 MCP__*。
    """

    # admin 改 mcp.json 后，非受理请求的 worker 最多经此秒数感知到（main._mcp_reload_loop 轮询）。
    # 项目无 Redis pub/sub 基建，「改配置」是低频操作，30s 最终一致够用；为它建跨进程消息通道属过度设计。
    reload_interval_s: int = 30


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
    # 默认 0.0 = 不过滤 = 行为完全不变；阈值口径是 DashScope qwen3-rerank 的 relevance_score
    #（rerank_adapter.py 默认模型已从下线的 gte-rerank-v2 迁移至 qwen3-rerank），与裸余弦相似度
    # 分布不同，不能照搬 0.5。生效前置条件：enable_rerank=True（默认）且 rerank_model_func
    # 已挂载（需 EMBEDDING__API_KEY），缺 key 时阈值静默失效，但无命中哨兵修复不受影响。
    min_rerank_score: float = 0.0
    # ingestion
    save_ingest_chunks: TruthyBool = True
    ingest_chunks_subdir: str = "ingest_chunks"
    ingest_chunks_snapshot: TruthyBool = False
    ingest_batch_size: int = 16
    max_async: int = 8
    # 每 worker 进程可驻留的 LightRAG 实例数上限（per-worker，不再按 worker 数整除；
    # 依据见 Settings.lightrag_lru_capacity_per_worker）。PG 后端下单实例常驻仅数十 MB
    # （NetworkX 图 + asyncpg 句柄），per-worker 6 → 4 worker 峰值 ~1.2-1.9GB，贴边 4GB 机器。
    lru_capacity: int = 6
    # 空闲实例回收：实例超过 instance_idle_ttl_sec 秒未被任何检索/索引引用 → 后台 reaper
    # finalize 释放（对标连接池 pool_recycle）。解决「学生点开一门课看一眼就走」的一次性
    # 实例白占槽位（S3-FIFO 论文里的一次性访问对象，我们用 TTL 比 probation 队列更便宜）。
    # instance_reap_interval_sec 是 reaper 循环周期。
    instance_idle_ttl_sec: int = 900
    instance_reap_interval_sec: int = 60
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


class RerankConfig(BaseModel):
    """PG 检索链路的 Cross-Encoder 精排（检索阶段增强，非索引）。

    复用 DashScope 托管 qwen3-rerank（云端不装 torch，与「稳态内存 ~0.4GB」预算一致；
    本地 BAAI/bge-reranker 需 torch+sentence-transformers，与既定内存预算冲突）。
    凭证复用 ``embedding.api_key``（同属 DashScope），不新增 key 字段。

    **默认关**（``enabled=False``）：行为零变化。论文支撑「精排是高确定性收益项」
    （arXiv:2606.21553 HotpotQA RRF top-20→cross-encoder→top-5，p<0.001），但 rerank 必须
    在自己数据上验证（不能假设普遍有效），故 eval 有正向收益后再开（``RERANK__ENABLED=true``）。
    .env 前缀 ``RERANK__*``。
    """

    enabled: TruthyBool = False             # 默认关；eval 验证有收益后开
    model: str = "qwen3-rerank"             # gte-rerank-v2 已于 2026-05-30 下线，迁移至此
    # 送进精排的候选数：RRF 融合后最多 dense_top_k+bm25_top_k 条（默认 20+20=40），截前
    # candidate_top_n=20 送精排，尾部候选原样保留不参与重排序（见 core/rag/rerank.py）。
    # 20 是按单门课程语料量级定的档（arXiv:2606.21553 用 RRF top-20）；可调。
    candidate_top_n: int = 20
    # 请求 token 上限（按 字符数/1.5 粗估累加，超则丢尾部候选避 400）。低于 qwen3-rerank 的
    # 120,000 留余量；默认规模（20 doc × ~1200 token）只用到约 25,000，是「有人把
    # candidate_top_n 调到 100」的兜底。
    max_request_tokens: int = 100_000
    timeout_s: float = 5.0                  # 查询热路径上的外部 API 往返，超时必须短；超时降级回 RRF
    # 相关性阈值：精排后丢弃 relevance_score 低于此值的候选（全滤掉->空结果->_execute_rag 拒答）。
    # 默认 0.0 = 不过滤 = 行为完全不变。口径与 lightrag.min_rerank_score 一致（同为 qwen3-rerank
    # 的 relevance_score），故两后端可共用同一数值。生效前置：enabled=True 且有 EMBEDDING__API_KEY。
    # 不可照搬裸余弦/RRF 分数的 0.5——cross-encoder 分数跨 query 不可比，阈值须在自己数据上标定。
    min_score: float = 0.0


class LlamaParseConfig(BaseModel):
    """LlamaParse / LlamaIndex（图像 PDF / 扫描件解析）。"""

    cloud_api_key: SecretStr = SecretStr("")  # 兼容 LLAMA_CLOUD_API_KEY 语义
    parse_api_key: SecretStr = SecretStr("")  # 作 cloud_api_key 的 fallback


class PdfConfig(BaseModel):
    """PDF 摄入解析配置（backend 二选一，无降级）。

    PDF 解析与 DOCX 同构（extract_pdf_sections → section 列表 → sentence_splitter），
    backend 决定走哪个解析器；切块统一 sentence_splitter，不再有 pdf_structured 策略。
    .env: PDF__BACKEND / PDF__DO_OCR / PDF__OCR_PROVIDER。

    - mupdf（默认）：PyMuPDF 轻量纯文本 + 章节(get_toc) + 页码，不装 torch；无 OCR/表格原子化。
    - docling（opt-in）：Docling 单引擎全包——版面/表格/标题/页码 + OCR（RapidOCR 后端）。
      需 pip install docling rapidocr-onnxruntime（[parse-docling] extra，首次运行模型下到 ~/.cache/docling）；
      未装时该 PDF 报错跳过（不降级 mupdf）。
    """

    backend: str = "mupdf"  # mupdf(默认) | docling；二选一，选定失败则跳过该文件，不降级
    do_ocr: TruthyBool = True  # 仅 docling：扫描件 OCR
    ocr_provider: str = "rapid"  # 仅 docling：rapid | easyocr


class ParsingConfig(BaseModel):
    """文档解析层配置（parsing/ 引擎，替代 worker 内 Docling 单例）。

    默认 mineru_api（托管 API）：云端不装 torch，worker 稳态内存从 ~2.5GB 降到 ~0.4GB
    （省下的正是给 LightRAG 腾的空间）。docling 为可选自托管引擎（装 parse-docling
    extra，数据不出域）。单引擎，失败即报错不降级（低质量兜底比失败更糟）。
    .env 前缀 PARSING__*（如 PARSING__ENGINE / PARSING__MINERU_API_KEY）。
    """

    engine: str = ""  # 空=用原 file_routing 解析（docling/mupdf，行为零变化）；mineru_api=换托管 API（去 torch，需配 MINERU_API_KEY）
    # 索引前过滤参考文献章节（References/参考文献/Bibliography）：论文类占 30%+ 字符，
    # 既省 embedding 成本又减少检索噪声（碎片化引用挤占 top_k）；教案类无此章节零影响。
    drop_references: TruthyBool = True
    # 有图注的图片是否也调 VLM 生成描述。默认 False = 现有行为（marker 后紧跟 Figure/图/表
    # 图注则删 marker、不调 VLM）。开启后每张图一次 VLM 调用，把「图 1-1 股权架构图」这类
    # 纯标题图里的结构/数字也转成文本 inline 进 markdown（pgvector 无多模态 embedding，图必须
    # 转文本才能被向量召回）。desc_cache 按图片字节 sha256 去重，重复索引不重花；并发度复用
    # image_ingest.semaphore。注：与 chunking.inline_image_descriptions（MinerU markdown 图 vs
    # DOCX/PDF 嵌入图两条独立管线，LightRAG 与 llamaindex_pg 后端均读取后者）是两条独立管线，
    # 同一课程若两者都开会对同一张图各调一次 VLM、且两套 desc_cache 不互通。
    # .env: PARSING__IMAGE_VLM_ALWAYS=true
    image_vlm_always: TruthyBool = False
    # ── MinerU 托管 API（https://mineru.net/apiManage/docs）──
    mineru_api_key: StrippedSecret = SecretStr("")
    mineru_base_url: str = "https://mineru.net"
    mineru_model: str = "vlm"  # vlm(MinerU2.5,表格/公式强项) | pipeline
    mineru_language: str = "ch"  # ch | en | ...
    enable_formula: TruthyBool = True
    enable_table: TruthyBool = True
    poll_interval: int = 5  # 轮询解析结果间隔（秒）
    poll_timeout: int = 1800  # 单文件解析轮询超时（秒，30 分钟）
    max_file_pages: int = 200  # MinerU 单文件页数上限
    max_file_mb: int = 200  # MinerU 单文件大小上限（MB）


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
    """Mem0 记忆：矛盾检测 / 跳过模式 / 批量刷新。"""

    # 检索相关性过滤阈值（P0-C）：mem0 V3 search 原生支持 threshold，>0 时过滤低相关噪声。
    # 默认 0.0=不过滤=行为完全不变。当前部署 mem0 版本对 threshold 关键字的兼容性未实测
    # （Docker 起不来），build_memory_context 已加 TypeError 自适应降级兜底。
    search_threshold: float = 0.0
    conflict_detect_enabled: TruthyBool = True
    conflict_similarity_threshold: float = 0.85
    conflict_min_days_gap: int = 7
    add_skip_patterns: str = "好的,谢谢,嗯,ok,明白了,收到,好,知道了,继续"
    add_min_length: int = 10
    flush_max_turns: int = 3
    flush_idle_timeout: float = 120.0
    flush_scan_interval: int = 30
    # L3 巩固（Phase 3）：热路径累计 importance 超此阈值 → enqueue consolidate_memory。
    # 对齐 Generative Agents 的 reflection 累计触发；quiz 里程碑无视阈值直接触发。
    consolidation_importance_threshold: float = 0.7


class SummaryConfig(BaseModel):
    """Session L2 摘要。"""

    window_size: int = 5
    buffer_size: int = 2
    compress_interval: int = 3
    # 跨进程压缩锁：生产 gunicorn -w 4 下，模块级 asyncio.Lock 只够单进程；
    # Redis per-session SET NX EX 防多 worker 重复烧 LLM。memory://（测试）或关闭时降级不加锁。
    distributed_lock_enabled: TruthyBool = True
    lock_ttl: int = 60
    # v2 显著度淘汰（替旧硬编码 _MAX_ITEMS_PER_LIST=5 的「按条数丢最旧」）：
    # 注入 token 预算上限（粗估，中文 1 字≈1 token），按 salience 降序保留到预算耗尽。
    # 默认 1200 与旧实现 5类×5项×~50字 注入量级相当，不显著膨胀。
    inject_token_budget: int = 1200
    # salience 半衰期（秒）：会话内（分钟级）recency≈1，主要由 kind_weight+n 决定；
    # 跨会话（小时/天级）老条目衰减。默认 1 小时给长会话内有意义的新鲜度区分。
    salience_half_life_s: int = 3600
    # 每 kind 保留上限（单值槽 fact/next_step 自然≤key 数，多值槽据此防一类占满预算）。
    max_items_per_kind: int = 8


class QuestionConfig(BaseModel):
    """FAQ 热点缓存阈值（admin FAQ 缓存，见 api/admin.py、scripts/seed_faq_cache.py）。"""

    faq_cache_threshold: int = 3


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
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llamaparse: LlamaParseConfig = Field(default_factory=LlamaParseConfig)
    pdf: PdfConfig = Field(default_factory=PdfConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
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
    context_budget: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)
    kb_seed: KbSeedConfig = Field(default_factory=KbSeedConfig)
    cost_quota: CostQuotaConfig = Field(default_factory=CostQuotaConfig)
    usage_tracking: UsageTrackingConfig = Field(default_factory=UsageTrackingConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    storage_gc: StorageGcConfig = Field(default_factory=StorageGcConfig)

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
        self.paths.db_path = self.paths.db_path or os.path.join(BASE_DIR, "data", "sessions.db")
        self.paths.question_log_dir = self.paths.question_log_dir or os.path.join(BASE_DIR, "logs", "question")
        self.paths.lightrag_workdir = self.paths.lightrag_workdir or os.path.join(BASE_DIR, "lightrag_store")
        self.paths.kb_store_dir = self.paths.kb_store_dir or os.path.join(BASE_DIR, "kb_store")
        self.paths.tutorbot_workspace_dir = self.paths.tutorbot_workspace_dir or os.path.join(BASE_DIR, "data", "tutorbot")
        self.paths.search_config_path = self.paths.search_config_path or os.path.join(BASE_DIR, "data", "search_config.json")
        self.paths.mcp_config_path = self.paths.mcp_config_path or os.path.join(BASE_DIR, "data", "mcp.json")
        self.paths.mcp_sessions_dir = self.paths.mcp_sessions_dir or os.path.join(BASE_DIR, "data", "sessions")
        self.paths.output_cards_path = self.paths.output_cards_path or os.path.join(BASE_DIR, "data", "output_cards.json")
        self.paths.parse_cache_dir = self.paths.parse_cache_dir or os.path.join(BASE_DIR, "data", "parse_cache")
        self.paths.ingest_chunks_dir = self.paths.ingest_chunks_dir or os.path.join(BASE_DIR, "data", "ingest_chunks")
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
        # context_window：catalog 显式配置覆盖 .env（空=不覆盖，走三级解析后两级）。需转 int。
        _cw_raw = str(cat.get("context_window", "")).strip()
        if _cw_raw:
            try:
                self.llm.context_window = int(_cw_raw)
            except ValueError:
                pass
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
    def lightrag_lru_capacity_per_worker(self) -> int:
        """每个 worker 进程可驻留的 LightRAG 实例数上限。

        PG 后端下 KV/Vector/DocStatus 已搬出进程（instance_pool.py:_IS_POSTGRES 桥接的
        PGKVStorage/PGVectorStorage/PGDocStatusStorage），单实例常驻只剩 NetworkX 图 +
        asyncpg 句柄（数十 MB），**不再按 worker 数整除**——旧的整除口径是 PG 改造之前的
        OOM 护栏（那时单实例把全文/向量/LLM 缓存全装进程内存，数百 MB），前提已失效。

        非 PG（默认内存后端，如 SQLite 部署）单实例仍是数百 MB，保留旧整除口径防 OOM：
        per-worker 不整除的话 `4 × 6 × 300MB ≈ 7.2GB` 会直接打爆 4GB 机器，是真实脚坑。

        ARQ worker（python -m arq，独立进程）拥有各自的模块级 _instances 池；它一次只跑
        一个 indexing job，索引结束 15 分钟后被 idle reaper 收走（见 instance_pool.py），
        故不需要引入「web 还是 worker」的进程角色判定——这是 TTL 相比纯计数上限的额外收益。
        """
        if not self.db.url.get_secret_value().startswith("postgres"):
            return max(2, self.lightrag.lru_capacity // max(1, self.backend_workers))
        return max(2, self.lightrag.lru_capacity)

    def validate_runtime_workers(
        self, known_workers: int | None = None
    ) -> bool:
        """运行时校验 ``backend_workers`` 与真实 worker 进程数是否一致（M-24）。

        ``backend_workers`` 驱动 DB 连接池缩放（database.py）、LLM 熔断阈值缩放
        （reliability.py），以及非 PG 模式下的 LightRAG LRU 容量整除
        （lightrag_lru_capacity_per_worker）——这些公式都假设它等于 gunicorn/uvicorn
        实际拉起的 ``-w`` 进程数。但 ``-w`` 数写在部署脚本里（Dockerfile/compose），
        与 ``.env`` 的 ``BACKEND_WORKERS`` 是两套手动维护的值，一旦不同步，缩放公式就会
        算偏（例如真 8 worker 但 env 写 4，实际连接数会翻倍打爆 Postgres）。本方法在
        进程启动期（lifespan）跑一次，把不一致暴露为显式告警。

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
