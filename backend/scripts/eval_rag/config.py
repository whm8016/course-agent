"""RAG 评测系统配置。

读取项目主配置（backend/config.py）中的 DashScope / LightRAG 参数，
在此基础上定义评测专用的默认值。
"""
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 让 import 能找到 backend 根目录（scripts/eval_rag/ → backend/）
# ---------------------------------------------------------------------------
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env：本模块用裸 os.getenv（不走 pydantic-settings），需自行 load_dotenv
# 才能保证任何入口（python -c / run_eval / CI）都读到 .env 凭据。override=False 不覆盖
# 入口已预设的环境变量（尊重 run_eval / main 等上层预设）。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(_BACKEND_ROOT) / ".env", override=False)
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# 课程与模式
# ---------------------------------------------------------------------------
COURSE_ID: str = "circuit_analysis"

MODES: list[str] = ["naive", "local", "global", "mix"]

# ---------------------------------------------------------------------------
# RAGAS 指标分层（业界标准：检索 → 生成 → 鲁棒性 → 领域定制）
# tier1 检索质量 | tier2 生成质量 | tier3 鲁棒性 + 领域定制（AspectCritic 自定义）
# 注：context_precision/recall/faithfulness 等均需 LLM 评判，不存在"零 LLM 成本"指标
# ---------------------------------------------------------------------------
METRICS_TIERS: dict[str, list[str]] = {
    "tier1_retrieval": ["context_precision", "context_recall"],
    "tier2_generation": ["faithfulness", "factual_correctness"],
    "tier3_robustness": ["noise_sensitivity"],
    "tier3_domain": ["teaching_accuracy", "safety"],
}
# 全量指标（按 tier 顺序展开）
METRICS_ALL: list[str] = [m for tier in METRICS_TIERS.values() for m in tier]

# ---------------------------------------------------------------------------
# LLM / Embedding（RAGAS 评测内部使用）
# 阅卷(RAGAS 指标)用 DeepSeek 强推理模型，合成出题用快/省模型，embedding 用千问。
# .env 实际变量名是双下划线 LLM__* / EMBEDDING__*（pydantic settings 嵌套字段），
# 旧的单下划线名（DASHSCOPE_API_KEY 等）保留作 fallback 兼容。
# ---------------------------------------------------------------------------
# DeepSeek chat LLM（阅卷与合成共用同一 key/base_url，模型型号分开）
LLM_API_KEY: str = (
    os.getenv("LLM__API_KEY") or os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY") or ""
)
LLM_BASE_URL: str = (
    os.getenv("LLM__BASE_URL") or os.getenv("LLM_BASE_URL")
    or "https://api.deepseek.com/v1"
)
# 阅卷(RAGAS 指标)用强推理模型，合成出题用快/省模型
JUDGE_LLM_MODEL: str = os.getenv("EVAL_JUDGE_MODEL", "deepseek-v4-pro")
GEN_LLM_MODEL: str = (
    os.getenv("LLM__TEXT_MODEL") or os.getenv("LLM_TEXT_MODEL") or "deepseek-v4-flash"
)
# RAGAS LLM 输出 token 上限：ragas 默认 max_tokens=1024，NER 抽实体 / faithfulness 分解 claims 等
# 结构化输出远超 1024 → 输出被截断 → instructor IncompleteOutputException + tenacity 疯狂重试
# （这正是 NERExtractor 龟速 422s/it 的根因，不是真推理慢）。合成与阅卷均给 8192：合成 NER 抽实体
# 输出最长；阅卷 faithfulness 对信息量大的题要拆大量 claims 逐条判断，4096 实测仍会截断（Job
# 截断 → 该题分数异常 → 拉低均值），故阅卷也升到 8192。
GEN_LLM_MAX_TOKENS: int = int(os.getenv("EVAL_GEN_LLM_MAX_TOKENS", "8192"))
JUDGE_LLM_MAX_TOKENS: int = int(os.getenv("EVAL_JUDGE_LLM_MAX_TOKENS", "8192"))

# Embedding（千问 DashScope，与 LLM 分属不同 provider）
EMBED_MODEL: str = os.getenv("EMBEDDING__MODEL") or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
EMBED_API_KEY: str = (
    os.getenv("EMBEDDING__API_KEY") or os.getenv("EMBEDDING_API_KEY") or LLM_API_KEY
)
EMBED_BASE_URL: str = (
    os.getenv("EMBEDDING__BASE_URL") or os.getenv("EMBEDDING_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# DashScope text-embedding-v3 需显式 dimensions：不传时（尤其 langchain 旧路径）可能返回
# 异常向量，导致 answer_relevancy 内部 cosine 相似度全 0（Bug 2 根因）
EMBED_DIMENSIONS: int = int(os.getenv("EMBEDDING__EMBEDDING_DIM") or os.getenv("EMBED_DIMENSIONS", "1024"))

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
EVAL_DIR: Path = Path(__file__).resolve().parent
CACHE_DIR: Path = EVAL_DIR / "cache"
RESULTS_DIR: Path = EVAL_DIR / "results"
DATASET_PATH: Path = EVAL_DIR / "qa_dataset.json"

# ---------------------------------------------------------------------------
# 查询间隔（秒）——避免 DashScope 限流
# ---------------------------------------------------------------------------
QUERY_DELAY: float = float(os.getenv("EVAL_QUERY_DELAY", "1.5"))

# 检索 top_k：对齐生产 tool_registry._execute_rag 的默认值（5）。production-parity 同此。
EVAL_TOP_K: int = int(os.getenv("EVAL_TOP_K", "5"))

# 成本估算单价（每 1M tokens，美元）。仅按 total_tokens 粗估（input/output 混合平均）
COST_PER_M_TOKENS: float = float(os.getenv("EVAL_COST_PER_M_TOKENS", "0.8"))

# CI 质量门禁阈值（run_eval 结束时检查，任一不达标 → exit 1 阻断 CI）
# 分数类：<metric>_min（下限）/ <metric>_max（上限，如 noise_sensitivity 越低越好）
# 延迟类：latency__<field>__<stat>（双下划线分隔，避免与 metric 名冲突）
QUALITY_GATES: dict[str, float] = {
    "faithfulness_min": 0.85,
    "context_precision_min": 0.80,
    "context_recall_min": 0.75,
    "factual_correctness_min": 0.70,
    "noise_sensitivity_max": 0.20,
    "teaching_accuracy_min": 0.95,  # AspectCritic 1/0，均值即 pass_rate
    "safety_min": 1.00,
    "latency__total_ms__p95": 5000,
}
