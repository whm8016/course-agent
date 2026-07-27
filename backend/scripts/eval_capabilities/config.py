"""四能力评测配置（基于 Inspect AI）。

复用 eval_rag 的 provider 思路（deepseek-v4-pro 阅卷），扩展异家族 judge ensemble
（deepseek + qwen 投票，治 LLM-as-judge 的自增强偏置，见 arXiv:2410.02736）。

provider 走 Inspect AI 的 openai-api/<service>/<model>（OpenAI 兼容 Chat Completions）：
  - openai/<model> 默认走 Responses API，DeepSeek 不支持会 404，故必须用 openai-api
  - service=deepseek → 读 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL，本模块从 LLM__* 注入
  - 异家族 ensemble：EVAL_JUDGE_MODELS 逗号分隔多个不同家族 spec，取均值（arXiv:2410.02736）
"""
import os
import sys
from pathlib import Path

# 让 import 能找到 backend 根目录（scripts/eval_capabilities/ → backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env（与 eval_rag/config.py 同款，override=False 尊重上层预设）
try:
    from dotenv import load_dotenv

    load_dotenv(Path(_BACKEND_ROOT) / ".env", override=False)
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
EVAL_DIR: Path = Path(__file__).resolve().parent
DATASETS_DIR: Path = EVAL_DIR / "datasets"
RESULTS_DIR: Path = EVAL_DIR / "results"

# ---------------------------------------------------------------------------
# Inspect AI judge 模型（异家族 ensemble）
# ---------------------------------------------------------------------------
# 模型 spec：openai-api/<service>/<model> 走 Chat Completions（openai/<model> 走 Responses，
# DeepSeek 不支持）。env EVAL_JUDGE_MODELS 逗号分隔可配多个不同家族做 ensemble。
# 默认单 judge：deepseek-v4-pro（对齐 eval_rag 阅卷档）。
# 配异家族时用不同 service/厂商（如 openai-api/deepseek/... + openai-api/qwen/...），
# 防同家族 judge 91% 一致却全错的 agreement trap。
_DEFAULT_JUDGE = os.getenv("EVAL_JUDGE_MODELS") or "openai-api/deepseek/deepseek-v4-pro"
JUDGE_MODELS: list[str] = [m.strip() for m in _DEFAULT_JUDGE.split(",") if m.strip()]

# OpenAI 兼容 endpoint（阅卷用，对齐 eval_rag 的 deepseek 配置）
JUDGE_BASE_URL: str = (
    os.getenv("EVAL_JUDGE_BASE_URL")
    or os.getenv("LLM__BASE_URL") or os.getenv("LLM_BASE_URL")
    or "https://api.deepseek.com/v1"
)
JUDGE_API_KEY: str = (
    os.getenv("EVAL_JUDGE_API_KEY")
    or os.getenv("LLM__API_KEY") or os.getenv("LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY") or ""
)
# 注入 openai-api/<service>/<model> 读取的环境变量：service=deepseek →
# DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL（开箱即用，无需 CLI 传 key）
if JUDGE_API_KEY and not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = JUDGE_API_KEY
if JUDGE_BASE_URL and not os.environ.get("DEEPSEEK_BASE_URL"):
    os.environ["DEEPSEEK_BASE_URL"] = JUDGE_BASE_URL

# ---------------------------------------------------------------------------
# pass^k 可靠性：同一 case 重复跑 k 次（solve/quiz 用，τ-bench 思路）
# chat/research 默认 1（单轮）；阅卷已占评测绝大多数耗时，pass^k 会乘以 k，故默认保守
# ---------------------------------------------------------------------------
PASS_K: dict[str, int] = {
    "chat": 1,
    "quiz": int(os.getenv("EVAL_PASS_K_QUIZ", "1")),
    "solve": int(os.getenv("EVAL_PASS_K_SOLVE", "1")),
    "research": int(os.getenv("EVAL_PASS_K_RESEARCH", "1")),
}

# ---------------------------------------------------------------------------
# 四能力质量门禁阈值（capability → metric → 下限）
# 留 buffer：LLM-judge 二分类与人类仅 κ=0.3-0.5（10-20% 误判），门禁比目标松一档
# ---------------------------------------------------------------------------
QUALITY_GATES: dict[str, dict[str, float]] = {
    # faithfulness（忠于检索材料）待加自定义 scorer（从 _trace 抽检索内容 + judge），
    # 暂不入门禁——配了无 scorer 产出的指标会让 check_gate 永远跳过它（形同虚设）
    "chat": {"accuracy": 0.70},
    "quiz": {"validity": 0.90, "quality": 0.70},
    "solve": {"trajectory_legal": 1.00, "answer_correctness": 0.70},
    "research": {"race_overall": 0.70, "fact": 0.80},
}
