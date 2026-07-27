"""上下文预算策略消融配置（第四批-2）。

对照 arXiv:2508.21433（The Complexity Trap, JetBrains, SWE-bench Verified ×5 模型）：
Observation Masking 相对 Raw 成本减半、解题率持平或略高；纯摘要引发 trajectory
elongation（多跑 13-15% 轮）；最优是 hybrid（掩码为主、摘要为最后手段）。

复用 eval_rag / eval_capabilities 的 sys.path + .env 注入范式（scripts/ 无 __init__.py，
是 namespace 包，python -m scripts.eval_context.run_eval 可直接跑）。
"""
import os
import sys
from pathlib import Path

# 让 import 找到 backend 根（scripts/eval_context/ → backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env（override=False 尊重上层预设；与 eval_rag/config.py 同款）
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
# 消融配置：arm × M（keep_recent_turns）
# arm 对照论文四臂：
#   raw          真基线，完全不裁（可能撑爆 context——这正是 complexity trap 要展示的成本爆炸）
#   masking      按轮掩码（保留最近 M 轮 tool 原文，更早替换占位），不摘要
#   summary_only 窗口外每轮 tool 结果 LLM 摘要塞回（测 H2：trajectory elongation）
#   hybrid       先掩码，被掩码轮数≥阈值才整体摘要（论文最优组合）
# M 扫描：masking 在 M=2/3/5 下看「保留窗口大小」对成本/解题率的 trade-off
# ---------------------------------------------------------------------------
CONTEXT_POLICY_CONFIGS: list[dict] = [
    {"label": "raw",          "arm": "raw",          "keep_recent_turns": 3},
    {"label": "masking_M2",   "arm": "masking",      "keep_recent_turns": 2},
    {"label": "masking_M3",   "arm": "masking",      "keep_recent_turns": 3},
    {"label": "masking_M5",   "arm": "masking",      "keep_recent_turns": 5},
    {"label": "summary_only", "arm": "summary_only", "keep_recent_turns": 3},
    {"label": "hybrid",       "arm": "hybrid",       "keep_recent_turns": 3},
]

# 单条 case 间延迟（避免 rate limit）；默认 0，env 可调
QUERY_DELAY: float = float(os.getenv("EVAL_CONTEXT_DELAY", "0"))
# loop 最大轮数（论文 max 250；本项目 max_iterations=10 远小，故默认 10）
MAX_ITERATIONS: int = int(os.getenv("EVAL_CONTEXT_MAX_ITER", "10"))
