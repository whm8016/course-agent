"""L2 摘要 v2 评测配置。

按 LongMemEval（arXiv:2410.10813）四类能力构造中文教学用例，对比「新 v2 合并（slot key
+ max(ts) 覆盖 + salience 淘汰）」与「旧 v1 基线（精确去重 + combined[-5:] 硬截断）」。

不依赖真实 LLM：每个 case 的 increments.items 是「假定 LLM 已正确抽取」的条目，eval 只验
合并/淘汰/渲染管线（即本次改造的部分）。LLM 抽取质量不在本评测范围。

四类：
  knowledge_update  单值槽改口后当前值唯一（旧实现必然失败 -> 回归基线）
  multi_session     多增量累积，fact 稳定不被挤
  temporal          resolved 消除 / 新鲜度淘汰
  abstention        空输入/全 resolved/未知 kind/预算紧张等边界
"""
import os
import sys
from pathlib import Path

# 让 import 找到 backend 根（scripts/eval_l2_summary/ -> backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env（override=False 尊重上层预设；与 eval_turn_budget/config.py 同款）
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

# 评测用时间戳基线（模拟一个会话内的轮次时间；每增量 +TS_STEP 秒）
TS_BASE: float = float(os.getenv("EVAL_L2_TS_BASE", "1700000000.0"))
TS_STEP: float = float(os.getenv("EVAL_L2_TS_STEP", "60.0"))

# 淘汰参数（与 settings.SummaryConfig 默认一致；env 可调便于消融）
TOKEN_BUDGET: int = int(os.getenv("EVAL_L2_TOKEN_BUDGET", "1200"))
MAX_PER_KIND: int = int(os.getenv("EVAL_L2_MAX_PER_KIND", "8"))
HALF_LIFE_S: int = int(os.getenv("EVAL_L2_HALF_LIFE_S", "3600"))
