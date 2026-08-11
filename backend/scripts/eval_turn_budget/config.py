"""走完整 turn 的上下文预算 A/B 评测配置。

对比两套上下文预算方案，**必须经 ``turn_runtime.start_turn``**（不能用直调 orchestrator 的
solver，也不能复用 eval_context 的 ``set_arm`` runner--``set_arm`` 会短路 coordinator 分支）：

- ``policy_default``     线上默认：``coordinator_enabled=False`` + ``context_policy.enabled=True``。
  回合前 ``ContextBuilder(resolve_budget)`` 标称 20% 裁历史；轮内 ``mask_old_observations`` 按轮掩码。
- ``coordinator_priority`` 论文那套：``coordinator_enabled=True`` + ``eviction_strategy="priority"``
  + ``carry_forward_location="history_prefix"``。回合前 ``plan_turn`` 走 MECW；轮内
  ``evict_tool_results`` 按优先级清墓碑。

两臂共用同一份 case、同一个模型、同一个课程，只切这三/四个开关。runner 用 try/finally 临时
覆写 ``get_settings()`` 单例字段并复原（不落盘 .env），故评测必须**串行**跑（全局单例）。

复用 eval_rag / eval_context 的 sys.path + .env 注入范式（scripts/ 无 __init__.py，是 namespace
包，``python -m scripts.eval_turn_budget.run_eval`` 可直接跑）。
"""
import os
import sys
from pathlib import Path

# 让 import 找到 backend 根（scripts/eval_turn_budget/ -> backend/）
_BACKEND_ROOT = str(Path(__file__).resolve().parents[2])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

# 加载 backend/.env（override=False 尊重上层预设；与 eval_context/config.py 同款）
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
# 两臂：settings 字段覆写（runner 临时 patch + finally 复原）
# ---------------------------------------------------------------------------
# policy_default = 当前线上默认（context_policy 三段式掩码）；coordinator_priority = 论文协调器
# （plan_turn MECW + evict_tool_results 优先级驱逐）。两臂的差异点全在这几个开关上：
#   - coordinator_enabled：回合前 plan_turn vs ContextBuilder；轮内 enforce vs apply
#   - eviction_strategy：enforce 内部 priority(清墓碑) vs mask(委托旧 apply)
#   - carry_forward_location：plan_turn 有摘要时 history_prefix(前插续接) vs system_prompt
# policy_enabled 两臂都 True：coordinator 开时 loop 走 enforce 分支此项不生效，保留 True 只为
# 兜底（万一 coordinator 分支异常回落）。
ARMS: list[dict] = [
    {
        "label": "policy_default",
        "coordinator_enabled": False,
        "policy_enabled": True,
        "eviction_strategy": "mask",
        "carry_forward_location": "system_prompt",
    },
    {
        "label": "coordinator_priority",
        "coordinator_enabled": True,
        "policy_enabled": True,
        "eviction_strategy": "priority",
        "carry_forward_location": "history_prefix",
    },
]

# 单条 case 间延迟（避免 rate limit）；默认 0，env 可调
QUERY_DELAY: float = float(os.getenv("EVAL_TURN_BUDGET_DELAY", "0"))
# 评测用的 user_id（便于事后清理 usage_tracking 账单污染；空=不设）
EVAL_USER_ID: str = os.getenv("EVAL_TURN_BUDGET_USER_ID", "eval_turn_budget")
