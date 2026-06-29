"""统一结构化流程日志辅助函数。

使用方式：
    from core.observability.flow import log_flow

    log_flow("agent_loop.round", iteration=2, has_tools=True, elapsed_ms=840)
    log_flow("chat.guardrail", safe=True, risk_type="none")

输出（JSON）：
    {
      "level": "INFO", "stage": "agent_loop.round",
      "turn_id": "abc-123",   # 由 ContextFilter 自动注入
      "iteration": 2, "has_tools": true, "elapsed_ms": 840
    }

约定：
- stage 命名：{domain}.{action}，例如 chat.guardrail / agent_loop.llm_round
- 字符串字段会自动截断并以 _chars + _head 两个子字段写入（避免日志爆炸）
- 数值/布尔/列表/字典直接写入
- elapsed_ms 是保留关键字，始终作为顶层字段
"""
from __future__ import annotations

import logging
from typing import Any

_HEAD = 400

_flow_logger = logging.getLogger("flow")


def _squash_ws(s: str) -> str:
    return " ".join((s or "").split())


def _clip(s: str, n: int = _HEAD) -> str:
    t = _squash_ws(s)
    return t if len(t) <= n else t[: n - 1] + "…"


def log_flow(
    stage: str,
    *,
    logger: logging.Logger | None = None,
    elapsed_ms: int | None = None,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """输出一条结构化流程 log。

    Args:
        stage:      流程阶段标识，如 "chat.guardrail"。
        logger:     使用的 logger；默认用模块级 "flow" logger。
        elapsed_ms: 耗时（毫秒），作为顶层字段。
        level:      log 级别，默认 INFO；可传 logging.WARNING / logging.DEBUG。
        **fields:   任意额外字段；字符串自动截断为 _chars + _head 两个子字段。
    """
    lg = logger or _flow_logger
    if not lg.isEnabledFor(level):
        return

    extra: dict[str, Any] = {"stage": stage}
    if elapsed_ms is not None:
        extra["elapsed_ms"] = elapsed_ms

    for key, val in fields.items():
        if val is None:
            continue
        if isinstance(val, (int, float, bool)):
            extra[key] = val
        elif isinstance(val, (list, dict)):
            extra[key] = val
        else:
            s = str(val)
            if len(s) <= 100:
                # short enum-like fields (status, mode, risk_type, etc.) stored as-is
                extra[key] = s
            else:
                # long text — truncate to avoid log explosion
                extra[f"{key}_chars"] = len(s)
                extra[f"{key}_head"] = _clip(s)

    lg.log(level, "stage=%s", stage, extra=extra, stacklevel=2)
