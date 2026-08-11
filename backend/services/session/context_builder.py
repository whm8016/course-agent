"""
Context Builder
===============

按 token 预算裁剪对话历史，防止超出模型上下文窗口。

设计原则：
- 从最新消息向前取，保证近期上下文优先保留。
- 优先用 tiktoken 精确计数，不可用时降级到 len//4。
- 支持按模型名自动推断 history token 预算。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.observability import log_flow

logger = logging.getLogger(__name__)

# count_tokens / _get_encoding 已下沉到 core.agentic.context_window（治 core/agentic
# 反向 import services 的层级违规）。此处 re-export 保向后兼容（test 等仍用
# `from services.session.context_builder import count_tokens`）。
from core.agentic.context_window import count_tokens  # noqa: E402,F401


# 历史预算占有效窗口的比例（resolve_budget 委托 context_window 解析窗口后按此折算）。
# 留作模块常量而非配置项：历史预算是「窗口的多少给历史」的工程默认，非用户可调旋钮。
_HISTORY_BUDGET_RATIO = 0.20
_MIN_BUDGET = 2000


def resolve_budget(model: str | None = None) -> int:
    """按模型有效窗口推断 history token 预算（= 有效窗口 × 20%，下限 _MIN_BUDGET）。

    窗口解析委托 ``context_window.resolve_effective_window``（三级：显式配置 -> 模型名模式 ->
    heuristic 兜底+告警）。默认路径（coordinator_enabled=False）仍走此函数裁历史，行为与旧实现
    逐字节一致（已知模型窗口不变；未知模型改走 heuristic 但默认 max_tokens=8192 -> 32768=旧值，
    仅多了告警）。
    """
    from core.agentic.context_window import resolve_effective_window

    window = resolve_effective_window(model)
    return max(_MIN_BUDGET, int(window * _HISTORY_BUDGET_RATIO))


def advertised_window(model: str | None = None) -> int:
    """模型有效上下文窗口（token）。委托 ``context_window.resolve_effective_window`` 三级解析。

    保留函数签名不破坏调用方；未知模型不再静默回落 32768，改走 heuristic 并告警（见
    context_window.resolve_effective_window）。
    """
    from core.agentic.context_window import resolve_effective_window

    return resolve_effective_window(model)


class ContextBuilder:
    """按 token 预算从近到远裁剪对话历史。"""

    def __init__(self, max_history_tokens: int = 4000) -> None:
        self._max_tokens = max_history_tokens

    def build(
        self,
        history: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        budget = max_tokens if max_tokens is not None else self._max_tokens
        before = len(history)
        result: list[dict[str, Any]] = []
        tokens_used = 0

        for msg in reversed(history):
            raw = msg.get("content")
            if isinstance(raw, list):
                tokens = count_tokens(json.dumps(raw, ensure_ascii=False))
            else:
                tokens = count_tokens(str(raw or ""))
            if tokens_used + tokens > budget:
                break
            result.insert(0, msg)
            tokens_used += tokens

        if before != len(result):
            log_flow("context.history_trim", logger=logger,
                     before=before, after=len(result),
                     tokens_used=tokens_used, budget=budget)

        return result

    def estimate_tokens(self, text: str) -> int:
        return count_tokens(text)