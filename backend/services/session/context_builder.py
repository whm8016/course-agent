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

# tiktoken 懒加载，import 失败则降级
_encoding = None

def _get_encoding():
    global _encoding
    if _encoding is not None:
        return _encoding
    try:
        import tiktoken
        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoding = False  # 标记为不可用
    return _encoding


def count_tokens(text: str) -> int:
    """tiktoken 精确计数，不可用时降级到 len // 4。"""
    if not text:
        return 0
    enc = _get_encoding()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# 按模型名推断 context window → history 预算
_MODEL_WINDOWS = {
    "qwen-max": 32768,
    "qwen-max-latest": 32768,
    "qwen-plus": 131072,
    "qwen-plus-latest": 131072,
    "qwen-turbo": 131072,
    "qwen-turbo-latest": 131072,
    "qwen-long": 1_000_000,
    "deepseek-chat": 65536,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
}
_HISTORY_BUDGET_RATIO = 0.35
_DEFAULT_WINDOW = 32768
_MIN_BUDGET = 2000


def resolve_budget(model: str | None = None) -> int:
    """按模型名推断 history token 预算。"""
    window = _MODEL_WINDOWS.get(model or "", _DEFAULT_WINDOW)
    return max(_MIN_BUDGET, int(window * _HISTORY_BUDGET_RATIO))


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