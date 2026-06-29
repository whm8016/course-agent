"""ContextFilter：把 contextvars 字段自动注入每条 LogRecord。

只需在 main.py 把这个 filter 挂到根 handler 上，90+ 个模块的 logger
无需任何改动就能在 JSON 输出里自动带上 turn_id / user_id / course_id 等字段。
"""
from __future__ import annotations

import logging

from .context import get_context_fields


class ContextFilter(logging.Filter):
    """将当前请求上下文字段合并到每条 LogRecord 中。"""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_context_fields().items():
            # 不覆盖调用方通过 extra={} 显式设置的同名字段
            if not hasattr(record, key):
                setattr(record, key, value)
        return True
