"""
Stream Types
============

类型化流事件，替代 plain dict 传递。

StreamEventType  — 所有合法事件类型枚举
StreamEvent      — 单个事件的结构化载体

设计原则：
- payload 字段兜住所有事件特有数据，避免为每种事件单独定义字段。
- to_dict() / from_dict() 保证与现有 SSE/WS 格式完全向前兼容。
- emit() 同时接受 StreamEvent 和 dict（见 stream_bus.py），存量代码无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StreamEventType(str, Enum):
    # 内容类
    THINKING = "thinking"
    THINKING_CHUNK = "thinking_chunk"
    TOKEN = "token"
    ANSWER = "answer"

    # 工具类
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # 流水线阶段类
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    PROGRESS = "progress"

    # 结果类
    RESULT = "result"
    DONE = "done"
    QUIZ = "quiz"

    # 错误 / 会话
    ERROR = "error"
    SESSION = "session"

    # 交互（ask_user 工具预留）
    WAIT_FOR_INPUT = "wait_for_input"


@dataclass
class StreamEvent:
    """单个流事件。

    Attributes:
        type:    事件类型。
        source:  发出事件的模块名（capability / tool / orchestrator）。
        payload: 事件携带的全部业务字段（除 type / source 外）。
    """

    type: StreamEventType
    source: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """转为向前兼容的 plain dict，用于 SSE/WS JSON 序列化。"""
        d: dict[str, Any] = {"type": self.type.value}
        if self.source:
            d["source"] = self.source
        d.update(self.payload)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamEvent:
        """从 plain dict 构造 StreamEvent，用于 emit(dict) 向后兼容。"""
        type_str = str(d.get("type", "token"))
        try:
            event_type = StreamEventType(type_str)
        except ValueError:
            event_type = StreamEventType.TOKEN
        source = str(d.get("source", ""))
        payload = {k: v for k, v in d.items() if k not in ("type", "source")}
        return cls(type=event_type, source=source, payload=payload)
