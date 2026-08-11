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
    # 出题 pipeline 的逐题事件：quiz_question（一道题生成成功）、quiz_question_error
    # （单题失败/校验无效——M-8 单题容错）。此前 pipeline 已 emit 这两个 type，但枚举
    # 里缺失，from_dict 把未知 type 降级成 token，前端（ChatWindow.tsx 监听 quiz_question）
    # 实际收不到，逐题事件静默丢失。补枚举让事件原样透传。
    QUIZ_QUESTION = "quiz_question"
    QUIZ_QUESTION_ERROR = "quiz_question_error"

    # 错误 / 会话
    ERROR = "error"
    SESSION = "session"

    # 交互（ask_user 工具）
    WAIT_FOR_INPUT = "wait_for_input"
    # ask_user 暂停时 loop emit 的问题卡片事件。补枚举前的同款坑：from_dict 把未知 type
    # 降级成 token（见上方 QUIZ_QUESTION 注释），前端永远收不到 ask_user_card。补上才透传。
    ASK_USER_CARD = "ask_user_card"
    # 深度研究 decompose 后的大纲确认卡片（用户过目/编辑子主题再执行 research）。
    # 同款坑：不补枚举则 from_dict 把 outline_card 降级成 token，前端收不到。补上才透传。
    OUTLINE_CARD = "outline_card"


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
