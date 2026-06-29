"""agentic 模块共用的数据类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """一次 LLM 返回的工具调用，经流式分片累积后解析得到。"""

    id: str           # OpenAI tool_call_id，用于匹配 role=tool 消息
    name: str         # 工具名称，如 "rag"、"web_search"
    arguments: dict[str, Any]   # 解析后的参数字典
    arguments_str: str = ""     # 原始 JSON 字符串，用于 OpenAI tool message 的 arguments 字段


@dataclass
class RoundResult:
    """一次 LLM 流式调用的累积结果。"""

    content: str                            # 模型输出的文本内容
    tool_calls: list[ToolCall] = field(default_factory=list)  # 本轮工具调用列表
    streamed_live: bool = False             # 是否已通过 live_sink 实时流式透传（真流式）
    elapsed_ms: int = 0                     # 本轮 LLM 调用总耗时（ms）
    ttft_ms: int | None = None              # 首 token 延迟（ms）
    reasoning: str = ""                     # 本轮推理（思考）全文，无则空串（非推理模型恒空，优雅降级）

    @property
    def has_tool_calls(self) -> bool:
        """本轮是否包含工具调用。"""
        return bool(self.tool_calls)


@dataclass
class DispatchOutcome:
    """dispatch_tool_calls() 返回的结果，替代原来的 (tool_messages, tools_used) 元组。"""

    tool_messages: list[dict[str, Any]]
    tools_used: list[str]
    pause: bool = False                         # ask_user 工具触发时为 True
    pause_payload: dict[str, Any] | None = None # ask_user 的卡片数据，发给前端
    pause_tool_call_id: str | None = None       # 对应的 tool_call_id，用于写回 role=tool content


@dataclass
class LoopOutcome:
    """run_agent_loop() 返回的最终结果。"""

    final_text: str                         # 最终答案文本
    rounds: int                             # 实际执行的 LLM 调用轮次
    tools_used: list[str] = field(default_factory=list)  # 本轮用到的工具名称（去重）
    completed: bool = True                  # 是否正常结束（False 表示被中断）
