"""
agentic 调度引擎
================

Web Chat 所有能力共用的调度内核。

对外导出：
    run_agent_loop  — tool_calls 驱动的 while 循环（主入口）
    LoopOutcome     — run_agent_loop 返回的结果数据类
    ToolCall        — LLM 返回的工具调用（解析后）
"""
from core.agentic.loop import run_agent_loop
from core.agentic.types import LoopOutcome, ToolCall

__all__ = ["run_agent_loop", "LoopOutcome", "ToolCall"]
