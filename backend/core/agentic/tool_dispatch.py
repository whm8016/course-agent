"""并行工具分发器。

接收一轮 LLM 返回的 ToolCall 列表，并发执行（上限 MAX_PARALLEL），
向 StreamBus 发送 tool_call / tool_result 事件，
并返回 DispatchOutcome（含 role=tool 消息列表及 pause 信息）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.agentic.types import DispatchOutcome, ToolCall
from core.observability import log_flow
from core.observability.metrics import observe_tool_call
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

MAX_PARALLEL = 8  # 单轮最大并发工具调用数


async def dispatch_tool_calls(
    tool_calls: list[ToolCall],
    course_id: str,
    enabled_tools: list[str],
    stream: StreamBus,
    user_id: str = "",
    rag_mode: str = "",
) -> DispatchOutcome:
    """并行执行工具调用，返回 DispatchOutcome。

    若某个工具返回 pause_for_user，则 DispatchOutcome.pause=True，
    loop 负责挂起并等待用户回复。

    rag_mode 非空时，注入到 rag 工具调用（覆盖 retrieve_context 默认 mode），
    供用户在对话界面选择检索模式；其它工具不受影响。
    """
    from core.agent.tool_registry import execute_tool

    sem = asyncio.Semaphore(MAX_PARALLEL)

    async def _run_one(tc: ToolCall) -> dict[str, Any]:
        async with sem:
            await stream.emit({"type": "tool_call", "tool": tc.name, "input": tc.arguments})
            _t = time.perf_counter()
            status = "ok"
            try:
                # rag 工具注入用户选择的检索模式（mode 由用户每请求选，不由 LLM 决定）
                call_kwargs = dict(tc.arguments)
                if tc.name == "rag" and rag_mode:
                    call_kwargs["mode"] = rag_mode
                result = await execute_tool(tc.name, course_id=course_id, user_id=user_id, **call_kwargs)
                content = str(result.content) if result else "（无返回结果）"
                # pause_for_user 先存到 role=tool 消息的 _pause 临时字段，
                # gather 后统一检测，不在并发路径里修改共享状态
                pause_payload = result.pause_for_user if result else None
            except Exception as exc:
                logger.warning("工具 '%s' 执行失败: %s", tc.name, exc)
                content = f"（工具执行失败：{exc}）"
                pause_payload = None
                status = "error"
            _elapsed = int((time.perf_counter() - _t) * 1000)
            log_flow("tool.result", tool_name=tc.name, status=status,
                     elapsed_ms=_elapsed, result_chars=len(content))
            observe_tool_call(tool_name=tc.name, status=status, elapsed_ms=_elapsed)
            if _elapsed > 2000:
                log_flow("tool.slow", level=logging.WARNING,
                         tool_name=tc.name, elapsed_ms=_elapsed)

            await stream.emit({"type": "tool_result", "tool": tc.name, "content": content[:2000]})
            msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": content,
            }
            if pause_payload is not None:
                msg["_pause_payload"] = pause_payload
            return msg

    raw_results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls], return_exceptions=True)

    tool_messages: list[dict[str, Any]] = []
    tools_used: list[str] = []
    pause = False
    pause_payload: dict[str, Any] | None = None
    pause_tool_call_id: str | None = None

    for tc, res in zip(tool_calls, raw_results):
        if isinstance(res, Exception):
            logger.warning("工具 '%s' gather 异常: %s", tc.name, res)
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": f"（工具执行失败：{res}）",
            })
        else:
            # 提取并清理临时 _pause_payload 字段
            pp = res.pop("_pause_payload", None)
            tool_messages.append(res)
            tools_used.append(tc.name)
            if pp is not None and not pause:
                pause = True
                pause_payload = pp
                pause_tool_call_id = tc.id

    return DispatchOutcome(
        tool_messages=tool_messages,
        tools_used=tools_used,
        pause=pause,
        pause_payload=pause_payload,
        pause_tool_call_id=pause_tool_call_id,
    )
