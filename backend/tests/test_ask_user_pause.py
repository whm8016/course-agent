"""ask_user 端到端：executor → dispatch pause → loop 暂停 await waiter → 回复后恢复。

ask_user 基建早已搭好（schema/executor/dispatch→pause/turn_runtime waiter/WS submit_reply），
本测试锁定其"通电"后的关键行为：
  1. _execute_ask_user 返回 pause_for_user payload；
  2. dispatch_tool_calls 拿到 ask_user 工具调用 → DispatchOutcome.pause=True；
  3. run_agent_loop 持 waiter 时，遇到 ask_user 暂停、emit ask_user_card、await 用户回复，
     把回复写回 role=tool 后继续，最终正常 finish（真实 executor + 真实 dispatch，仅 mock LLM）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.loop import run_agent_loop
from core.agentic.tool_dispatch import dispatch_tool_calls
from core.agentic.types import ToolCall
from core.context import UnifiedContext
from core.stream_bus import StreamBus


# ── 1. executor ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_ask_user_returns_pause_payload():
    from core.agent.tool_registry import _execute_ask_user
    r = await _execute_ask_user(
        questions=[{"id": "q1", "prompt": "想要哪个范围？", "options": ["代数", "几何"]}],
        intro="需要澄清一下",
    )
    assert r.pause_for_user is not None
    assert r.pause_for_user["intro"] == "需要澄清一下"
    assert r.pause_for_user["questions"][0]["id"] == "q1"


# ── 2. dispatch → pause ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_ask_user_sets_pause():
    """dispatch_tool_calls 执行真实 ask_user executor → DispatchOutcome.pause=True。"""
    stream = MagicMock()
    stream.emit = AsyncMock()
    tc = ToolCall(
        id="call_1", name="ask_user",
        arguments={"questions": [{"id": "q1", "prompt": "范围？"}]},
        arguments_str='{"questions":[]}',
    )
    outcome = await dispatch_tool_calls(
        [tc], course_id="c", enabled_tools=["ask_user"], stream=stream, user_id="u",
    )
    assert outcome.pause is True
    assert outcome.pause_tool_call_id == "call_1"
    assert outcome.pause_payload["questions"][0]["id"] == "q1"


# ── 3. loop pause → waiter → resume（真实 executor + dispatch，仅 mock LLM）────

def _chunk(content: str = "", tool_call: dict | None = None):
    """构造一个 OpenAI 流式 chunk 的 mock（同 test_agent_loop 的 _make_chunk 风格）。"""
    delta = MagicMock()
    delta.content = content or None
    delta.reasoning_content = None
    delta.tool_calls = None
    if tool_call:
        ft = MagicMock()
        ft.index = 0
        ft.id = tool_call.get("id", "call_1")
        ft.function = MagicMock()
        ft.function.name = tool_call["name"]
        ft.function.arguments = tool_call.get("arguments", "")
        delta.tool_calls = [ft]
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _aiter(items):
    for it in items:
        yield it


@pytest.mark.asyncio
async def test_loop_asks_user_then_resumes():
    """loop 遇 ask_user → emit ask_user_card + await waiter；回复后写回 role=tool，继续到 finish。"""
    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="出几道题",
        enabled_tools=["ask_user"], mode="quiz",
    )
    waiter = AsyncMock(return_value={"text": "数学范围", "answers": []})
    ctx.metadata["wait_for_user_reply"] = waiter
    bus = StreamBus()

    # 第 1 次 LLM 调用：返回 ask_user 工具调用；第 2 次：返回最终答案
    call1 = MagicMock()
    call1.__aiter__ = lambda self: _aiter([_chunk(tool_call={
        "id": "call_1", "name": "ask_user",
        "arguments": '{"questions":[{"id":"q1","prompt":"哪个范围？"}]}',
    })])
    call2 = MagicMock()
    call2.__aiter__ = lambda self: _aiter([_chunk("好的，就出数学题。")])

    with patch("core.agentic.loop._default_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(side_effect=[call1, call2])
        outcome = await run_agent_loop(
            context=ctx, stream=bus, system_prompt="你是助教。",
            tool_schemas=[{"type": "function", "function": {"name": "ask_user", "parameters": {"type": "object"}}}],
        )

    waiter.assert_awaited_once()              # 暂停等待用户回复，恰好一次
    assert "数学题" in outcome.final_text      # 回复后恢复，正常 finish
    assert "ask_user" in outcome.tools_used

    # ask_user_card 事件确实 emit 给前端
    if not bus._closed:
        await bus.close()
    types = []
    async for ev in bus.subscribe():
        types.append(ev.to_dict()["type"])
    assert "ask_user_card" in types
