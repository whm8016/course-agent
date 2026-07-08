"""M-4：agent loop 最后一轮仍返回 tool_calls（或 content 为空）时 final_text 兜底，不让用户得到空回复。

两个触发场景：
  A. 轮次预算耗尽：每一轮（含最后一轮强制禁工具轮）模型都吐 tool_calls / 空 content，
     循环走完没走到 else 分支赋值 → final_text 原本会是空串。
  B. 最后一轮走 else 但 result.content 为空 → final_text 直接被赋空串。

修复后：循环出口检测到 final_text 为空，回退到本轮已生成的旁白；旁白也空则用明确兜底语，
保证 emit 的 answer 事件永不为空。
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.stream_bus import StreamBus


def _make_chunk(content: str = "", tool_calls: list[dict] | None = None):
    delta = MagicMock()
    delta.content = content or None
    delta.tool_calls = None
    delta.reasoning_content = None
    if tool_calls:
        tc_list = []
        for i, tc in enumerate(tool_calls):
            fake_tc = MagicMock()
            fake_tc.index = i
            fake_tc.id = tc.get("id", f"call_{i}")
            fake_tc.function = MagicMock()
            fake_tc.function.name = tc.get("name", "")
            fake_tc.function.arguments = tc.get("arguments", "")
            tc_list.append(fake_tc)
        delta.tool_calls = tc_list
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _async_iter(items):
    for item in items:
        yield item


async def _collect(bus: StreamBus) -> list[dict[str, Any]]:
    if not bus._closed:
        await bus.close()
    return [e.to_dict() async for e in bus.subscribe()]


@pytest.fixture
def ctx():
    return UnifiedContext(course_id="c", user_id="u1", user_message="q", enabled_tools=["rag"])


@pytest.fixture
def bus():
    return StreamBus()


# ── 场景 A：轮次预算耗尽，每一轮都吐 tool_calls（含最后一轮强制禁工具轮仍残留）──────────

@pytest.mark.asyncio
async def test_loop_budget_exhausted_with_side_note_fallback(ctx, bus):
    """每轮都调工具，最后一轮（禁工具）模型仍吐 tool_calls 但同时给了一段旁白文字。
    循环耗尽走完，final_text 应回退到那段旁白，而非空。"""
    # 工具轮：有 tool_calls + 旁白
    tool_with_note = [
        _make_chunk(content="正在检索资料…", tool_calls=[{
            "id": "c1", "name": "rag", "arguments": json.dumps({"query": "x"}),
        }]),
    ]
    # 最后一轮（禁工具 schemas=None）：模型仍吐 tool_calls + 又一段旁白
    final_with_note = [
        _make_chunk(content="综合以上检索结果。", tool_calls=[{
            "id": "c2", "name": "rag", "arguments": json.dumps({"query": "y"}),
        }]),
    ]

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if kwargs.get("tools"):
            mock.__aiter__ = lambda self: _async_iter(tool_with_note)
        else:
            mock.__aiter__ = lambda self: _async_iter(final_with_note)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "检索结果"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx, stream=bus, system_prompt="你是助教",
            tool_schemas=[{"type": "function",
                           "function": {"name": "rag", "description": "检索", "parameters": {}}}],
            max_iterations=2,
        )

    # 修复前：final_text 会是空串（循环耗尽，从未走 else 赋值）
    # 修复后：回退到最后一段非空旁白
    assert outcome.final_text.strip() != "", "final_text 不应为空（M-4 兜底）"
    assert "检索结果" in outcome.final_text or "综合" in outcome.final_text

    events = await _collect(bus)
    answer_events = [e for e in events if e["type"] == "answer"]
    assert answer_events and answer_events[-1]["content"].strip() != ""


# ── 场景 B：最后一轮走 else（无 tool_calls）但 content 为空 ─────────────────────

@pytest.mark.asyncio
async def test_loop_empty_content_final_round_fallback(ctx, bus):
    """最后一轮（禁工具）模型只返回空 content、无 tool_calls（走 else 分支）。
    修复前：final_text = result.content = ""，用户得到空 answer。
    修复后：出口兜底，回退到之前工具轮的旁白。"""
    tool_with_note = [
        _make_chunk(content="这是有用的旁白。", tool_calls=[{
            "id": "c1", "name": "rag", "arguments": json.dumps({"query": "x"}),
        }]),
    ]
    # 最后一轮：空 content，无 tool_calls
    empty_final = [_make_chunk(content="")]

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if call_count == 1:
            mock.__aiter__ = lambda self: _async_iter(tool_with_note)
        else:
            mock.__aiter__ = lambda self: _async_iter(empty_final)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "检索结果"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx, stream=bus, system_prompt="你是助教",
            tool_schemas=[{"type": "function",
                           "function": {"name": "rag", "description": "检索", "parameters": {}}}],
            max_iterations=2,
        )

    # 修复前：空串；修复后：回退到工具轮旁白
    assert outcome.final_text.strip() != "", "final_text 不应为空（M-4 兜底）"
    assert "旁白" in outcome.final_text


# ── 场景 C：全程无旁白也无最终文字（极端），兜底到明确提示语 ─────────────────────

@pytest.mark.asyncio
async def test_loop_all_empty_uses_explicit_fallback(ctx, bus):
    """全程无任何非空 content（既无工具旁白也无答案），兜底到明确提示语。"""
    tool_chunks = [_make_chunk(content="", tool_calls=[{
        "id": "c1", "name": "rag", "arguments": json.dumps({"query": "x"}),
    }])]
    empty_final = [_make_chunk(content="")]

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if kwargs.get("tools"):
            mock.__aiter__ = lambda self: _async_iter(tool_chunks)
        else:
            mock.__aiter__ = lambda self: _async_iter(empty_final)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "检索结果"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx, stream=bus, system_prompt="你是助教",
            tool_schemas=[{"type": "function",
                           "function": {"name": "rag", "description": "检索", "parameters": {}}}],
            max_iterations=2,
        )

    # 兜底提示语非空且可读
    assert outcome.final_text.strip() != ""
    assert "重试" in outcome.final_text or "未能生成" in outcome.final_text

    events = await _collect(bus)
    answer_events = [e for e in events if e["type"] == "answer"]
    assert answer_events and answer_events[-1]["content"].strip() != ""
