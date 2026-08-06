"""chat KB Seed 预检索单测（chat_pipeline.retrieve_kb_seed + loop._build_messages 注入口）。

纯 mock execute_tool，不依赖真实知识库/LLM。覆盖：
  1. 命中 → 拼 header + 证据，emit tool_call/tool_result，strategy 固定 fact
  2. 无命中（success=False）→ 返空串，emit tool_call 但不发 tool_result（前端可单独渲染 tool_call）
  3. 超时（wait_for 触发 TimeoutError）→ 返空串降级，不崩
  4. enabled_tools 无 rag → 短路，execute_tool 根本不被调用、无任何事件
  5. 消融开关 metadata["kb_seed"]=False → 强制关闭（即便 settings 默认开）
  6. 超长证据 → 按 max_chars 截断
  7. loop 层：extra_context 非空时 user 消息前多一条 role=system
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from core.agent.tool_protocol import ToolResult
from core.context import UnifiedContext
from core.stream_bus import StreamBus

HEADER = "[知识库预检索]\n证据如下："


def _ctx(**kwargs) -> UnifiedContext:
    base: dict[str, Any] = dict(
        course_id="c1",
        user_id="u1",
        user_message="什么是基尔霍夫电压定律？",
        enabled_tools=["rag"],
    )
    base.update(kwargs)
    return UnifiedContext(**base)


async def _events(bus: StreamBus) -> list[dict[str, Any]]:
    """收集 StreamBus 已 emit 的事件（对齐 test_agent_loop._collect_events 范式）。"""
    if not bus._closed:
        await bus.close()
    return [e.to_dict() async for e in bus.subscribe()]


# ---------------------------------------------------------------------------
# 1. 命中
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_hit():
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx()
    bus = StreamBus()
    result = ToolResult(content="KVL：沿闭合回路各元件电压代数和为零。", success=True)
    with patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=result)) as m:
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert out.startswith("[知识库预检索]")
    assert "KVL" in out
    # 固定 strategy=fact（避开每次跑 2 次 LightRAG 查询的 graph_augmented_retrieve）
    assert m.call_args.kwargs["strategy"] == "fact"
    assert m.call_args.kwargs["query"] == ctx.user_message
    events = await _events(bus)
    assert any(e["type"] == "tool_call" and e["tool"] == "rag" for e in events)
    assert any(e["type"] == "tool_result" for e in events)


# ---------------------------------------------------------------------------
# 2. 无命中
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_no_hit():
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx()
    bus = StreamBus()
    result = ToolResult(content="（知识库中未检索到相关内容。）", success=False)
    with patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=result)):
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert out == ""
    events = await _events(bus)
    # tool_call 在调用前已 emit；无命中则不发 tool_result（前端 ToolCallRow 不依赖 tool_result 配对）
    assert any(e["type"] == "tool_call" for e in events)
    assert not any(e["type"] == "tool_result" for e in events)


# ---------------------------------------------------------------------------
# 3. 超时降级
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_timeout(monkeypatch):
    from settings import get_settings

    # 压低超时阈值，让 wait_for 真触发 TimeoutError（被 except Exception 兜住降级空串）
    monkeypatch.setattr(get_settings().kb_seed, "timeout_s", 0.01)
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx()
    bus = StreamBus()

    async def _slow(*a, **kw):
        await asyncio.sleep(1.0)
        return ToolResult(content="不该到达", success=True)

    with patch("core.agent.tool_registry.execute_tool", AsyncMock(side_effect=_slow)):
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert out == ""
    events = await _events(bus)
    assert not any(e["type"] == "tool_result" for e in events)


# ---------------------------------------------------------------------------
# 4. 未挂 rag → 短路
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_rag_not_enabled():
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx(enabled_tools=["web_search"])  # 不含 rag
    bus = StreamBus()
    with patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=ToolResult(content="x"))) as m:
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert out == ""
    m.assert_not_called()  # 短路：根本不调工具
    assert await _events(bus) == []  # 也不 emit 任何事件


# ---------------------------------------------------------------------------
# 5. 消融开关 metadata 覆盖（强制关闭）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_metadata_override_disables():
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx(metadata={"kb_seed": False})  # 即便 settings.kb_seed.enabled 默认 True
    bus = StreamBus()
    with patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=ToolResult(content="x"))) as m:
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert out == ""
    m.assert_not_called()


# ---------------------------------------------------------------------------
# 6. 超长证据截断
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_kb_seed_truncates_overlong():
    from core.capabilities.chat_pipeline import retrieve_kb_seed

    ctx = _ctx()
    bus = StreamBus()
    long_text = "证据" * 3000  # 6000 字 > max_chars(4000)
    with patch("core.agent.tool_registry.execute_tool",
               AsyncMock(return_value=ToolResult(content=long_text, success=True))):
        out = await retrieve_kb_seed(ctx, bus, header=HEADER)

    assert "...[已截断]" in out
    # 截断后总长应明显短于 header + 原始 6000 字
    assert len(out) < len(HEADER) + len(long_text)


# ---------------------------------------------------------------------------
# 7. loop 层：extra_context 注入为 user 前的独立 system 消息
# ---------------------------------------------------------------------------
def test_build_messages_extra_context():
    from core.agentic.loop import _build_messages

    ctx = _ctx(user_message="你好")
    # 无 extra_context：system(prompt) + user
    roles = [m["role"] for m in _build_messages("SYS", ctx, binding="dashscope")]
    assert roles == ["system", "user"]

    # 有 extra_context：user 前多一条 system（紧贴 user）
    msgs = _build_messages("SYS", ctx, binding="dashscope", extra_context="预检索证据")
    assert [m["role"] for m in msgs] == ["system", "system", "user"]
    assert msgs[-2]["content"] == "预检索证据"
    assert msgs[-1]["content"] == "你好"
