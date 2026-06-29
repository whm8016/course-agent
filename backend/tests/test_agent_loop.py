"""core/agentic/loop.py 单元测试。

使用 mock LLM 客户端，不发起真实 API 请求。
覆盖四个场景：
  1. 直接 finish（无工具调用）
  2. 一轮工具调用 + 一轮 finish（RAG 典型路径）
  3. 轮次预算耗尽 → 强制 finish
  4. 工具执行异常时循环不崩溃
"""
from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.agentic.loop import run_agent_loop
from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.stream_bus import StreamBus


# ---------------------------------------------------------------------------
# 辅助：构造假的流式 LLM 响应
# ---------------------------------------------------------------------------

def _make_chunk(content: str = "", tool_calls: list[dict] | None = None):
    """构造一个 OpenAI 流式 chunk 的 mock 对象。"""
    delta = MagicMock()
    delta.content = content or None
    delta.tool_calls = None
    delta.reasoning_content = None  # 显式设置，避免 MagicMock 默认行为

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
    """将列表包装成异步生成器。"""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx():
    return UnifiedContext(
        course_id="test_course",
        user_id="u1",
        user_message="什么是快速排序？",
        enabled_tools=["rag"],
    )


@pytest.fixture
def bus():
    return StreamBus()


async def _collect_events(bus: StreamBus) -> list[dict[str, Any]]:
    """收集 StreamBus 已 emit 的事件，统一转为 plain dict 方便断言。"""
    if not bus._closed:
        await bus.close()
    events = []
    async for event in bus.subscribe():
        events.append(event.to_dict())
    return events


# ---------------------------------------------------------------------------
# 场景 1：直接 finish，无工具调用
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_round_no_tools(ctx, bus):
    """模型直接给出答案，不调用任何工具。"""
    finish_chunks = [
        _make_chunk("快速排序是"),
        _make_chunk("一种高效的"),
        _make_chunk("排序算法。"),
    ]

    mock_stream = MagicMock()
    mock_stream.__aiter__ = lambda self: _async_iter(finish_chunks)

    with patch("core.agentic.loop._default_client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="你是课程助教。",
            tool_schemas=None,  # 不挂载任何工具
        )

    assert isinstance(outcome, LoopOutcome)
    assert "快速排序" in outcome.final_text
    assert outcome.rounds == 1
    assert outcome.tools_used == []

    events = await _collect_events(bus)
    event_types = [e["type"] for e in events]
    assert "token" in event_types
    assert "answer" in event_types
    assert "done" in event_types


# ---------------------------------------------------------------------------
# 场景 2：一轮工具调用 + 一轮 finish
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_then_finish(ctx, bus):
    """模型先调用 rag 检索，获得结果后输出最终答案。"""
    # 第一轮：模型返回 tool_calls（调用 rag）
    tool_call_chunks = [
        _make_chunk(tool_calls=[{
            "id": "call_abc",
            "name": "rag",
            "arguments": json.dumps({"query": "快速排序"}),
        }]),
    ]
    # 第二轮：模型基于检索结果给出答案
    finish_chunks = [
        _make_chunk("根据知识库，"),
        _make_chunk("快速排序时间复杂度为 O(n log n)。"),
    ]

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if call_count == 1:
            mock.__aiter__ = lambda self: _async_iter(tool_call_chunks)
        else:
            mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "快速排序平均时间复杂度 O(n log n)，最坏 O(n^2)。"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="你是课程助教。",
            tool_schemas=[{
                "type": "function",
                "function": {"name": "rag", "description": "检索知识库", "parameters": {}},
            }],
        )

    assert "rag" in outcome.tools_used
    assert outcome.rounds == 2

    events = await _collect_events(bus)
    event_types = [e["type"] for e in events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "answer" in event_types
    assert "done" in event_types


# ---------------------------------------------------------------------------
# 场景 3：轮次预算耗尽，强制 finish
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_iterations_forced_finish(ctx, bus):
    """模型持续调用工具时，循环在 max_iterations 轮后强制结束。"""
    # 有工具时返回 tool_calls
    tool_chunks = [
        _make_chunk(tool_calls=[{
            "id": "call_loop",
            "name": "rag",
            "arguments": json.dumps({"query": "test"}),
        }]),
    ]
    # 无工具时（强制 finish 轮）返回文字答案
    finish_chunks = [_make_chunk("最终答案。")]

    call_count = 0
    max_iter = 3

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        # 最后一轮 kwargs 中没有 tools，说明是强制 finish 轮
        if not kwargs.get("tools"):
            mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        else:
            mock.__aiter__ = lambda self: _async_iter(tool_chunks)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "检索结果"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="你是课程助教。",
            tool_schemas=[{
                "type": "function",
                "function": {"name": "rag", "description": "检索知识库", "parameters": {}},
            }],
            max_iterations=max_iter,
        )

    assert outcome.completed is True
    assert outcome.rounds <= max_iter

    events = await _collect_events(bus)
    assert any(e["type"] == "done" for e in events)


# ---------------------------------------------------------------------------
# 场景 4：工具执行异常，循环不崩溃
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_failure_graceful(ctx, bus):
    """工具执行抛异常时，错误被安全包装，循环继续并输出最终答案。"""
    tool_chunks = [
        _make_chunk(tool_calls=[{
            "id": "call_err",
            "name": "rag",
            "arguments": json.dumps({"query": "test"}),
        }]),
    ]
    finish_chunks = [_make_chunk("仍然可以回答。")]

    call_count = 0

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        mock = MagicMock()
        if call_count == 1:
            mock.__aiter__ = lambda self: _async_iter(tool_chunks)
        else:
            mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        return mock

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(side_effect=RuntimeError("RAG 服务不可用"))):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="你是课程助教。",
            tool_schemas=[{
                "type": "function",
                "function": {"name": "rag", "description": "检索知识库", "parameters": {}},
            }],
        )

    assert outcome.completed is True

    events = await _collect_events(bus)
    # tool_result 事件应包含错误信息，而非崩溃
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert "失败" in tool_results[0]["content"] or "failed" in tool_results[0]["content"].lower()


# ---------------------------------------------------------------------------
# 场景 5：最后一轮强制 finish → 真 chunk-by-chunk 流式（live_sink 透传）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_round_live_streaming(ctx, bus):
    """最后一轮（tools=None）应通过 live_sink 逐 chunk 真流式透传，
    而非把完整答案按固定字符数切成假 token。"""
    tool_chunks = [
        _make_chunk(tool_calls=[{
            "id": "call_live",
            "name": "rag",
            "arguments": json.dumps({"query": "test"}),
        }]),
    ]
    # 最后一轮：3 个 content chunk（拼接后共 9 个字符）
    finish_chunks = [
        _make_chunk("最终答案"),
        _make_chunk("分为"),
        _make_chunk("两部分。"),
    ]

    async def _fake_create(**kwargs):
        mock = MagicMock()
        # 最后一轮 kwargs 中没有 tools，说明是强制 finish 轮
        if not kwargs.get("tools"):
            mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        else:
            mock.__aiter__ = lambda self: _async_iter(tool_chunks)
        return mock

    fake_tool_result = MagicMock()
    fake_tool_result.content = "检索结果"
    fake_tool_result.pause_for_user = None

    with patch("core.agentic.loop._default_client") as mock_client, \
         patch("core.agent.tool_registry.execute_tool", AsyncMock(return_value=fake_tool_result)):
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="你是课程助教。",
            tool_schemas=[{
                "type": "function",
                "function": {"name": "rag", "description": "检索知识库", "parameters": {}},
            }],
            max_iterations=2,  # 第二轮即最后一轮 → 触发 live_sink
        )

    assert outcome.completed is True
    assert outcome.rounds == 2
    assert outcome.final_text == "最终答案分为两部分。"

    events = await _collect_events(bus)
    token_events = [e for e in events if e["type"] == "token"]
    # 真流式：3 个 content chunk → 3 个 token 事件（一一对应）
    # 假流式则会把 9 字符按 _TOKEN_CHUNK_SIZE=8 切成 2 个，len != 3 → 断言可区分
    assert len(token_events) == 3, f"真流式应逐 chunk 发 token，实际 {len(token_events)} 个"
    assert "".join(e["content"] for e in token_events) == "最终答案分为两部分。"


# ---------------------------------------------------------------------------
# 场景 6：带图附件 → 仍用 TEXT_MODEL（对标 DeepTutor，不硬切 vision），图片照常注入
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_keeps_text_model_with_image(tmp_path, bus):
    """ctx 带图 → run_agent_loop 仍用 TEXT_MODEL（对标 DeepTutor：chat 始终同一模型，
    不因有图硬切 VISION_MODEL），但图片照常乐观注入进 messages（content 为含
    image_url 的 list）。模型不支持时由 Stage-2 降级处理（见场景 7）。"""
    from config import TEXT_MODEL
    from core.attachment import from_image_path

    img = tmp_path / "q.png"
    img.write_bytes(b"\x89PNGfake")

    ctx_img = UnifiedContext(
        course_id="test_course",
        user_id="u1",
        user_message="图里是什么？",
        attachments=[from_image_path(str(img))],
        enabled_tools=["rag"],
    )

    finish_chunks = [_make_chunk("这是一张电路图。")]
    captured: dict[str, Any] = {}

    async def _fake_create(**kwargs):
        captured["model"] = kwargs.get("model")
        captured["messages"] = kwargs.get("messages")
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        return mock

    with patch("core.agentic.loop._default_client") as mock_client:
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx_img,
            stream=bus,
            system_prompt="你是助教",
            tool_schemas=None,
        )

    # 始终 chat 主模型（TEXT_MODEL），不切 VISION_MODEL
    assert captured["model"] == TEXT_MODEL
    # 图片照常乐观注入（注入阶段不 gate）
    last_user = [m for m in captured["messages"] if m["role"] == "user"][-1]
    assert isinstance(last_user["content"], list)
    assert any(p.get("type") == "image_url" for p in last_user["content"])
    assert "电路图" in outcome.final_text


# ---------------------------------------------------------------------------
# 场景 7：模型不支持图片 → Stage-2 降级剥图，用同一 TEXT_MODEL 重试纯文本
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_loop_stage2_image_fallback(tmp_path, bus):
    """模型拒绝图片输入（异常命中 image 关键词）且 not supports_vision →
    剥掉图片用同一 TEXT_MODEL 重试纯文本成功（对标 DeepTutor Stage-2 fallback）。"""
    from config import TEXT_MODEL
    from core.attachment import from_image_path

    img = tmp_path / "q.png"
    img.write_bytes(b"\x89PNGfake")

    ctx_img = UnifiedContext(
        course_id="test_course",
        user_id="u1",
        user_message="图里是什么？",
        attachments=[from_image_path(str(img))],
        enabled_tools=[],
    )

    finish_chunks = [_make_chunk("（已转为文字作答）")]
    call_count = 0
    captured: list[dict[str, Any]] = []

    async def _fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        # deepcopy：strip_image_parts_inplace 会原地改 messages，不拷贝则两次快照指向同一已剥图 list
        captured.append({
            "model": kwargs.get("model"),
            "messages": copy.deepcopy(kwargs.get("messages")),
        })
        if call_count == 1:
            raise RuntimeError("model does not support image input")
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter(finish_chunks)
        return mock

    with patch("core.agentic.loop._default_client") as mock_client:
        mock_client.chat.completions.create = _fake_create
        outcome = await run_agent_loop(
            context=ctx_img,
            stream=bus,
            system_prompt="你是助教",
            tool_schemas=None,
        )

    # 两次 create 调用都用 TEXT_MODEL（同模型重试，不换 vision model）
    assert call_count == 2
    assert all(c["model"] == TEXT_MODEL for c in captured)
    # 第一次 messages 含图，第二次（降级剥图后）不含 image block
    first_user = [m for m in captured[0]["messages"] if m["role"] == "user"][-1]
    second_user = [m for m in captured[1]["messages"] if m["role"] == "user"][-1]
    assert any(p.get("type") == "image_url" for p in first_user["content"])
    assert not any(
        isinstance(p, dict) and p.get("type") == "image_url"
        for p in second_user["content"]
    )
    assert outcome.completed is True
