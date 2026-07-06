"""步骤3：loop.py binding 透传测试。

验证 _build_messages 与 run_agent_loop 把 binding 透传到
prepare_multimodal_messages / _create_with_image_fallback——修复用户选了非默认
binding（如 anthropic）profile 时，图片直注格式拼错 + Stage-2 剥图容错误判的 bug。

asyncio_mode=auto，async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.agentic.loop import _build_messages, run_agent_loop
from core.agentic.types import LoopOutcome
from core.attachment import Attachment
from core.context import UnifiedContext
from core.stream_bus import StreamBus


def _make_chunk(content: str = ""):
    delta = MagicMock()
    delta.content = content or None
    delta.tool_calls = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _async_iter(items):
    for item in items:
        yield item


# ── _build_messages 直接透传（最直接验证 anthropic 图片格式修复点）──────────────


def test_build_messages_threads_binding_to_multimodal():
    ctx = UnifiedContext(
        user_message="看这张图",
        attachments=[Attachment(type="image", base64="AAAA", mime_type="image/png")],
    )
    with patch("core.agentic.loop.prepare_multimodal_messages") as m:
        msgs = _build_messages("SYS", ctx, binding="anthropic")
    assert m.call_args.kwargs["binding"] == "anthropic"
    assert m.call_args.args[1] is ctx.attachments
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


def test_build_messages_openai_binding():
    ctx = UnifiedContext(
        attachments=[Attachment(type="image", base64="AAAA", mime_type="image/png")],
    )
    with patch("core.agentic.loop.prepare_multimodal_messages") as m:
        _build_messages("SYS", ctx, binding="openai")
    assert m.call_args.kwargs["binding"] == "openai"


# ── run_agent_loop 端到端透传 binding 到 _create_with_image_fallback ────────────


async def test_run_agent_loop_threads_binding_to_create():
    """run_agent_loop(binding=anthropic) → eff_binding 流到 _create_with_image_fallback 第3参。"""
    ctx = UnifiedContext(
        user_message="hi",
        attachments=[Attachment(type="image", base64="AAAA", mime_type="image/png")],
    )
    bus = StreamBus()

    captured: dict = {}

    async def _fake_create(llm_client, create_kwargs, binding, model, circuit_breaker=None):
        captured["binding"] = binding
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter([_make_chunk("答案")])
        return mock

    with patch("core.agentic.loop._create_with_image_fallback", side_effect=_fake_create):
        outcome = await run_agent_loop(
            context=ctx,
            stream=bus,
            system_prompt="SYS",
            tool_schemas=None,
            binding="anthropic",
        )
    assert isinstance(outcome, LoopOutcome)
    assert captured["binding"] == "anthropic"
    await bus.close()


async def test_run_agent_loop_default_binding_back_compat():
    """不传 binding 时回退全局 LLM_BINDING（向后兼容，run_agent_loop 既有调用方不破）。"""
    ctx = UnifiedContext(user_message="hi")
    bus = StreamBus()

    async def _fake_create(llm_client, create_kwargs, binding, model, circuit_breaker=None):
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter([_make_chunk("ok")])
        return mock

    with patch("core.agentic.loop._create_with_image_fallback", side_effect=_fake_create):
        outcome = await run_agent_loop(
            context=ctx, stream=bus, system_prompt="SYS", tool_schemas=None
        )
    assert isinstance(outcome, LoopOutcome)
    await bus.close()
