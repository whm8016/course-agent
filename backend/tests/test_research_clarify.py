"""深度研究「前置澄清」回归（plan 阶段 1：rephrase 阶段挂 ask_user 暂停/恢复）。

验证：
1. rephrase 仅在「clarify_enabled + 注入了 wait_for_user_reply callable」时挂 ask_user；
   无 waiter（HTTP/IM 入口）或开关关时不挂——否则 LLM 一调 ask_user，loop 因无 waiter
   直接结束（loop.py ask_user 分支），research 提前夭折。
2. ask_user waiter 超时不挂死：返回 skip payload（loop._format_reply → "User skipped."，续跑），
   绝不返回 None（None 会被 loop 当 turn 取消而 break）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.research.pipeline import ResearchPipeline
from core.stream_bus import StreamBus


def _pipe() -> ResearchPipeline:
    """构造 ResearchPipeline 并手动设好 _rt/_layers（直测 _rephrase，跳过 run() 的 profile 解析）。"""
    p = ResearchPipeline()
    p._rt = ProfileRuntime()
    p._layers = CommonContextLayers()
    return p


def _ctx(with_waiter: bool) -> UnifiedContext:
    ctx = UnifiedContext(
        course_id="c1",
        user_message="研究X",
        enabled_tools=["rag", "web_search"],
        mode="research",
        session_id="s1",
    )
    if with_waiter:

        async def _waiter():  # pragma: no cover - gate 测试不真等回复
            return {"text": "", "answers": None}

        ctx.metadata["wait_for_user_reply"] = _waiter
    return ctx


async def _capture_rephrase_tools(ctx, settings_research=None) -> list[str]:
    """跑一次 _rephrase，返回 run_agent_loop 收到的 enabled_tools（即 rephrase 实际挂载的工具）。"""
    pipe = _pipe()
    captured: dict = {}

    async def _fake_loop(**kw):
        captured["tools"] = list(kw["context"].enabled_tools)
        return LoopOutcome(final_text="精炼：X", rounds=1, tools_used=[], completed=True)

    patches = [
        patch("core.research.pipeline.run_agent_loop", new=_fake_loop),
        patch("core.research.pipeline.get_tool_schemas", return_value=[]),
        patch("core.research.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
    ]
    if settings_research is not None:
        patches.append(
            patch("core.research.pipeline.get_settings", new=lambda: type("S", (), {"research": settings_research}))
        )
    for p in patches:
        p.start()
    try:
        await pipe._rephrase(topic="X", context=ctx, stream=StreamBus(), cfg={"rephrase": {"system": ""}})
    finally:
        for p in patches:
            p.stop()
    return captured.get("tools", [])


@pytest.mark.asyncio
async def test_rephrase_mounts_ask_user_when_waiter_present():
    """WS 入口（有 waiter）+ 默认开关开 → rephrase 工具含 ask_user。"""
    tools = await _capture_rephrase_tools(_ctx(with_waiter=True))
    assert "ask_user" in tools
    assert "rag" in tools and "web_search" in tools


@pytest.mark.asyncio
async def test_rephrase_omits_ask_user_without_waiter():
    """HTTP/IM 入口无 waiter → 不挂 ask_user（防 loop 直接结束）；rag/web_search 仍挂。"""
    tools = await _capture_rephrase_tools(_ctx(with_waiter=False))
    assert "ask_user" not in tools
    assert "rag" in tools and "web_search" in tools


@pytest.mark.asyncio
async def test_rephrase_omits_ask_user_when_clarify_disabled():
    """clarify_enabled=False（总开关关）→ 即使有 waiter 也不挂 ask_user。"""
    disabled = type(
        "R", (), {"clarify_enabled": False, "clarify_wait_timeout_s": 120, "clarify_max_questions": 3}
    )()
    tools = await _capture_rephrase_tools(_ctx(with_waiter=True), settings_research=disabled)
    assert "ask_user" not in tools


@pytest.mark.asyncio
async def test_wait_for_user_reply_timeout_returns_skip(monkeypatch):
    """ask_user 等待超时 → 返回 skip dict（非 None、不挂死），让 loop 走 "User skipped." 续跑。"""
    from core.stream import StreamEvent, StreamEventType
    from services.session.turn_runtime import TurnRuntimeManager

    # 强制 turn_runtime.get_settings() 返回短超时 stub（不依赖 settings 单例缓存是否命中）
    _stub = type(
        "S",
        (),
        {
            "research": type(
                "R",
                (),
                {"clarify_enabled": True, "clarify_wait_timeout_s": 1, "clarify_max_questions": 3},
            )()
        },
    )()
    monkeypatch.setattr("services.session.turn_runtime.get_settings", lambda: _stub)

    class _FakeOrch:
        """伪 orchestrator：模拟 loop 的 ask_user 暂停——直接 await waiter（超时拿 skip dict）。"""

        def __init__(self) -> None:
            self.reply: object = "NOT_CALLED"

        async def handle(self, ctx):
            waiter = ctx.metadata.get("wait_for_user_reply")
            if callable(waiter):
                self.reply = await waiter()
            yield StreamEvent(type=StreamEventType.ANSWER, source="fake", payload={"content": "ok"})

    fake = _FakeOrch()
    monkeypatch.setattr("core.orchestrator.get_orchestrator", lambda: fake)

    mgr = TurnRuntimeManager()
    ctx = UnifiedContext(
        course_id="c1",
        user_id="u1",
        user_message="hi",
        mode="chat",
        session_id="s1",
        conversation_history=[],
    )
    turn_id = await mgr.start_turn(ctx)
    # 不提交回复 → waiter 应在 1s 超时返回 skip dict，turn 正常结束（不挂死）
    async for _ev in mgr.subscribe_turn(turn_id):
        pass
    assert fake.reply == {"text": "", "answers": None}
