"""错误信封边界测试（可观测性重整 Step 4/5）。

断点 2/3 的复现：能力异常经 orchestrator/TrackedCapability 裸透传 str(exc) 给学生；
agent loop 中途失败伪装成正常 answer 事件。修后契约：边界只发带信封的 error 事件
（分类 code + 中文文案 + ref），内部异常文本不进用户可见 message。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.context import UnifiedContext
from core.orchestrator import CourseOrchestrator
from core.stream import StreamEvent, StreamEventType
from core.stream_bus import StreamBus
from core.capability_protocol import TrackedCapability


def _ctx(mode: str = "chat") -> UnifiedContext:
    return UnifiedContext(
        course_id="c1", user_id="u1", session_id="", user_message="hi", mode=mode,
        conversation_history=[],
    )


# ---------------------------------------------------------------------------
# Step 4a：TrackedCapability 兜底 → 信封（非裸 str(exc)）
# ---------------------------------------------------------------------------


class _BoomCapability(TrackedCapability):
    """run_with_tracking 抛内部异常的最小能力（manifest 用鸭子类型顶替）。"""

    manifest = SimpleNamespace(name="deep_solve")

    async def run_with_tracking(self, context, stream, *, t0):
        raise ValueError("upstream secret exploded: Error code: 401")

    def error_label(self) -> str:
        return "deep solve 失败"


@pytest.mark.asyncio
async def test_tracked_capability_error_uses_envelope():
    from core.observability.context import bind_context

    bind_context(turn_id="turnabcdefgh1234")  # ref 来源：turn_id 前 8 位
    bus = StreamBus()
    cap = _BoomCapability()
    await cap.run(_ctx("deep_solve"), bus)
    events = [e for e in bus._history if e.type == StreamEventType.ERROR]
    assert len(events) == 1
    payload = events[0].payload
    # 契约：分类 code + 中文文案（能力名前缀保留）；内部异常文本不进 message
    assert payload["code"] == "internal"
    assert payload["message"].startswith("deep solve 失败：")
    assert "upstream secret exploded" not in payload["message"]
    assert payload["exc_type"] == "ValueError"
    assert "upstream secret exploded" in payload["exc_head"]
    assert payload["ref"] == "turnabcd"


@pytest.mark.asyncio
async def test_tracked_capability_llm_error_classified():
    """LLM 重试耗尽 → code=llm_unavailable（学生看到的是固定中文，非英文内部信息）。"""
    from core.llm.reliability import LLMRetryError

    class _LLMBoomCapability(_BoomCapability):
        async def run_with_tracking(self, context, stream, *, t0):
            raise LLMRetryError("LLM call failed after 4 attempts: Error code: 401")

    bus = StreamBus()
    await _LLMBoomCapability().run(_ctx("deep_solve"), bus)
    events = [e for e in bus._history if e.type == StreamEventType.ERROR]
    assert events[0].payload["code"] == "llm_unavailable"
    assert "Error code: 401" not in events[0].payload["message"]
    assert events[0].payload["exc_type"] == "LLMRetryError"


# ---------------------------------------------------------------------------
# Step 4b：orchestrator._run 兜底 → 信封
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_capability_raise_uses_envelope(monkeypatch):
    """capability.run 抛异常（chat 裸委托路径）→ error 事件带信封，message 非裸 str。"""
    from core.orchestrator import unregister_bus
    from core.registry import get_capability_registry

    class _RaisingCap:
        name = "chat"

        async def run(self, context, bus):
            raise RuntimeError("Connection error: internal endpoint leak")

    registry = get_capability_registry()
    monkeypatch.setitem(registry._capabilities, "chat", _RaisingCap())
    orch = CourseOrchestrator()
    events: list[StreamEvent] = []
    try:
        async for ev in orch.handle(_ctx("chat")):
            events.append(ev)
    finally:
        unregister_bus("")  # handle 内部按空 turn_id 注册时兜底清理
    errs = [e for e in events if e.type == StreamEventType.ERROR]
    assert len(errs) == 1
    payload = errs[0].payload
    assert payload["code"] == "llm_unavailable"  # "connection error" → 供应商侧
    assert "internal endpoint leak" not in payload["message"]
    assert payload["message"]  # 非空中文文案
    assert payload["exc_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Step 5：agent loop 中途失败 → 真 error 事件（不再伪装正常 answer）
# ---------------------------------------------------------------------------


def _tool_round():
    from core.agentic.types import RoundResult, ToolCall

    return RoundResult(
        content="我先查一下",
        tool_calls=[ToolCall(id="t1", name="rag", arguments={"query": "x"})],
    )


@pytest.mark.asyncio
async def test_mid_loop_failure_emits_error_not_fake_answer(monkeypatch):
    """iteration>0 的 LLM 失败：发分类 error 事件 + done(error_code)；不发伪装 answer。"""
    from core.agentic import loop as loop_mod
    from core.agentic.types import DispatchOutcome
    from core.observability.context import bind_context

    calls = {"n": 0}

    async def _fake_one_round(messages, schemas, model, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_round()
        raise RuntimeError("Connection error: upstream died mid-way")

    async def _fake_dispatch(tool_calls, *a, **kw):
        return DispatchOutcome(
            tool_messages=[{"role": "tool", "content": "结果", "tool_call_id": "t1"}],
            tools_used=["rag"],
        )

    monkeypatch.setattr(loop_mod, "_one_round", _fake_one_round)
    monkeypatch.setattr(loop_mod, "dispatch_tool_calls", _fake_dispatch)
    bind_context(turn_id="turnmidfail0001")

    bus = StreamBus()
    outcome = await loop_mod.run_agent_loop(
        context=_ctx("chat"), stream=bus, system_prompt="S",
        tool_schemas=[{"type": "function", "function": {"name": "rag", "parameters": {}}}],
    )
    types = [e.type for e in bus._history]
    # 契约：ERROR + DONE 在场，ANSWER 不在场（旧实现发固定文案的伪 answer）
    assert StreamEventType.ERROR in types
    assert StreamEventType.DONE in types
    assert StreamEventType.ANSWER not in types
    err = next(e for e in bus._history if e.type == StreamEventType.ERROR)
    assert err.payload["code"] == "llm_unavailable"
    assert err.payload["ref"] == "turnmidf"
    assert "upstream died" not in err.payload["message"]
    done = next(e for e in bus._history if e.type == StreamEventType.DONE)
    assert done.payload["metadata"].get("error_code") == "llm_unavailable"
    # outcome 文本优雅降级（非旧固定文案，非英文内部信息）
    assert outcome.final_text


@pytest.mark.asyncio
async def test_first_round_failure_still_raises(monkeypatch):
    """iteration==0 失败仍向上抛（TrackedCapability/orchestrator 信封兜底）——回归。"""
    from core.agentic import loop as loop_mod

    async def _always_raise(messages, schemas, model, *a, **kw):
        raise RuntimeError("Connection error: first round died")

    monkeypatch.setattr(loop_mod, "_one_round", _always_raise)
    with pytest.raises(RuntimeError, match="first round died"):
        await loop_mod.run_agent_loop(
            context=_ctx("chat"), stream=StreamBus(), system_prompt="S",
        )
