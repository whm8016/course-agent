"""turn_runtime 收集 RESULT 事件产物 → CapabilityCompleteEvent 也对 quiz/research 触发。

根因：旧实现只收集 ANSWER 事件；quiz / deep_research 的 pipeline 设
emit_terminal_events=False、以 result 事件收尾（quiz: questions，research: report），
这两类能力从不发 ANSWER → CapabilityCompleteEvent 从不发布 → L3 outbox
（record_episode）、巩固触发（add_importance）、L2 摘要压缩对它们全不跑，
学生自测、调研过程进不了学情。

修法：RESULT 事件抽出可写文本（report 字符串 / questions 列表转 JSON），
agent_output 为空时以 result 文本替代发布事件；metadata 一并透传。
"""
from __future__ import annotations

import asyncio

from core.context import UnifiedContext
from core.stream import StreamEvent, StreamEventType
from events.event_bus import get_event_bus, CapabilityCompleteEvent
from services.session.turn_runtime import TurnRuntimeManager


class _ResultOrch:
    """伪 orchestrator：按预设事件列表吐流（模拟 quiz/research 收尾的 result 事件）。"""

    def __init__(self, events: list[StreamEvent]) -> None:
        self._events = events

    async def handle(self, ctx):
        for ev in self._events:
            yield ev


async def _run_and_capture(events: list[StreamEvent], mode: str = "quiz"):
    """跑一个 turn（伪 orchestrator），捕获发布的 CapabilityCompleteEvent（无则 None）。"""
    received: list[CapabilityCompleteEvent] = []

    async def _handler(ev: CapabilityCompleteEvent) -> None:
        received.append(ev)

    bus = get_event_bus()
    from events.event_bus import EventType
    bus.subscribe(EventType.CAPABILITY_COMPLETE, _handler)

    from unittest.mock import patch

    mgr = TurnRuntimeManager()
    ctx = UnifiedContext(
        course_id="c1",
        user_id="u1",
        user_message="出 3 道导数题",
        mode=mode,
        session_id="sess-1",
        conversation_history=[],
    )
    with patch(
        "core.orchestrator.get_orchestrator", return_value=_ResultOrch(events)
    ), patch(
        "services.session.turn_runtime.get_settings",
        return_value=type(
            "S",
            (),
            {
                "research": type(
                    "R", (), {"clarify_enabled": False, "clarify_wait_timeout_s": 0, "clarify_max_questions": 3}
                )(),
                "context_budget": type("CB", (), {"coordinator_enabled": False})(),
            },
        )(),
    ):
        turn_id = await mgr.start_turn(ctx)
        async for _ev in mgr.subscribe_turn(turn_id):
            pass
    await asyncio.sleep(0.05)  # handler 同步 await publish，事件应已入列
    return received


async def test_quiz_result_triggers_complete_event():
    """quiz：无 ANSWER、只有 result(questions) → 也发 CapabilityCompleteEvent，题目进 agent_output。"""
    events = [
        StreamEvent(
            type=StreamEventType.RESULT,
            payload={
                "questions": [{"question_id": "q_1", "topic": "导数", "stem": "求 f(x)=x² 的导数"}],
                "metadata": {"tools_used": ("rag",), "count_generated": 1},
            },
        ),
        StreamEvent(type=StreamEventType.DONE, payload={}),
    ]
    received = await _run_and_capture(events, mode="quiz")

    assert len(received) == 1
    ev = received[0]
    assert ev.mode == "quiz"
    assert "求 f(x)=x² 的导数" in ev.agent_output  # 题目 JSON 进 agent_output
    assert ev.metadata.get("tools_used") == ("rag",)  # result metadata 透传


async def test_research_result_triggers_complete_event():
    """deep_research：result(report) → 报告文本进 agent_output。"""
    events = [
        StreamEvent(
            type=StreamEventType.RESULT,
            payload={
                "research_id": "r1",
                "report": "# 导数与积分的关系\n（报告正文）",
                "metadata": {},
            },
        ),
    ]
    received = await _run_and_capture(events, mode="deep_research")

    assert len(received) == 1
    assert "报告正文" in received[0].agent_output


async def test_answer_still_wins_over_result():
    """chat 正常路径：有 ANSWER 时 agent_output 仍是 ANSWER 文本（result 不覆盖）。"""
    events = [
        StreamEvent(type=StreamEventType.RESULT, payload={"report": "报告"}),
        StreamEvent(type=StreamEventType.ANSWER, payload={"content": "KVL 是基尔霍夫电压定律"}),
    ]
    received = await _run_and_capture(events, mode="chat")

    assert len(received) == 1
    assert received[0].agent_output == "KVL 是基尔霍夫电压定律"


async def test_no_output_no_event():
    """既无 ANSWER 也无可写 result → 不发事件（原行为）。"""
    events = [
        StreamEvent(type=StreamEventType.RESULT, payload={"metadata": {"x": 1}}),
        StreamEvent(type=StreamEventType.DONE, payload={}),
    ]
    received = await _run_and_capture(events, mode="quiz")

    assert received == []
