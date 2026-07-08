"""H-15：AgentLoop 处理 TRM ERROR 事件，避免 IM 用户沉默。

根因：``AgentLoop._run_turn`` 在 ``subscribe_turn`` 循环里只累加 ANSWER 事件，忽略
ERROR 事件。当 LLM/orchestrator 异常时，TRM 发 ``StreamEventType.ERROR``（payload.message
是错误描述）但没有 ANSWER → ``final_text`` 为空 → ``_process_message`` 返回 None →
IM 用户（QQ/飞书/web 直发）收到的是**无回复的沉默**。

修法：循环里捕获 ERROR 事件记下错误；循环结束后若无有效答案文本且有 ERROR，用友好兜底
文案回复（不向 IM 用户暴露内部异常细节），保证「用户总能收到反馈」。

时序：
  - 纯 ERROR（无 ANSWER）→ 兜底文案。✓ 不再沉默。
  - 有 ANSWER 后 ERROR（尾声告警）→ 用 ANSWER 文本（答案仍可用）。✓ 不丢答案。
  - 正常 ANSWER → 原行为不变。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bot.agent.loop import AgentLoop  # noqa: E402
from core.bot.bus.queue import MessageBus  # noqa: E402
from core.bot.session.manager import SessionManager  # noqa: E402
from core.stream import StreamEvent, StreamEventType  # noqa: E402


class _FakeTRM:
    """最小 TRM：subscribe_turn 按预设事件列表回放。"""

    def __init__(self, events: list[StreamEvent]):
        self._events = events
        self.start_turn_call_count = 0
        self.cancel_turn_calls: list[str] = []

    async def start_turn(self, ctx):
        self.start_turn_call_count += 1
        return "fake-turn-id"

    async def subscribe_turn(self, turn_id: str) -> AsyncIterator[StreamEvent]:
        for ev in self._events:
            yield ev

    async def cancel_turn(self, turn_id: str):
        self.cancel_turn_calls.append(turn_id)


def _make_loop(tmp_path: Path) -> AgentLoop:
    bus = MessageBus()
    sm = SessionManager(tmp_path)
    return AgentLoop(bus=bus, workspace=tmp_path, session_manager=sm, default_session_key="s1")


async def test_error_event_no_silence(tmp_path):
    """纯 ERROR（无 ANSWER）→ 返回兜底文案，IM 用户不再沉默。"""
    loop = _make_loop(tmp_path)
    fake = _FakeTRM([
        StreamEvent(type=StreamEventType.ERROR, payload={"message": "LLM timeout"}),
    ])
    with patch(
        "services.session.turn_runtime.get_turn_runtime_manager", return_value=fake
    ):
        text = await loop._run_turn("hi", session_key="s1")

    assert text  # 非空（旧实现会返回空 → 沉默）
    assert "暂时无法处理" in text
    # cancel_turn 不该被调（不是 CancelledError 路径）
    assert fake.cancel_turn_calls == []


async def test_answer_present_overrides_error(tmp_path):
    """有 ANSWER 后又来 ERROR（尾声告警）→ 用 ANSWER 文本，不丢答案。"""
    loop = _make_loop(tmp_path)
    fake = _FakeTRM([
        StreamEvent(type=StreamEventType.ANSWER, payload={"content": "答案是42"}),
        StreamEvent(type=StreamEventType.ERROR, payload={"message": "tail warn"}),
    ])
    with patch(
        "services.session.turn_runtime.get_turn_runtime_manager", return_value=fake
    ):
        text = await loop._run_turn("问题", session_key="s1")

    assert text == "答案是42"


async def test_normal_answer_unchanged(tmp_path):
    """正常 ANSWER 路径行为不变（无 ERROR，无兜底）。"""
    loop = _make_loop(tmp_path)
    fake = _FakeTRM([
        StreamEvent(type=StreamEventType.ANSWER, payload={"content": "你好"}),
    ])
    with patch(
        "services.session.turn_runtime.get_turn_runtime_manager", return_value=fake
    ):
        text = await loop._run_turn("嗨", session_key="s1")

    assert text == "你好"
