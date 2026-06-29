"""core/bot/agent/loop.py 单元测试。

验证 bot 走主编排链路（TRM + Orchestrator）后的胶水逻辑：
  1. process_direct() 正常收集 ANSWER 事件返回文本
  2. 会话历史传入 UnifiedContext.conversation_history
  3. 有 course_id → enabled_tools 含 rag
  4. 无 course_id → enabled_tools 不含 rag
  5. LLM 返回空 → 不存入 session
  6. persona 注入 ctx.metadata["bot_persona"]

不依赖真实 API key / DB / orchestrator —— 用 _FakeTRM 替换 TurnRuntimeManager。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.bot.agent.loop import AgentLoop
from core.bot.bus.queue import MessageBus
from core.stream import StreamEvent, StreamEventType


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _answer_event(content: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.ANSWER, payload={"content": content})


class _FakeTRM:
    """替换 TurnRuntimeManager，捕获 ctx 并产出 ANSWER 事件流。"""

    def __init__(self, answer_text: str = "测试回答") -> None:
        self.answer_text = answer_text
        self.last_ctx: Any = None
        self.turn_id = "fake-turn-1"

    async def start_turn(self, ctx: Any) -> str:
        self.last_ctx = ctx
        return self.turn_id

    async def subscribe_turn(self, turn_id: str, after_seq: int = 0):
        if self.answer_text:
            yield _answer_event(self.answer_text)

    async def cancel_turn(self, turn_id: str) -> None:
        pass


def _make_loop(tmp_path: Path, course_id: str = "", persona: str = "") -> AgentLoop:
    return AgentLoop(
        bus=MessageBus(),
        workspace=tmp_path,
        course_id=course_id,
        persona=persona,
        max_iterations=5,
    )


def _patch_trm(fake: _FakeTRM):
    return patch(
        "services.session.turn_runtime.get_turn_runtime_manager",
        return_value=fake,
    )


# ---------------------------------------------------------------------------
# 场景 1：process_direct 正常收集 ANSWER 返回文本
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_direct_returns_text(tmp_path):
    loop = _make_loop(tmp_path)
    fake = _FakeTRM("你好！")
    with _patch_trm(fake):
        result = await loop.process_direct("你好", session_key="s1")
    assert result == "你好！"


# ---------------------------------------------------------------------------
# 场景 2：会话历史传入 UnifiedContext.conversation_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_passed_to_context(tmp_path):
    loop = _make_loop(tmp_path)
    fake = _FakeTRM("第二轮回答")
    with _patch_trm(fake):
        await loop.process_direct("第一条消息", session_key="s_hist")
        await loop.process_direct("第二条消息", session_key="s_hist")
    ctx = fake.last_ctx
    roles = [m["role"] for m in ctx.conversation_history]
    assert "user" in roles
    assert "assistant" in roles
    # 当前消息不在 history 里（它是 user_message）
    assert ctx.user_message == "第二条消息"


# ---------------------------------------------------------------------------
# 场景 3：有 course_id → enabled_tools 含 rag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_with_course_id_enables_rag(tmp_path):
    loop = _make_loop(tmp_path, course_id="cs101")
    fake = _FakeTRM()
    with _patch_trm(fake):
        await loop.process_direct("问题", session_key="s_rag")
    assert "rag" in fake.last_ctx.enabled_tools
    assert fake.last_ctx.course_id == "cs101"


# ---------------------------------------------------------------------------
# 场景 4：无 course_id → enabled_tools 不含 rag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_without_course_id_no_rag(tmp_path):
    loop = _make_loop(tmp_path, course_id="")
    fake = _FakeTRM()
    with _patch_trm(fake):
        await loop.process_direct("问题", session_key="s_no_rag")
    assert "rag" not in fake.last_ctx.enabled_tools
    assert "web_search" in fake.last_ctx.enabled_tools


# ---------------------------------------------------------------------------
# 场景 5：LLM 返回空字符串 → 不存入 session（session 只有 user 消息）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_reply_not_saved(tmp_path):
    loop = _make_loop(tmp_path)
    fake = _FakeTRM("")
    with _patch_trm(fake):
        result = await loop.process_direct("问题", session_key="s_empty")
    assert result == ""
    session = loop.sessions.get_or_create("s_empty")
    roles = [m["role"] for m in session.messages]
    assert "assistant" not in roles


# ---------------------------------------------------------------------------
# 场景 6：persona 注入 ctx.metadata["bot_persona"]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persona_injected_to_metadata(tmp_path):
    loop = _make_loop(tmp_path, persona="你是数学辅导老师。")
    fake = _FakeTRM()
    with _patch_trm(fake):
        await loop.process_direct("问题", session_key="s_persona")
    assert fake.last_ctx.metadata.get("bot_persona") == "你是数学辅导老师。"
