"""学情事件层（L0）测试：record_learning_event 写入 + turn 完成 wired 'asked' 事件。

学情分析四模块设计 §第二期 p2-events-table：learning_events 表承接 asked/answered/feedback
三类信号。本文件验写入链路；'answered' 随 p2-b rollup、'feedback' 随 Phase 4 各自落地。
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_record_learning_event_persists(client):
    """record_learning_event 落一条事件，字段（含 metadata）正确持久化。"""
    from core.analytics.events import record_learning_event
    from core.db.database import AsyncSessionLocal, LearningEvent

    await record_learning_event(
        user_id="u_le1", course_id="c_le1", verb="answered",
        object_id="q1", object_text="2+2=?", session_id="s1",
        metadata={"is_correct": True, "difficulty": "easy"},
    )

    async with AsyncSessionLocal() as db:
        ev = (await db.execute(
            select(LearningEvent).where(LearningEvent.actor_user_id == "u_le1")
        )).scalar_one()
    assert ev.verb == "answered"
    assert ev.course_id == "c_le1"
    assert ev.object_id == "q1"
    assert ev.object_text == "2+2=?"
    assert ev.session_id == "s1"
    assert ev.metadata_ == {"is_correct": True, "difficulty": "easy"}


@pytest.mark.asyncio
async def test_capability_complete_records_asked_event(client):
    """CAPABILITY_COMPLETE 的学情事件订阅者落一条 verb=asked 事件，含 mode/tools_used 上下文。"""
    import main as main_mod
    from core.db.database import AsyncSessionLocal, LearningEvent

    event = SimpleNamespace(
        user_id="u_asked", course_id="c_asked", session_id="sess_asked",
        user_message="什么是戴维南定理？", mode="chat",
        metadata={"tools_used": ("calc",)},
    )
    await main_mod._on_complete_record_learning_event(event)

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(LearningEvent).where(LearningEvent.actor_user_id == "u_asked")
        )).scalars().all()
    assert len(rows) == 1
    ev = rows[0]
    assert ev.verb == "asked"
    assert ev.course_id == "c_asked"
    assert "戴维南" in ev.object_text
    assert ev.session_id == "sess_asked"
    assert ev.metadata_["mode"] == "chat"
    assert ev.metadata_["tools_used"] == ["calc"]


@pytest.mark.asyncio
async def test_add_message_populates_course_id(client):
    """P1：add_message 写时落盘 course_id（反查 Session），后续课程级查询免 JOIN。"""
    import os

    from core.db.database import AsyncSessionLocal, Message, Session, User
    from core.memory.memory import add_message

    async with AsyncSessionLocal() as db:
        u = User(username=f"msg_{os.urandom(3).hex()}", password_hash="x")
        db.add(u)
        await db.flush()
        db.add(Session(id="sess_msg", course_id="c_msg", user_id=u.id))
        await db.commit()

    async with AsyncSessionLocal() as db:
        await add_message(db, "sess_msg", "user", "hello")
        await db.commit()

    async with AsyncSessionLocal() as db:
        m = (await db.execute(
            select(Message).where(Message.session_id == "sess_msg")
        )).scalar_one()
    assert m.course_id == "c_msg"


@pytest.mark.asyncio
async def test_notebook_answer_writes_answered_event(client):
    """学生提交作答（user_answer 非空）→ POST /question/notebook/entries/upsert 落一条
    verb=answered 事件：course_id 经 Session 反查，metadata 带 is_correct/difficulty。

    P0-a：补全事件层 answered 信号（此前仅 asked 被 wired）。
    """
    from core.db.database import AsyncSessionLocal, LearningEvent, Session, User

    username = f"ans_{os.urandom(3).hex()}"
    r = await client.post("/api/auth/register", json={
        "username": username, "password": "testpass123", "display_name": "A",
    })
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['token']}"}

    async with AsyncSessionLocal() as db:
        u = (await db.execute(select(User).where(User.username == username))).scalar_one()
        uid = u.id
        db.add(Session(id="sess_ans", course_id="c_ans", user_id=uid))
        await db.commit()

    r = await client.post("/api/question/notebook/entries/upsert", headers=headers, json={
        "session_id": "sess_ans", "question_id": "q1", "question": "1+1=?",
        "user_answer": "2", "is_correct": True, "difficulty": "easy",
    })
    assert r.status_code == 200, r.text

    async with AsyncSessionLocal() as db:
        evs = (await db.execute(
            select(LearningEvent).where(LearningEvent.actor_user_id == uid)
        )).scalars().all()
    assert len(evs) == 1
    ev = evs[0]
    assert ev.verb == "answered"
    assert ev.course_id == "c_ans"            # 经 Session 反查得到
    assert ev.object_id == "q1"
    assert ev.session_id == "sess_ans"
    assert ev.metadata_["is_correct"] is True
    assert ev.metadata_["difficulty"] == "easy"


@pytest.mark.asyncio
async def test_notebook_save_without_answer_writes_no_event(client):
    """仅保存题目（user_answer 为空，如出题后未作答）不落 answered 事件，避免污染统计。"""
    from core.db.database import AsyncSessionLocal, LearningEvent, Session, User

    username = f"noans_{os.urandom(3).hex()}"
    r = await client.post("/api/auth/register", json={
        "username": username, "password": "testpass123", "display_name": "B",
    })
    assert r.status_code == 200, r.text
    headers = {"Authorization": f"Bearer {r.json()['token']}"}

    async with AsyncSessionLocal() as db:
        u = (await db.execute(select(User).where(User.username == username))).scalar_one()
        db.add(Session(id="sess_noans", course_id="c_noans", user_id=u.id))
        await db.commit()

    # user_answer 缺省 "" → 不落 answered 事件
    r = await client.post("/api/question/notebook/entries/upsert", headers=headers, json={
        "session_id": "sess_noans", "question_id": "q1", "question": "1+1=?",
    })
    assert r.status_code == 200, r.text

    async with AsyncSessionLocal() as db:
        evs = (await db.execute(
            select(LearningEvent).where(LearningEvent.session_id == "sess_noans")
        )).scalars().all()
    assert evs == []
