"""Async session and message storage with user isolation (PostgreSQL)."""
from __future__ import annotations

import json
import logging
import time
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import Message, Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session CRUD (async, with user_id isolation)
# ---------------------------------------------------------------------------

async def create_session(
    db: AsyncSession,
    course_id: str,
    title: str = "新对话",
    user_id: str = "",
    mode: str = "chat",
) -> dict:
    now = time.time()
    session = Session(
        id=uuid.uuid4().hex[:12],
        course_id=course_id,
        user_id=user_id,
        title=title,
        mode=mode,
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    await db.flush()
    return {
        "id": session.id,
        "course_id": session.course_id,
        "user_id": session.user_id,
        "title": session.title,
        "mode": session.mode,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


async def list_sessions(
    db: AsyncSession, course_id: str | None = None, user_id: str = "",
) -> list[dict]:
    stmt = select(Session).where(Session.user_id == user_id)
    if course_id:
        stmt = stmt.where(Session.course_id == course_id)
    stmt = stmt.order_by(Session.updated_at.desc())
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "course_id": r.course_id,
            "user_id": r.user_id,
            "title": r.title,
            "mode": r.mode,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
        }
        for r in rows
    ]


async def get_session(db: AsyncSession, session_id: str) -> dict | None:
    result = await db.execute(select(Session).where(Session.id == session_id))
    r = result.scalar_one_or_none()
    if not r:
        return None
    return {
        "id": r.id,
        "course_id": r.course_id,
        "user_id": r.user_id,
        "title": r.title,
        "mode": r.mode,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


async def update_session_title(db: AsyncSession, session_id: str, title: str):
    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(title=title, updated_at=time.time())
    )


async def update_session_mode(db: AsyncSession, session_id: str, mode: str):
    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(mode=mode, updated_at=time.time())
    )


async def delete_session(db: AsyncSession, session_id: str):
    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.execute(delete(Session).where(Session.id == session_id))


# ---------------------------------------------------------------------------
# Message CRUD (async)
# ---------------------------------------------------------------------------

def _row_dict(r: Message, metadata: dict | None = None) -> dict:
    """Message 行 -> API dict（幂等命中/插入返回/get_messages 三处共用）。"""
    return {
        "id": r.id,
        "session_id": r.session_id,
        "role": r.role,
        "content": r.content,
        "msg_type": r.msg_type,
        "metadata": metadata if metadata is not None else json.loads(r.metadata_ or "{}"),
        "created_at": r.created_at,
        "turn_id": r.turn_id,
    }


async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    msg_type: str = "text",
    metadata: dict | None = None,
    turn_id: str | None = None,
    created_at: float | None = None,
    course_id: str | None = None,
    touch_session: bool = True,
) -> dict:
    """写入一条消息行。

    turn_id 非空时按 (session_id, turn_id, role) 幂等：已有同键行直接返回它
    （服务端回合日志重复投递场景），不重复插入。uq_messages_session_turn_role
    唯一索引是 DB 层兜底。

    created_at 可显式传入（服务端回合日志用：同回合 user/assistant 两行传严格
    递增戳，保证 keyset/展示排序 (created_at, id) 下 user 在前——随机 id 做不了
    同回合内的确定性 tiebreak）。

    course_id / touch_session 供批量写者省往返：调用方已持有 Session 行时传
    course_id 免去逐行反查；一行会话多消息时只让最后一行 touch（updated_at
    每回合动一次即可）。
    """
    now = time.time() if created_at is None else created_at
    if turn_id:
        existing = (await db.execute(
            select(Message).where(
                Message.session_id == session_id,
                Message.turn_id == turn_id,
                Message.role == role,
            )
        )).scalars().first()
        if existing is not None:
            return _row_dict(existing)
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    # P1：写时落盘 course_id（反查 Session，省去读侧 JOIN；宪法原则 3）
    if course_id is None:
        course_id = (await db.execute(
            select(Session.course_id).where(Session.id == session_id)
        )).scalar_one_or_none() or ""
    msg = Message(
        id=uuid.uuid4().hex[:16],
        session_id=session_id,
        course_id=course_id,
        turn_id=turn_id,
        role=role,
        content=content,
        msg_type=msg_type,
        metadata_=meta_json,
        created_at=now,
    )
    db.add(msg)
    if touch_session:
        await db.execute(
            update(Session).where(Session.id == session_id).values(updated_at=now)
        )
    await db.flush()
    return _row_dict(msg, metadata or {})


async def get_messages(db: AsyncSession, session_id: str) -> list[dict]:
    # 排序带 id tiebreak：created_at 相同（同回合两行/时钟分辨率）时随机 id 会乱序
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
    )
    return [_row_dict(r) for r in result.scalars().all()]
