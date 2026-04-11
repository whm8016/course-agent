"""Async session and message storage with user isolation (PostgreSQL)."""
from __future__ import annotations

import json
import logging
import time
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import Message, Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session CRUD (async, with user_id isolation)
# ---------------------------------------------------------------------------

async def create_session(
    db: AsyncSession, course_id: str, title: str = "新对话", user_id: str = "",
) -> dict:
    now = time.time()
    session = Session(
        id=uuid.uuid4().hex[:12],
        course_id=course_id,
        user_id=user_id,
        title=title,
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
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


async def update_session_title(db: AsyncSession, session_id: str, title: str):
    await db.execute(
        update(Session)
        .where(Session.id == session_id)
        .values(title=title, updated_at=time.time())
    )


async def delete_session(db: AsyncSession, session_id: str):
    await db.execute(delete(Message).where(Message.session_id == session_id))
    await db.execute(delete(Session).where(Session.id == session_id))


# ---------------------------------------------------------------------------
# Message CRUD (async)
# ---------------------------------------------------------------------------

async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    msg_type: str = "text",
    metadata: dict | None = None,
) -> dict:
    now = time.time()
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    msg = Message(
        id=uuid.uuid4().hex[:16],
        session_id=session_id,
        role=role,
        content=content,
        msg_type=msg_type,
        metadata_=meta_json,
        created_at=now,
    )
    db.add(msg)
    await db.execute(
        update(Session).where(Session.id == session_id).values(updated_at=now)
    )
    await db.flush()
    return {
        "id": msg.id,
        "session_id": msg.session_id,
        "role": msg.role,
        "content": msg.content,
        "msg_type": msg.msg_type,
        "metadata": metadata or {},
        "created_at": msg.created_at,
    }


async def get_messages(db: AsyncSession, session_id: str) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    rows = result.scalars().all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "session_id": r.session_id,
            "role": r.role,
            "content": r.content,
            "msg_type": r.msg_type,
            "metadata": json.loads(r.metadata_ or "{}"),
            "created_at": r.created_at,
        })
    return out
