from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.cache import cache_delete, cache_get, cache_set
from core.database import get_db
from core.memory import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    update_session_mode,
    update_session_title,
)
from core.orchestrator import normalize_mode

router = APIRouter()

_SESSION_LIST_TTL = 30


class CreateSessionBody(BaseModel):
    course_id: str
    title: str = "新对话"
    mode: str = "chat"


class UpdateSessionBody(BaseModel):
    title: str


class UpdateSessionModeBody(BaseModel):
    mode: str


class AddMessageBody(BaseModel):
    role: str
    content: str
    msg_type: str = "text"
    metadata: dict | None = None


async def _check_session_owner(db: AsyncSession, session_id: str, user_id: str) -> dict:
    """Retrieve session and verify ownership."""
    session = await get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id", "") != user_id:
        raise HTTPException(status_code=403, detail="无权访问此会话")
    return session


def _sessions_cache_key(user_id: str, course_id: str | None) -> str:
    return f"sessions:{user_id}:{course_id or 'all'}"


@router.get("/sessions")
async def api_list_sessions(
    course_id: str | None = None,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ck = _sessions_cache_key(user["id"], course_id)
    cached = await cache_get(ck)
    if cached is not None:
        return cached
    result = {"sessions": await list_sessions(db, course_id, user_id=user["id"])}
    await cache_set(ck, result, ttl=_SESSION_LIST_TTL)
    return result


@router.post("/sessions")
async def api_create_session(
    body: CreateSessionBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await create_session(
        db,
        body.course_id,
        body.title,
        user_id=user["id"],
        mode=normalize_mode(body.mode),
    )
    from core.cache import cache_delete_pattern
    await cache_delete_pattern(f"sessions:{user['id']}:*")
    return session


@router.get("/sessions/{session_id}")
async def api_get_session(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _check_session_owner(db, session_id, user["id"])


@router.patch("/sessions/{session_id}")
async def api_update_session(
    session_id: str,
    body: UpdateSessionBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    await update_session_title(db, session_id, body.title)
    from core.cache import cache_delete_pattern
    await cache_delete_pattern(f"sessions:{user['id']}:*")
    return {"ok": True}


@router.patch("/sessions/{session_id}/mode")
async def api_update_session_mode(
    session_id: str,
    body: UpdateSessionModeBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    await update_session_mode(db, session_id, normalize_mode(body.mode))
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def api_delete_session(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    await delete_session(db, session_id)
    from core.cache import cache_delete_pattern
    await cache_delete_pattern(f"sessions:{user['id']}:*")
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def api_get_messages(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    return {"messages": await get_messages(db, session_id)}


@router.post("/sessions/{session_id}/messages")
async def api_add_message(
    session_id: str,
    body: AddMessageBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    msg = await add_message(db, session_id, body.role, body.content, body.msg_type, body.metadata)
    return msg
