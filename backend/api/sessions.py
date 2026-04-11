from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.database import get_db
from core.memory import (
    add_message,
    create_session,
    delete_session,
    get_messages,
    get_session,
    list_sessions,
    update_session_title,
)

router = APIRouter()


class CreateSessionBody(BaseModel):
    course_id: str
    title: str = "新对话"


class UpdateSessionBody(BaseModel):
    title: str


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


@router.get("/sessions")
async def api_list_sessions(
    course_id: str | None = None,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return {"sessions": await list_sessions(db, course_id, user_id=user["id"])}


@router.post("/sessions")
async def api_create_session(
    body: CreateSessionBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await create_session(db, body.course_id, body.title, user_id=user["id"])
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
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def api_delete_session(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await _check_session_owner(db, session_id, user["id"])
    await delete_session(db, session_id)
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
