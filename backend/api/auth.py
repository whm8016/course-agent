"""Auth endpoints: register, login, profile."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Header, Request, WebSocket
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.auth import (
    authenticate_user,
    create_token,
    create_user,
    decode_token,
    get_user_by_id,
)
from core.db.database import AsyncSessionLocal, get_db, TeacherInvite
from core.db.limiter import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterBody(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=4, max_length=128)
    display_name: str = ""
    invite_code: str | None = None


class LoginBody(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Dependency: extract current user from JWT
# ---------------------------------------------------------------------------

async def get_current_user(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    user = await get_user_by_id(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


async def get_current_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    return user


async def get_current_teacher(user: dict = Depends(get_current_user)) -> dict:
    """Allow teacher and admin roles."""
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="仅教师或管理员可访问")
    return user


async def ws_authenticate(websocket: WebSocket) -> dict | None:
    """Authenticate WebSocket via query param ?token=xxx. Returns user dict or None (closes socket)."""
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4001, reason="缺少 token")
        return None
    payload = decode_token(token)
    if not payload:
        await websocket.close(code=4001, reason="token 无效或已过期")
        return None
    async with AsyncSessionLocal() as db:
        user = await get_user_by_id(db, payload["sub"])
    if not user:
        await websocket.close(code=4001, reason="用户不存在")
        return None
    await websocket.accept()
    return user


async def get_optional_user(
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> dict | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    payload = decode_token(token)
    if not payload:
        return None
    return await get_user_by_id(db, payload["sub"])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register")
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterBody, db: AsyncSession = Depends(get_db)):
    role = "student"
    invite_row = None

    if body.invite_code:
        result = await db.execute(
            select(TeacherInvite).where(TeacherInvite.code == body.invite_code)
        )
        invite_row = result.scalar_one_or_none()
        if not invite_row:
            raise HTTPException(status_code=400, detail="邀请码无效")
        if invite_row.used_by is not None:
            raise HTTPException(status_code=400, detail="邀请码已被使用")
        if invite_row.expires_at and invite_row.expires_at < time.time():
            raise HTTPException(status_code=400, detail="邀请码已过期")
        role = "teacher"

    try:
        user = await create_user(db, body.username, body.password, body.display_name, role=role)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if invite_row:
        await db.execute(
            update(TeacherInvite)
            .where(TeacherInvite.id == invite_row.id)
            .values(used_by=user["id"])
        )

    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}


@router.post("/login")
@limiter.limit("15/minute")
async def login(
    request: Request,
    body: LoginBody,
    db: AsyncSession = Depends(get_db),
):
    user = await authenticate_user(db, body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}

@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"user": user}
