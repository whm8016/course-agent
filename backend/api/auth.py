"""Auth endpoints: register, login, profile."""
from __future__ import annotations

import logging
from fastapi import Request
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import (
    authenticate_user,
    create_token,
    create_user,
    decode_token,
    get_user_by_id,
)
from core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterBody(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=4, max_length=128)
    display_name: str = ""


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
async def register(body: RegisterBody, db: AsyncSession = Depends(get_db)):
    try:
        user = await create_user(db, body.username, body.password, body.display_name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}


@router.post("/login")
async def login(
    body: LoginBody,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    print(request.url)
    print(request.headers)
    
    print(request.method)
    print(request.url.path)
    print(request.query_params)
    print(request.client)

    print(request.cookies)
    print(request.headers)
    user = await authenticate_user(db, body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(user["id"], user["username"])
    return {"token": token, "user": user}

@router.get("/me")
async def me(user: dict = Depends(get_current_user)):
    return {"user": user}
