"""User authentication: JWT tokens, password hashing, async user CRUD."""
from __future__ import annotations

import logging
import time
import uuid

import bcrypt
import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from settings import get_settings
JWT_SECRET = get_settings().security.jwt_secret.get_secret_value()
JWT_EXPIRE_HOURS = get_settings().security.jwt_expire_hours
from core.db.database import User

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_token(user_id: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "exp": time.time() + JWT_EXPIRE_HOURS * 3600,
        "iat": time.time(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except jwt.PyJWTError:
        return None


# ---------------------------------------------------------------------------
# User CRUD (async)
# ---------------------------------------------------------------------------

async def create_user(
    db: AsyncSession,
    username: str,
    password: str,
    display_name: str = "",
    role: str = "student",
) -> dict:
    # H-18：删除「用户名 == ADMIN_USERNAME 自动提权为 admin」。
    # 旧逻辑让任何人只要注册名为 admin 的用户名就获得管理员权限——典型越权提权。
    # admin 角色现仅由既有的安全通道授予：
    #   1. 迁移 002/003 把历史 is_admin=TRUE 用户升级（一次性）；
    #   2. 现有 admin 经 PUT /admin/users/{id}/role 显式授权（admin.py:662）；
    #   3. DBA 在数据库直接 UPDATE（应急/初始 bootstrap）。
    # ADMIN_USERNAME 仅作为保留名保留（不预占：撞名注册仍走唯一约束返回 409，
    # 不再附赠 admin 角色），消除「撞名即提权」攻击面。
    user = User(
        id=uuid.uuid4().hex[:12],
        username=username,
        password_hash=hash_password(password),
        display_name=display_name or username,
        role=role,
        is_admin=(role == "admin"),
        created_at=time.time(),
    )
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise ValueError("用户名已存在")
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "is_admin": (user.role == "admin"),
    }


async def authenticate_user(db: AsyncSession, username: str, password: str) -> dict | None:
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "is_admin": (user.role == "admin"),
    }


async def get_user_by_id(db: AsyncSession, user_id: str) -> dict | None:
    result = await db.execute(
        select(
            User.id,
            User.username,
            User.display_name,
            User.role,
            User.is_admin,
        ).where(User.id == user_id)
    )
    row = result.first()
    if not row:
        return None
    return {
        "id": row.id,
        "username": row.username,
        "display_name": row.display_name,
        "role": row.role,
        "is_admin": (row.role == "admin"),
    }
