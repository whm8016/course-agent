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
from core.codes import normalize_code
from core.db.database import (
    AsyncSessionLocal,
    ApplicationStatus,
    TeacherApplication,
    TeacherInvite,
    get_db,
)
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


def is_admin_user(user: dict) -> bool:
    """管理员判定单一数据源：以 role 为权威字段。

    is_admin 布尔列仅为历史冗余（迁移期反向同步用过），不参与任何判定——避免
    role/is_admin 双轨制下漏同步导致的判定不一致（不同接口读不同字段）。所有
    "是不是管理员"的判断（权限校验 / 越权放行 / 回显派生）都应走此函数。
    """
    return user.get("role") == "admin"


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """管理员专用依赖（FastAPI Depends）；非管理员 403。

    llm/mcp/search_config 等模块统一用此 Depends（取代各自复制的 _require_admin），
    把"管理员判定"收敛到 is_admin_user 一处，杜绝双轨制。
    """
    if not is_admin_user(user):
        raise HTTPException(status_code=403, detail="仅管理员可访问")
    return user


# 历史别名：admin.py / main.py / llama_rag.py 已大量使用 get_current_admin，保留以
# 减少改动面；与 require_admin 完全等价（同一 function，同走 is_admin_user）。
get_current_admin = require_admin


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
@limiter.limit("30/minute")
async def register(request: Request, body: RegisterBody, db: AsyncSession = Depends(get_db)):
    role = "student"
    invite_row = None

    if body.invite_code:
        result = await db.execute(
            select(TeacherInvite).where(TeacherInvite.code == normalize_code(body.invite_code))
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


class ApplyTeacherBody(BaseModel):
    reason: str = Field(..., min_length=1, max_length=500)


@router.post("/apply-teacher", status_code=201)
@limiter.limit("5/minute")
async def apply_teacher(
    request: Request,
    body: ApplyTeacherBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """学生提交教师身份申请，等待管理员审批（与邀请码即时升级并存）。"""
    # 守卫1：已是教师/管理员，无需申请
    if user.get("role") in ("teacher", "admin"):
        raise HTTPException(status_code=409, detail="您已是教师或管理员")
    # 守卫2：已有待审批申请（业务层校验；DB 部分唯一索引兜底并发双击）
    existing = await db.execute(
        select(TeacherApplication).where(
            TeacherApplication.user_id == user["id"],
            TeacherApplication.status == ApplicationStatus.PENDING.value,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="您已提交申请，请等待审批")
    application = TeacherApplication(
        user_id=user["id"],
        reason=body.reason.strip(),
        status=ApplicationStatus.PENDING.value,
    )
    db.add(application)
    await db.flush()
    logger.info("用户 %s 提交教师申请 app=%s", user["id"], application.id)
    return {
        "id": application.id,
        "status": application.status,
        "message": "申请已提交，等待管理员审批",
    }


@router.get("/teacher-applications/me")
async def my_teacher_application(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """查看当前用户最新的教师申请状态（无申请则 status=None）。"""
    result = await db.execute(
        select(TeacherApplication)
        .where(TeacherApplication.user_id == user["id"])
        .order_by(TeacherApplication.created_at.desc())
        .limit(1)
    )
    app = result.scalar_one_or_none()
    if not app:
        return {"status": None}
    return {
        "id": app.id,
        "status": app.status,
        "reason": app.reason,
        "created_at": app.created_at,
        "reviewed_at": app.reviewed_at,
        "review_note": app.review_note,
    }


@router.post("/login")
@limiter.limit("40/minute")
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
