"""REST API for TutorBot management, notifications, social binding, and reminders.

所有端点需登录；bot / reminder 按 owner（user_id）隔离。跨用户越权统一返回 404
（不泄露 bot 存在性）。admin 可额外操作扁平 legacy bot（owner_id 为空），但不能
触碰其他用户的私有 bot。bot_id 强制安全约束以防目录穿越。
"""

from __future__ import annotations

import re
import time
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user, is_admin_user
from core.db.database import BotNotification, UserSocialBinding, get_db
from core.bot.manager import get_bot_manager, BotConfig
from core.bot.notification import NotificationService
from services.cron.service import CronJob, CronOwner, CronSchedule, get_cron_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bot", tags=["bot"])

# bot_id 安全约束（防止目录穿越 / owner 命名空间污染）
_BOT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


# --- Request/Response Models ---

class BotCreateRequest(BaseModel):
    bot_id: str
    name: str
    description: str = ""
    persona: str = ""
    model: str | None = None
    course_id: str = ""  # 绑定课程后 bot 可调用 rag 工具
    channels: dict[str, Any] = {}


class BotMessageRequest(BaseModel):
    content: str
    chat_id: str = "web"


class NotifyRequest(BaseModel):
    user_id: str | None = None
    course_id: str | None = None
    content: str
    platform: str | None = None


class BindRequest(BaseModel):
    platform: str
    platform_user_id: str
    chat_id: str = ""
    display_name: str = ""


class ReminderScheduleRequest(BaseModel):
    kind: str              # "at" | "every" | "cron"
    at_ms: int | None = None
    every_seconds: int | None = None
    expr: str | None = None
    tz: str | None = None


class ReminderCreateRequest(BaseModel):
    name: str = ""
    message: str
    schedule: ReminderScheduleRequest
    channel: str = "web"   # "qq" | "feishu" | "web"
    chat_id: str = ""
    session_key: str = ""


# --- 鉴权 / 归属 helpers ---

def _owner_of(user: dict) -> str:
    return str(user.get("id") or "")


def _is_admin(user: dict) -> bool:
    # 走 auth.is_admin_user 单一判定源（只读 role），消除 role/is_admin 双轨制
    return is_admin_user(user)


def _resolve_owner(manager, owner_id: str, bot_id: str, is_admin: bool) -> str:
    """定位 bot 的实际 owner_id（用于后续操作）。

    命中自己的 → owner_id；admin 命中 legacy → ""；否则 404（不区分"不存在"与"不属于"）。
    """
    own_ids = {b["bot_id"] for b in manager.list_bots(owner_id)}
    if bot_id in own_ids:
        return owner_id
    if is_admin:
        legacy_ids = {b["bot_id"] for b in manager.list_bots("", include_legacy=True)}
        if bot_id in legacy_ids:
            return ""
    raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")


# --- Bot CRUD ---

@router.get("/list")
async def list_bots(user: dict = Depends(get_current_user)):
    """List bot instances owned by the current user (admin also sees legacy)."""
    manager = get_bot_manager()
    owner = _owner_of(user)
    bots = manager.list_bots(owner, include_legacy=_is_admin(user))
    return {"bots": bots}


# --- Notifications（web 定时提醒触达，按 user_id 隔离）---

@router.get("/notifications")
async def list_notifications(
    unread: bool = False,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """当前用户的 bot 通知（cron 触发的 web 提醒落库后，前端轮询拉取）。"""
    owner = _owner_of(user)
    q = select(BotNotification).where(BotNotification.user_id == owner)
    if unread:
        q = q.where(BotNotification.read.is_(False))
    q = q.order_by(BotNotification.created_at.desc()).limit(50)
    result = await db.execute(q)
    rows = result.scalars().all()
    return {
        "notifications": [
            {
                "id": r.id,
                "bot_id": r.bot_id,
                "content": r.content,
                "read": r.read,
                "created_at": r.created_at,
            }
            for r in rows
        ],
        "unread_count": sum(1 for r in rows if not r.read),
    }


@router.post("/notifications/{notif_id}/read")
async def mark_notification_read(
    notif_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """标记一条通知已读（仅本人）。"""
    result = await db.execute(
        select(BotNotification).where(BotNotification.id == notif_id)
    )
    n = result.scalar_one_or_none()
    if not n:
        raise HTTPException(status_code=404, detail="通知不存在")
    if n.user_id != _owner_of(user):
        raise HTTPException(status_code=403, detail="无权操作")
    n.read = True
    await db.flush()
    return {"status": "read"}


@router.post("/create")
async def create_bot(req: BotCreateRequest, user: dict = Depends(get_current_user)):
    """Create and start a new bot (owned by the current user)."""
    if not _BOT_ID_RE.match(req.bot_id):
        raise HTTPException(
            status_code=400, detail="bot_id 需匹配 ^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$"
        )
    manager = get_bot_manager()
    owner = _owner_of(user)
    config = BotConfig(
        name=req.name,
        description=req.description,
        persona=req.persona,
        model=req.model,
        course_id=req.course_id,
        channels=req.channels,
        owner_id=owner,
    )
    instance = await manager.start_bot(owner, req.bot_id, config)
    return {"status": "started", "bot": instance.to_dict()}


@router.post("/{bot_id}/start")
async def start_bot(bot_id: str, user: dict = Depends(get_current_user)):
    """Start a stopped bot."""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    instance = await manager.start_bot(actual_owner, bot_id)
    return {"status": "started", "bot": instance.to_dict()}


@router.post("/{bot_id}/stop")
async def stop_bot(bot_id: str, user: dict = Depends(get_current_user)):
    """Stop a running bot."""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    success = await manager.stop_bot(actual_owner, bot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found or not running")
    return {"status": "stopped", "bot_id": bot_id}


@router.delete("/{bot_id}")
async def delete_bot_route(bot_id: str, user: dict = Depends(get_current_user)):
    """删除 bot（停止 + 删除持久化配置）。"""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    ok = await manager.delete_bot(actual_owner, bot_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    return {"deleted": True}


class BotUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    persona: str | None = None
    course_id: str | None = None


@router.put("/{bot_id}")
async def update_bot_route(bot_id: str, req: BotUpdateRequest, user: dict = Depends(get_current_user)):
    """更新 bot 配置（name/description/persona/course_id）；运行中则重启以应用新配置。"""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    try:
        await manager.update_bot(
            actual_owner,
            bot_id,
            name=req.name,
            description=req.description,
            persona=req.persona,
            course_id=req.course_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found")
    return {"updated": True, "bot_id": bot_id}


@router.get("/{bot_id}/history")
async def get_bot_history(bot_id: str, limit: int = 100, user: dict = Depends(get_current_user)):
    """Get conversation history for a bot."""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    history = manager.get_bot_history(actual_owner, bot_id, limit=limit)
    return {"messages": history}


@router.post("/{bot_id}/message")
async def send_message(bot_id: str, req: BotMessageRequest, user: dict = Depends(get_current_user)):
    """Send a message to a bot and get the response (carries caller identity)."""
    manager = get_bot_manager()
    owner = _owner_of(user)
    actual_owner = _resolve_owner(manager, owner, bot_id, _is_admin(user))
    try:
        response = await manager.send_message(
            actual_owner, bot_id, req.content, chat_id=req.chat_id, user_id=owner
        )
        return {"response": response}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Notifications（教师/管理员主动推送）---

@router.post("/notify")
async def send_notification(
    req: NotifyRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Push notification to a student or broadcast to a course."""
    if not (_is_admin(user) or user.get("role") == "teacher"):
        raise HTTPException(status_code=403, detail="仅教师/管理员可发送通知")
    manager = get_bot_manager()
    svc = NotificationService(manager)

    if req.user_id:
        sent_to = await svc.push_to_student(db, req.user_id, req.content, platform=req.platform)
        return {"status": "sent", "platforms": sent_to}
    elif req.course_id:
        count = await svc.broadcast(db, req.course_id, req.content)
        return {"status": "broadcast", "notified_count": count}
    else:
        raise HTTPException(status_code=400, detail="Either user_id or course_id is required")


# --- Social Platform Binding（绑定当前登录用户的 IM 账号）---

@router.post("/bind")
async def create_binding(
    req: BindRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bind the current user to a social platform account."""
    owner = _owner_of(user)
    existing = await db.execute(
        select(UserSocialBinding).where(
            UserSocialBinding.platform == req.platform,
            UserSocialBinding.platform_user_id == req.platform_user_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="This platform account is already bound")

    binding = UserSocialBinding(
        user_id=owner,
        platform=req.platform,
        platform_user_id=req.platform_user_id,
        chat_id=req.chat_id,
        display_name=req.display_name,
        created_at=time.time(),
    )
    db.add(binding)
    await db.flush()
    return {"status": "bound", "id": binding.id, "platform": req.platform}


@router.delete("/bind/{binding_id}")
async def delete_binding(
    binding_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a social platform binding (only the owner's)."""
    owner = _owner_of(user)
    result = await db.execute(
        select(UserSocialBinding).where(UserSocialBinding.id == binding_id)
    )
    binding = result.scalar_one_or_none()
    if not binding:
        raise HTTPException(status_code=404, detail="Binding not found")
    if binding.user_id != owner and not _is_admin(user):
        raise HTTPException(status_code=403, detail="无权删除他人的绑定")
    await db.execute(delete(UserSocialBinding).where(UserSocialBinding.id == binding_id))
    return {"status": "deleted"}


@router.post("/bind/code")
async def gen_bind_code(user: dict = Depends(get_current_user)):
    """生成一次性绑定码（6 位，10 分钟有效）：在 IM 私聊 bot 发「绑定 <码>」即可绑定账号。"""
    from core.bot.binding import add_bind_code

    owner = _owner_of(user)
    code = add_bind_code(owner)
    return {"code": code, "expires_in": 600}


@router.get("/bindings/me")
async def my_bindings(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """当前登录用户的 IM 绑定列表。"""
    owner = _owner_of(user)
    result = await db.execute(
        select(UserSocialBinding).where(UserSocialBinding.user_id == owner)
    )
    bindings = result.scalars().all()
    return {
        "bindings": [
            {
                "id": b.id,
                "platform": b.platform,
                "platform_user_id": b.platform_user_id,
                "chat_id": b.chat_id,
                "display_name": b.display_name,
                "created_at": b.created_at,
            }
            for b in bindings
        ]
    }


@router.get("/bindings/{user_id}")
async def get_user_bindings(user_id: str, user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get social platform bindings (own or, for admin, any user's)."""
    owner = _owner_of(user)
    if user_id != owner and not _is_admin(user):
        raise HTTPException(status_code=403, detail="无权查看他人的绑定")
    result = await db.execute(
        select(UserSocialBinding).where(UserSocialBinding.user_id == user_id)
    )
    bindings = result.scalars().all()
    return {
        "bindings": [
            {
                "id": b.id,
                "platform": b.platform,
                "platform_user_id": b.platform_user_id,
                "chat_id": b.chat_id,
                "display_name": b.display_name,
                "created_at": b.created_at,
            }
            for b in bindings
        ]
    }


# --- Reminders (Cron) ---

def _job_to_dict(job: CronJob) -> dict:
    return {
        "id": job.id,
        "name": job.name,
        "message": job.message,
        "schedule": {
            "kind": job.schedule.kind,
            "at_ms": job.schedule.at_ms,
            "every_seconds": job.schedule.every_seconds,
            "expr": job.schedule.expr,
            "tz": job.schedule.tz,
        },
        "channel": job.owner.channel,
        "chat_id": job.owner.chat_id,
        "enabled": job.enabled,
        "created_at_ms": job.created_at_ms,
        "state": {
            "next_run_at_ms": job.state.next_run_at_ms,
            "last_run_at_ms": job.state.last_run_at_ms,
            "last_status": job.state.last_status,
            "last_error": job.state.last_error,
        },
    }


def _reminder_owner_key(actual_owner: str, bot_id: str) -> str:
    return f"partner:{actual_owner}:{bot_id}"


@router.get("/{bot_id}/reminders")
async def list_reminders(bot_id: str, user: dict = Depends(get_current_user)):
    """List all scheduled reminders for a bot."""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    cron = get_cron_service()
    jobs = cron.list_jobs(owner_key=_reminder_owner_key(actual_owner, bot_id))
    return {"reminders": [_job_to_dict(j) for j in jobs]}


@router.post("/{bot_id}/reminders")
async def create_reminder(bot_id: str, req: ReminderCreateRequest, user: dict = Depends(get_current_user)):
    """Create a scheduled reminder for a bot.

    When the reminder fires, the bot sends the message via the specified channel (qq/feishu/web).
    """
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))

    schedule = CronSchedule(
        kind=req.schedule.kind,
        at_ms=req.schedule.at_ms,
        every_seconds=req.schedule.every_seconds,
        expr=req.schedule.expr,
        tz=req.schedule.tz,
    )
    owner = CronOwner(
        kind="partner",
        partner_id=f"{actual_owner}:{bot_id}",
        channel=req.channel,
        chat_id=req.chat_id,
        session_key=req.session_key,
        user_id=_owner_of(user),
    )

    try:
        cron = get_cron_service()
        job = cron.add_job(
            name=req.name,
            message=req.message,
            schedule=schedule,
            owner=owner,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"status": "created", "reminder": _job_to_dict(job)}


@router.delete("/{bot_id}/reminders/{job_id}")
async def cancel_reminder(bot_id: str, job_id: str, user: dict = Depends(get_current_user)):
    """Cancel a scheduled reminder."""
    manager = get_bot_manager()
    actual_owner = _resolve_owner(manager, _owner_of(user), bot_id, _is_admin(user))
    cron = get_cron_service()
    success = cron.cancel_job(job_id, owner_key=_reminder_owner_key(actual_owner, bot_id))
    if not success:
        raise HTTPException(status_code=404, detail="Reminder not found or not owned by this bot")
    return {"status": "cancelled", "job_id": job_id}
