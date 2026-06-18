"""REST API for TutorBot management, notifications, and social binding."""

from __future__ import annotations

import time
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import AsyncSessionLocal, UserSocialBinding, User, get_db
from core.bot.manager import get_bot_manager, BotConfig
from core.bot.notification import NotificationService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bot", tags=["bot"])


# --- Request/Response Models ---

class BotCreateRequest(BaseModel):
    bot_id: str
    name: str
    description: str = ""
    persona: str = ""
    model: str | None = None
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
    user_id: str
    platform: str
    platform_user_id: str
    chat_id: str = ""
    display_name: str = ""


# --- Bot CRUD ---

@router.get("/list")
async def list_bots():
    """List all bot instances (running + configured)."""
    manager = get_bot_manager()
    return {"bots": manager.list_bots()}


@router.post("/create")
async def create_bot(req: BotCreateRequest):
    """Create and start a new bot."""
    manager = get_bot_manager()
    config = BotConfig(
        name=req.name,
        description=req.description,
        persona=req.persona,
        model=req.model,
        channels=req.channels,
    )
    instance = await manager.start_bot(req.bot_id, config)
    return {"status": "started", "bot": instance.to_dict()}


@router.post("/{bot_id}/start")
async def start_bot(bot_id: str):
    """Start a stopped bot."""
    manager = get_bot_manager()
    instance = await manager.start_bot(bot_id)
    return {"status": "started", "bot": instance.to_dict()}


@router.post("/{bot_id}/stop")
async def stop_bot(bot_id: str):
    """Stop a running bot."""
    manager = get_bot_manager()
    success = await manager.stop_bot(bot_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Bot '{bot_id}' not found or not running")
    return {"status": "stopped", "bot_id": bot_id}


@router.get("/{bot_id}/history")
async def get_bot_history(bot_id: str, limit: int = 100):
    """Get conversation history for a bot."""
    manager = get_bot_manager()
    history = manager.get_bot_history(bot_id, limit=limit)
    return {"messages": history}


@router.post("/{bot_id}/message")
async def send_message(bot_id: str, req: BotMessageRequest):
    """Send a message to a bot and get the response."""
    manager = get_bot_manager()
    try:
        response = await manager.send_message(bot_id, req.content, chat_id=req.chat_id)
        return {"response": response}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Notifications ---

@router.post("/notify")
async def send_notification(req: NotifyRequest, db: AsyncSession = Depends(get_db)):
    """Push notification to a student or broadcast to a course."""
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


# --- Social Platform Binding ---

@router.post("/bind")
async def create_binding(req: BindRequest, db: AsyncSession = Depends(get_db)):
    """Bind a user to a social platform account."""
    # Verify user exists
    user_result = await db.execute(select(User).where(User.id == req.user_id))
    if not user_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="User not found")

    # Check existing binding
    existing = await db.execute(
        select(UserSocialBinding).where(
            UserSocialBinding.platform == req.platform,
            UserSocialBinding.platform_user_id == req.platform_user_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="This platform account is already bound")

    binding = UserSocialBinding(
        user_id=req.user_id,
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
async def delete_binding(binding_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a social platform binding."""
    result = await db.execute(
        delete(UserSocialBinding).where(UserSocialBinding.id == binding_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Binding not found")
    return {"status": "deleted"}


@router.get("/bindings/{user_id}")
async def get_user_bindings(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get all social platform bindings for a user."""
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
