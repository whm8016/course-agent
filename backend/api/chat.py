from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import get_current_user
from config import REDIS_URL
from core.orchestrator import run_agent_stream

logger = logging.getLogger(__name__)

router = APIRouter()

limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 50


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    course_id: str = body.get("course_id", "stamp")
    message: str = body.get("message", "")
    history: list[dict] = body.get("history", [])
    image_path: str | None = body.get("image_path")
    session_id: str | None = body.get("session_id")

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    logger.info(
        "POST /api/chat user=%s course=%s msg_len=%d session=%s",
        user["id"], course_id, len(message), session_id,
    )

    async def event_generator():
        try:
            async for event in run_agent_stream(course_id, message, history, image_path):
                data = json.dumps(event, ensure_ascii=False)
                yield f"data: {data}\n\n"
        except Exception as e:
            logger.exception("Agent pipeline error")
            error_data = json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
