from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.database import get_db
from core.learner_profile import build_memory_context, update_learner_memory
from core.limiter import limiter
from core.orchestrator import normalize_mode, run_agent_stream

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 20


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    body = await request.json()
    course_id: str = body.get("course_id", "stamp")
    message: str = body.get("message", "")
    history: list[dict] = body.get("history", [])
    image_path: str | None = body.get("image_path")
    session_id: str | None = body.get("session_id")
    mode: str = normalize_mode(body.get("chat_mode", "chat"))

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    logger.info(
        "POST /api/chat user=%s course=%s mode=%s session=%s question=「%s」",
        user["id"], course_id, mode, session_id, message[:120],
    )

    async def event_generator():
        answer_content = ""
        final_mode = mode
        try:
            async for event in run_agent_stream(
                course_id,
                message,
                history,
                image_path,
                mode=mode,
                memory_context=build_memory_context(user),
            ):
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected, stop stream user=%s course=%s session=%s",
                        user["id"],
                        course_id,
                        session_id,
                    )
                    return
                if event.get("type") == "answer":
                    answer_content = str(event.get("content") or "")
                if event.get("type") == "done":
                    metadata = event.get("metadata") or {}
                    final_mode = str(metadata.get("mode") or final_mode)
                    await update_learner_memory(
                        db,
                        user["id"],
                        course_id=course_id,
                        mode=final_mode,
                        user_message=message,
                        assistant_answer=answer_content,
                    )
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
