from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import get_current_user
from config import LIGHTRAG_TIMEOUT_SEC, REDIS_URL
from core.lightrag_engine import (
    index_course_with_lightrag,
    is_lightrag_available,
    retrieve_with_lightrag,
    stream_answer_with_contexts,
)

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 50
TOOL_RESULT_CONTEXT_LIMIT = 4
TOOL_RESULT_CONTEXT_MAX_CHARS = 300


def _compact_contexts_for_sse(contexts: list[object]) -> list[object]:
    compacted: list[object] = []
    for ctx in contexts[:TOOL_RESULT_CONTEXT_LIMIT]:
        if isinstance(ctx, str):
            compacted.append(ctx[:TOOL_RESULT_CONTEXT_MAX_CHARS])
            continue
        if isinstance(ctx, dict):
            row = dict(ctx)
            for key in ("content", "text", "chunk"):
                value = row.get(key)
                if isinstance(value, str) and len(value) > TOOL_RESULT_CONTEXT_MAX_CHARS:
                    row[key] = f"{value[:TOOL_RESULT_CONTEXT_MAX_CHARS]}...(truncated)"
            compacted.append(row)
            continue
        compacted.append(str(ctx)[:TOOL_RESULT_CONTEXT_MAX_CHARS])
    return compacted


class IndexBody(BaseModel):
    course_id: str
    force: bool = False
    source_dir: str | None = None


@router.post("/chat/lightrag")
@limiter.limit("20/minute")
async def chat_with_lightrag(request: Request, user: dict = Depends(get_current_user)):
    ok, reason = is_lightrag_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    body = await request.json()
    course_id: str = body.get("course_id", "stamp")
    message: str = body.get("message", "")
    history: list[dict] = body.get("history", [])
    mode: str | None = body.get("mode")
    session_id: str | None = body.get("session_id")
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    logger.info(
        "[trace=%s] POST /api/chat/lightrag user=%s course=%s msg_len=%d session=%s mode=%s",
        trace_id, user["id"], course_id, len(message), session_id, mode,
    )

    async def event_generator():
        try:
            logger.info("[trace=%s] retrieve_start t=%dms", trace_id, elapsed_ms())
            yield f"data: {json.dumps({'type': 'thinking', 'content': '正在使用 LightRAG 检索知识图谱与向量证据...'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'tool_call', 'tool': 'lightrag_query', 'input': {'course_id': course_id, 'mode': mode or 'mix'}}, ensure_ascii=False)}\n\n"

            retrieve_result = await asyncio.wait_for(
                retrieve_with_lightrag(course_id=course_id, message=message, history=history, mode=mode),
                timeout=LIGHTRAG_TIMEOUT_SEC,
            )

            contexts = retrieve_result.get("contexts") or []
            logger.info(
                "[trace=%s] retrieve_end t=%dms strategy=%s contexts=%d",
                trace_id,
                elapsed_ms(),
                retrieve_result.get("retrieve_strategy", "unknown"),
                len(contexts),
            )
            if contexts:
                compact_contexts = _compact_contexts_for_sse(contexts)
                yield f"data: {json.dumps({'type': 'tool_result', 'tool': 'lightrag_query', 'contexts': compact_contexts}, ensure_ascii=False)}\n\n"

            answer_parts: list[str] = []
            first_token_logged = False
            async for token in stream_answer_with_contexts(
                course_id=course_id,
                message=message,
                contexts=contexts,
                history=history,
            ):
                if not first_token_logged:
                    logger.info("[trace=%s] first_token t=%dms", trace_id, elapsed_ms())
                    first_token_logged = True
                answer_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            answer = "".join(answer_parts)
            yield f"data: {json.dumps({'type': 'answer', 'content': answer}, ensure_ascii=False)}\n\n"
            logger.info("[trace=%s] done t=%dms answer_chars=%d", trace_id, elapsed_ms(), len(answer))
            yield f"data: {json.dumps({'type': 'done', 'metadata': {'engine': 'lightrag', 'mode': retrieve_result.get('mode', 'mix'), 'retrieve_strategy': retrieve_result.get('retrieve_strategy', 'unknown')}}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            logger.warning("[trace=%s] timeout t=%dms", trace_id, elapsed_ms())
            error_data = json.dumps({"type": "error", "content": "LightRAG 查询超时"}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
        except Exception as exc:
            logger.exception("[trace=%s] LightRAG pipeline error t=%dms", trace_id, elapsed_ms())
            error_data = json.dumps({"type": "error", "content": str(exc)}, ensure_ascii=False)
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


@router.post("/chat/lightrag/index")
@limiter.limit("10/minute")
async def index_lightrag(request: Request, body: IndexBody, user: dict = Depends(get_current_user)):
    ok, reason = is_lightrag_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    logger.info(
        "POST /api/chat/lightrag/index user=%s course=%s force=%s source_dir=%s",
        user["id"],
        body.course_id,
        body.force,
        body.source_dir,
    )
    result = await asyncio.wait_for(
        index_course_with_lightrag(body.course_id, force=body.force, source_dir=body.source_dir),
        timeout=LIGHTRAG_TIMEOUT_SEC,
    )
    return {"engine": "lightrag", "course_id": body.course_id, **result}
