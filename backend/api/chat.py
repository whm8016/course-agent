from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.courses import check_course_access
from api.upload import assert_upload_owner, resolve_upload_path
from core.attachment import Attachment, AttachmentType
from core.context import UnifiedContext
from core.llm.multimodal import _guess_mime_type
from core.db.database import get_db
from core.db.limiter import limiter
from core.agent.orchestrator import normalize_mode
from core.observability import log_flow
from services.session.turn_runtime import get_turn_runtime_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 20
MAX_IMAGES_PER_TURN = 4  # 单轮最多图片附件数（防多图 base64 涨 token 致超时）


class ChatRequest(BaseModel):
    course_id: str = Field(default="stamp", description="课程 ID")
    message: str = Field(default="", description="用户消息")
    chat_mode: str = Field(default="chat", description="模式：chat / deep_solve / deep_research / quiz")
    history: list[dict[str, Any]] = Field(default_factory=list, description="历史消息列表")
    session_id: str | None = Field(default=None, description="会话 ID（可选）")
    image_path: str | None = Field(default=None, description="图片上传路径（可选，向后兼容旧单图入口）")
    attachments: list[Attachment] = Field(default_factory=list, description="附件列表（图片，支持多图）")
    tools: list[str] = Field(default_factory=list, description="启用的工具，如 ['rag', 'web_search']")
    model_profile_id: str | None = Field(default=None, description="本次对话使用的 LLM 供应商 profile id（对标 DeepTutor：用户下拉选中；不传走默认/active）")


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    course_id: str = body.course_id
    message: str = body.message
    history: list[dict] = body.history
    session_id: str | None = body.session_id
    mode: str = normalize_mode(body.chat_mode)

    # 附件解析：attachments 列表优先，回退旧 image_path 单图
    attachments: list[Attachment] = [a for a in (body.attachments or [])]
    if body.image_path and not any(a.is_image() for a in attachments):
        attachments.append(Attachment(type=AttachmentType.IMAGE, url=body.image_path))

    # 限流：单轮图片数上限（防多图 base64 涨 token 致超时）
    image_count = sum(1 for a in attachments if a.is_image())
    if image_count > MAX_IMAGES_PER_TURN:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_IMAGES_PER_TURN} 张图片")

    # 多租户归属校验 + URL→磁盘路径 + 读盘填 base64（图片与文档统一处理）。
    # base64 是图片注入（multimodal）与文档文本提取（loop）的共同前提；在此一次
    # 性填好，下游 loop 不再重复读盘。本地 /api/uploads/ 路径才校验归属 + 读盘，
    # 外部 http(s) URL 不读（安全：不下载远端文档，图片由 multimodal 以 URL 形式发）。
    for att in attachments:
        url = att.url or ""
        is_local = any(url.startswith(p) for p in ("/api/uploads/", "/uploads/"))
        if not is_local:
            continue
        assert_upload_owner(url.rsplit("/", 1)[-1], user)
        att.file_path = resolve_upload_path(url) or att.file_path
        if att.file_path and os.path.isfile(att.file_path):
            att.base64 = base64.b64encode(Path(att.file_path).read_bytes()).decode("ascii")
            if att.is_image():
                if not att.mime_type:
                    att.mime_type = _guess_mime_type(att.filename or att.file_path)
            elif not att.filename:
                att.filename = os.path.basename(att.file_path)

    await check_course_access(db, course_id, user)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    log_flow("http.chat.start", user_id=str(user["id"]), course_id=course_id,
             mode=mode, session_id=session_id or "", question=message[:120],
             attachments=image_count, tools=body.tools)

    # 读 Mem0 记忆（L3）：用当前用户消息做 query 语义检索相关记忆注入
    from core.memory.mem0_client import build_memory_context as _mem_ctx_fn, has_any as _mem0_has_any
    _mem_ctx = await _mem_ctx_fn(str(user["id"]), message)
    _has_memory = await _mem0_has_any(str(user["id"]))
    logger.info(
        "[chat] memory context built user_id=%s has_memory=%s ctx_len=%d",
        user["id"], _has_memory, len(_mem_ctx)
    )

    # 读 Session Summary（L2）：早期对话摘要
    _session_summary = ""
    if session_id:
        try:
            from core.memory.session_summary import get_summary_manager
            summary_mgr = get_summary_manager()
            _session_summary = await summary_mgr.get_summary(db, session_id)
            if _session_summary:
                logger.info(
                    "[chat] L2 summary loaded user_id=%s session=%s summary_len=%d",
                    user["id"], session_id, len(_session_summary)
                )
        except Exception as e:
            logger.warning("[chat] L2 summary load failed: %s", e)

    ctx = UnifiedContext(
        course_id=course_id,
        user_id=str(user["id"]),
        session_id=session_id or "",
        user_message=message,
        conversation_history=history,
        attachments=attachments,
        mode=mode,
        enabled_tools=body.tools,
        memory_context=_mem_ctx,  # L3
        session_summary=_session_summary,  # L2
        llm_profile_id=body.model_profile_id or "",
        metadata={"has_memory": _has_memory},
    )
    

    async def event_generator():
        trm = get_turn_runtime_manager()
        turn_id = await trm.start_turn(ctx)

        try:
            async for event in trm.subscribe_turn(turn_id):
                yield f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

                if await request.is_disconnected():
                    log_flow("http.chat.sse_disconnect", turn_id=turn_id,
                             user_id=str(user["id"]), course_id=course_id)
                    await trm.cancel_turn(turn_id)
                    return
        finally:
            await trm.cancel_turn(turn_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
