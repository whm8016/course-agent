from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from api.courses import check_course_access
from api.upload import (
    enforce_image_limit,
    materialize_attachments,
    resolve_attachments,
)
from core.attachment import Attachment
from core.context import UnifiedContext
from core.db.database import get_db
from core.db.limiter import limiter
from core.agent.mode_normalize import normalize_mode
from core.observability import log_flow
from services.session.turn_runtime import get_turn_runtime_manager

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 10


class ChatRequest(BaseModel):
    course_id: str = Field(default="", description="课程 ID（空=自由问答，不接入知识库）")
    message: str = Field(default="", description="用户消息")
    chat_mode: str = Field(default="chat", description="模式：chat / deep_solve / deep_research / quiz")
    history: list[dict[str, Any]] = Field(default_factory=list, description="历史消息列表")
    session_id: str | None = Field(default=None, description="会话 ID（可选）")
    image_path: str | None = Field(default=None, description="图片上传路径（可选，向后兼容旧单图入口）")
    attachments: list[Attachment] = Field(default_factory=list, description="附件列表（图片，支持多图）")
    tools: list[str] = Field(default_factory=list, description="启用的工具，如 ['rag', 'web_search']")
    model_profile_id: str | None = Field(default=None, description="本次对话使用的 LLM 供应商 profile id（对标 ：用户下拉选中；不传走默认/active）")
    rag_mode: str = Field(default="naive", description="LightRAG 检索模式：mix/naive/local/global，默认 naive（纯向量+rerank，0 次内部 LLM 调用）")


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    course_id: str = body.course_id or "general"
    message: str = body.message
    history: list[dict] = body.history
    session_id: str | None = body.session_id
    mode: str = normalize_mode(body.chat_mode)

    # rag_mode 白名单校验：只允许合法的 LightRAG 查询模式，非法/空值回退 naive
    rag_mode = (body.rag_mode or "").strip().lower()
    if rag_mode not in {"mix", "naive", "local", "global"}:
        rag_mode = "naive"

    # 附件解析 + 图片限流 + 物化（归属校验 + 读 base64），统一在 api.upload
    attachments = resolve_attachments(body.attachments, body.image_path)
    enforce_image_limit(attachments)
    materialize_attachments(attachments, user)
    image_count = sum(1 for a in attachments if a.is_image())

    await check_course_access(db, course_id, user)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    log_flow("http.chat.start", user_id=str(user["id"]), course_id=course_id,
             mode=mode, session_id=session_id or "", question=message[:120],
             attachments=image_count, tools=body.tools, rag_mode=rag_mode)

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
        rag_mode=rag_mode,
    )
    

    async def event_generator():
        trm = get_turn_runtime_manager()
        turn_id = await trm.start_turn(ctx)

        # 先把 turn_id 作为独立 SSE chunk 下发给前端（不进 bus，避免 subscribe 回放
        # _history 时重复）。前端据此调用 POST /chat/answer_now 触发"立即回答"。
        yield f"data: {json.dumps({'type': 'turn_started', 'turn_id': turn_id}, ensure_ascii=False)}\n\n"

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


class AnswerNowRequest(BaseModel):
    turn_id: str = Field(..., description="要触发立即回答的 turn id（来自 turn_started 事件）")


@router.post("/chat/answer_now")
@limiter.limit("30/minute")
async def answer_now(
    request: Request,
    body: AnswerNowRequest,
    user: dict = Depends(get_current_user),
):
    """触发"立即回答"：让正在思考的 turn 在下一轮顶部基于已有信息直接作答。

    前端流式过程中点"立即回答"按钮调用本端点（fire-and-forget，不中断当前 SSE 流）。
    后端 set answer_now_event，run_agent_loop 下一轮顶部检测到即跳过工具循环直接回答，
    答案仍经原 SSE 流下发。turn 不存在或已结束返回 404（静默，前端按钮此时多半已隐藏）。
    """
    trm = get_turn_runtime_manager()
    # 传 user_id 做归属校验：turn 不属于当前用户 → trm 返回 False → 404。
    # 防止 B 用户拿 A 的 turn_id 触发 A 的对话提前作答（Turn IDOR）。
    ok = await trm.request_answer_now(body.turn_id, user_id=str(user["id"]))
    log_flow("http.chat.answer_now", user_id=str(user["id"]),
             turn_id=body.turn_id, ok=ok)
    if not ok:
        raise HTTPException(status_code=404, detail="turn 不存在或已结束")
    return {"ok": True}
