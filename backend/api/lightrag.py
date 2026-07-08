from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_teacher
from api.courses import check_course_access
from core.db.database import get_db
from settings import get_settings
LIGHTRAG_TIMEOUT_SEC = get_settings().lightrag.timeout_sec
from core.rag import get_indexer, is_lightrag_available

logger = logging.getLogger(__name__)

from core.db.limiter import limiter

router = APIRouter()

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 10
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


class LightRagChatRequest(BaseModel):
    course_id: str = Field(default="stamp", description="课程 ID")
    message: str = Field(default="", description="用户消息")
    history: list[dict[str, Any]] = Field(default_factory=list, description="历史消息列表")
    mode: str | None = Field(default=None, description="显式模式（可选，优先级高于 chat_mode）")
    chat_mode: str = Field(default="chat", description="模式：chat / deep_solve / deep_research / quiz")
    session_id: str | None = Field(default=None, description="会话 ID（可选）")
    tools: list[str] = Field(default_factory=list, description='启用的工具，如 ["rag", "web_search"]')
    image_path: str | None = Field(default=None, description="图片上传路径（可选）")


# [DEPRECATED] 旧 chat 路径（agentic_pipeline 四阶段：Thinking→Acting→Observing→Responding）。
# 新路径 POST /api/chat（run_agent_loop + rag tool）已具备 LightRAG 检索 + 安全护栏 + 真流式，
# 可完整替代本路径（rag tool 直接复用 lightrag_engine）。彻底下线需联动前端 api.ts 的
# ragEnabled 路由切换，作为独立的前后端重构任务。



@router.post("/chat/lightrag/index")
@limiter.limit("10/minute")
async def index_lightrag(
    request: Request,
    body: IndexBody,
    user: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    ok, reason = is_lightrag_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    # 课程归属校验：教师只能索引自己拥有的课程，防止传他人 course_id 触发索引
    # （绕过 _get_owned_kb 的课程越权）。非 owner → check_course_access 抛 403。
    await check_course_access(db, body.course_id, user)

    logger.info(
        "POST /api/chat/lightrag/index user=%s course=%s force=%s source_dir=%s",
        user["id"],
        body.course_id,
        body.force,
        body.source_dir,
    )

    indexer = get_indexer("lightrag")
    result = await asyncio.wait_for(
        indexer.index(body.course_id, [], force=body.force, source_dir=body.source_dir),
        timeout=LIGHTRAG_TIMEOUT_SEC,
    )
    return {
        "engine": "lightrag",
        "course_id": body.course_id,
        "indexed_docs": result.chunks_created,
        "indexed_files": result.files_indexed,
        "skipped": result.status == "skipped",
        "reason": result.error,
    }