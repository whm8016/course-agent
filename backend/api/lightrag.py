from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_teacher, get_current_user
from api.courses import check_course_access
from api.upload import resolve_upload_path
from config import AGENTIC_RAG_BACKEND, FAQ_CACHE_THRESHOLD, LIGHTRAG_TIMEOUT_SEC
from core.db.database import get_db
from core.memory.learner_profile import build_memory_context, update_learner_memory
from core.memory.graph_memory import update_graphs_from_conversation
from core.skills.output_skills import generate_skill_outputs
from core.rag.lightrag_engine import (
    index_course_with_lightrag,
    is_lightrag_available,
    agentic_pipeline,
)
from core.llm.llm import chat_stream
from core.agent.orchestrator import normalize_mode
from core.llm.prompts import get_course_prompt
from core.agent.safety_pipeline import (
    INTENT_CHITCHAT,
    INTENT_KNOWLEDGE,
    classify_intent,
    evaluate_guardrail,
    evaluate_hallucination,
)

logger = logging.getLogger(__name__)

from core.db.cache import faq_answer_get, faq_answer_set, faq_record
from core.db.limiter import limiter

router = APIRouter()

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 20
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
@limiter.limit("200/minute")
async def chat_with_lightrag(
    request: Request,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ok, reason = is_lightrag_available()
    if not ok:
        raise HTTPException(status_code=503, detail=reason)

    body = await request.json()
    course_id: str = body.get("course_id", "stamp")
    message: str = body.get("message", "")
    history: list[dict] = body.get("history", [])
    mode: str | None = body.get("mode")
    chat_mode: str = normalize_mode(body.get("chat_mode", "chat"))
    session_id: str | None = body.get("session_id")
    enabled_tools: list[str] = body.get("tools", [])  # 如 ["rag","web_search"]；与 AGENTIC_RAG_BACKEND 一起决定实际检索
    if enabled_tools:
        if AGENTIC_RAG_BACKEND == "llamaindex":
            enabled_tools = [
                "llamaindex_rag" if t == "rag" else t for t in enabled_tools
            ]
        else:
            enabled_tools = [
                "rag" if t == "llamaindex_rag" else t for t in enabled_tools
            ]
    logger.info(
        "AGENTIC_RAG_BACKEND=%s enabled_tools=%s",
        AGENTIC_RAG_BACKEND,
        enabled_tools,
    )
    image_path: str | None = resolve_upload_path(body.get("image_path"))
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:8]
    t0 = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    await check_course_access(db, course_id, user)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    logger.info(
        "[trace=%s] POST /api/chat/lightrag user=%s course=%s session=%s chat_mode=%s rag_mode=%s question=「%s」has_image=%s",
        trace_id, user["id"], course_id, session_id, chat_mode, mode, message[:120], image_path is not None,
    )

    async def event_generator():
        answer = ""
        try:
            if await request.is_disconnected():
                logger.info("[trace=%s] client already disconnected before stream start", trace_id)
                return

            # ── FAQ 高频统计 + 缓存命中检查 ───────────────────────────
            faq_count = await faq_record(course_id, message)
            if FAQ_CACHE_THRESHOLD > 0 and faq_count >= FAQ_CACHE_THRESHOLD:
                cached = await faq_answer_get(course_id, message)
                if cached:
                    logger.info(
                        "[trace=%s] FAQ cache HIT course=%s count=%d question=%s",
                        trace_id, course_id, faq_count, message[:60],
                    )
                    yield f"data: {json.dumps({'type': 'answer', 'content': cached}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'metadata': {'engine': 'faq_cache'}}, ensure_ascii=False)}\n\n"
                    return

            # ── Step 1: Intent classification ────────────────────────
            intent_result = await classify_intent(message, history)
            logger.info(
                "[trace=%s] intent=%s confidence=%.2f reason=%s t=%dms",
                trace_id, intent_result.intent, intent_result.confidence,
                intent_result.reason, elapsed_ms(),
            )
            yield f"data: {json.dumps({'type': 'thinking', 'content': f'意图识别: {intent_result.intent}'}, ensure_ascii=False)}\n\n"

            # ── Step 2: Safety guardrail ──────────────────────────────
            guard_result = evaluate_guardrail(message)
            guardrail_dict = guard_result.to_dict()
            logger.info(
                "[trace=%s] guardrail safe=%s risk_type=%s score=%.2f t=%dms",
                trace_id, guard_result.safe, guard_result.risk_type,
                guard_result.risk_score, elapsed_ms(),
            )

            if not guard_result.safe:
                logger.warning(
                    "[trace=%s] guardrail BLOCKED risk=%s score=%.2f question=「%s」",
                    trace_id, guard_result.risk_type, guard_result.risk_score,
                    message[:80],
                )

            # ── Step 3: Route by intent ──────────────────────────────
            contexts: list = []
            retrieve_result: dict = {}
            hallucination_dict: dict = {}

            if intent_result.intent == INTENT_CHITCHAT:
                logger.info(
                    "[trace=%s] ▶ route=chitchat (skip RAG, direct LLM) t=%dms",
                    trace_id, elapsed_ms(),
                )
                yield f"data: {json.dumps({'type': 'thinking', 'content': '闲聊模式，直接回复...'}, ensure_ascii=False)}\n\n"

                system_prompt = await get_course_prompt(course_id)
                mem_ctx = build_memory_context(user)
                if mem_ctx:
                    system_prompt += f"\n\n{mem_ctx}"
                if not guard_result.safe:
                    system_prompt += "\n\n【安全提示】请围绕课程内容回答，拒绝不当请求。"

                from core.rag.lightrag_engine import _normalize_history, _cap_history
                safe_history = _cap_history(_normalize_history(history))

                answer_parts: list[str] = []
                first_token_logged = False
                async for token in chat_stream(
                    system_prompt=system_prompt,
                    history=safe_history,
                    user_message=message,
                    image_path=image_path,
                ):
                    if await request.is_disconnected():
                        return
                    if not first_token_logged:
                        logger.info("[trace=%s] first_token t=%dms", trace_id, elapsed_ms())
                        first_token_logged = True
                    answer_parts.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

                answer = "".join(answer_parts)
              

            else:
                logger.info(
                    "[trace=%s] ▶ route=knowledge (agentic pipeline) t=%dms",
                    trace_id, elapsed_ms(),
                )
                # yield f"data: {json.dumps({'type': 'thinking', 'content': '正在分析问题、检索证据并整理回答...'}, ensure_ascii=False)}\n\n"

                # answer_parts: list[str] = []
                # first_token_logged = False
                # async for token in agentic_pipeline(
                #     course_id=course_id,
                #     message=message,
                #     history=history,
                #     mode=mode,
                # ):
                #     if await request.is_disconnected():
                #         logger.info("[trace=%s] client disconnected during agentic stream", trace_id)
                #         return
                #     if not first_token_logged:
                #         logger.info("[trace=%s] first_token t=%dms (agentic)", trace_id, elapsed_ms())
                #         first_token_logged = True
                #     answer_parts.append(token)
                #     yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                # answer = "".join(answer_parts)
                answer_parts: list[str] = []
                first_token_logged = False
                agentic_contexts: list = []
                _STAGE_LABELS = {
                    "thinking": "分析问题",
                    "retrieving": "检索知识图谱",
                    "observing": "整理证据",
                    "responding": "生成回答",
                }
                async for event in agentic_pipeline(
                    course_id=course_id,
                    message=message,
                    history=history,
                    mode=mode,
                    enabled_tools=enabled_tools,
                    image_path=image_path,
                    memory_context=build_memory_context(user),
                ):
                    if await request.is_disconnected():
                        return
                    if event["type"] == "stage":
                        stage = event["stage"]
                        state = event["state"]
                        label = _STAGE_LABELS.get(stage, stage)
                        call_state = "running" if state == "start" else "complete"
                        display = f"{label}..." if state == "start" else f"{label} ✓"
                        yield f"data: {json.dumps({'type': 'thinking', 'content': display, 'stage': stage, 'call_state': call_state}, ensure_ascii=False)}\n\n"
                    elif event["type"] == "stage_chunk":
                        yield f"data: {json.dumps({'type': 'thinking_chunk', 'content': event['content'], 'stage': event['stage']}, ensure_ascii=False)}\n\n"
                    elif event["type"] == "token":
                        token = event["content"]
                        if not first_token_logged:
                            logger.info("[trace=%s] first_token t=%dms (agentic)", trace_id, elapsed_ms())
                            first_token_logged = True
                        answer_parts.append(token)
                        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                    elif event["type"] == "contexts":
                        agentic_contexts = event.get("contexts", [])
                answer = "".join(answer_parts)

                logger.info(
                    "[trace=%s] agentic_pipeline done answer_chars=%d t=%dms",
                    trace_id, len(answer), elapsed_ms(),
                )

                if await request.is_disconnected():
                    return

                # hallucination check，使用 agentic pipeline 实际检索到的 contexts
                hallu_result = await evaluate_hallucination(answer, agentic_contexts)
                hallucination_dict = hallu_result.to_dict()
                logger.info(
                    "[trace=%s] hallucination grounded=%s confidence=%.2f t=%dms",
                    trace_id, hallu_result.grounded, hallu_result.confidence, elapsed_ms(),
                )

            yield f"data: {json.dumps({'type': 'answer', 'content': answer}, ensure_ascii=False)}\n\n"
            logger.info(
                "[trace=%s] ✅ DONE intent=%s answer_chars=%d total_time=%dms question=「%s」",
                trace_id, intent_result.intent, len(answer), elapsed_ms(),
                message[:60],
            )

            metadata: dict[str, object] = {
                "engine": "lightrag",
                "mode": chat_mode,
                "intent": intent_result.intent,
                "intent_confidence": intent_result.confidence,
                "guardrail": guardrail_dict,
            }
            if isinstance(retrieve_result, dict):
                metadata["retrieve_mode"] = retrieve_result.get("mode", mode or "mix")
                metadata["retrieve_strategy"] = retrieve_result.get("retrieve_strategy", "lightrag_native")
            elif retrieve_result:
                metadata["retrieve_mode"] = mode or "mix"
                metadata["retrieve_strategy"] = "lightrag_native"
            metadata["answer_engine"] = "lightrag_native"
            if hallucination_dict:
                metadata["hallucination"] = hallucination_dict

            await update_learner_memory(
                db,
                user["id"],
                course_id=course_id,
                mode=chat_mode,
                user_message=message,
                assistant_answer=answer,
            )
            await update_graphs_from_conversation(
                db,
                user["id"],
                course_id=course_id,
                user_message=message,
                assistant_answer=answer,
            )

            # ── FAQ 答案写缓存（阈值达到且本次未命中）──────────────────
            if FAQ_CACHE_THRESHOLD > 0 and faq_count >= FAQ_CACHE_THRESHOLD and answer:
                await faq_answer_set(course_id, message, answer)
                logger.info(
                    "[trace=%s] FAQ answer cached course=%s count=%d question=%s",
                    trace_id, course_id, faq_count, message[:60],
                )

            # ── Custom Output Skills 补充框 ──────────────────────────
            try:
                skill_outputs = await generate_skill_outputs(
                    course_id=course_id,
                    user_message=message,
                    assistant_answer=answer,
                )
                for so in skill_outputs:
                    yield f"data: {json.dumps({'type': 'skill_output', **so}, ensure_ascii=False)}\n\n"
            except Exception:
                logger.debug("Skill output generation failed", exc_info=True)

            yield f"data: {json.dumps({'type': 'done', 'metadata': metadata}, ensure_ascii=False)}\n\n"
        except asyncio.TimeoutError:
            logger.warning("[trace=%s] timeout t=%dms", trace_id, elapsed_ms())
            error_data = json.dumps({"type": "error", "content": "LightRAG 查询超时"}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"
        except Exception:
            logger.exception("[trace=%s] LightRAG pipeline error t=%dms", trace_id, elapsed_ms())
            error_data = json.dumps(
                {"type": "error", "content": "对话处理失败，请稍后重试"},
                ensure_ascii=False,
            )
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
async def index_lightrag(request: Request, body: IndexBody, user: dict = Depends(get_current_teacher)):
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
