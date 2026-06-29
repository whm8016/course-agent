"""Agent orchestrator — mode 规范化与非 chat 流式节点。

chat 路径已迁移到 core/agentic/loop.py（tool_calls 驱动的 Agent Loop）。
deep_solve / deep_research / quiz 各自走独立 Capability pipeline
（core/solve、core/research、core/question）。

本文件仅保留仍被引用的少量节点：
  normalize_mode            — API 层 mode 规范化（api/chat、api/lightrag 用）
  _stream_summarize_events  — 学习小结流式生成（SummarizeCapability 用）
  _stream_vision_events     — 图片分析流式生成（VisionCapability 用）

历史遗留的 router_node（LLM 意图分类）、quiz_node（结构化出题）、OFF_TOPIC_REPLY
已被 tool_calls 机制（LLM 自主决定是否调 rag）和独立 pipeline 取代，已移除。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from core.context import UnifiedContext
from core.llm.llm import chat_stream
from core.llm.prompts import SUMMARY_PROMPT, get_course_prompt

logger = logging.getLogger(__name__)

# 多 worker 下调整并发限制: 总并发 / worker 数
_WORKERS = int(__import__("os").getenv("BACKEND_WORKERS", "4"))
_MAX_CONCURRENT_LLM = int(__import__("os").getenv("MAX_CONCURRENT_LLM", "25"))
_LLM_SEMAPHORE = asyncio.Semaphore(max(1, _MAX_CONCURRENT_LLM // _WORKERS))


def normalize_mode(mode: str | None) -> str:
    """规范化前端传入的 chat_mode，兼容旧值。"""
    allowed = {"chat", "deep_solve", "deep_research", "quiz", "vision", "summarize"}
    if not mode:
        return "chat"
    normalized = mode.strip().lower()
    # 兼容旧前端传 "research"
    if normalized == "research":
        normalized = "deep_research"
    return normalized if normalized in allowed else "chat"


async def _stream_summarize_events(ctx: UnifiedContext) -> AsyncGenerator[dict, None]:
    """基于对话历史流式生成学习小结。"""
    yield {"type": "thinking", "content": "正在分析对话历史，生成学习小结..."}

    history_text = ""
    for msg in ctx.conversation_history[-14:]:
        role_label = "学生" if msg["role"] == "user" else "助教"
        history_text += f"{role_label}: {msg['content']}\n\n"

    if not history_text.strip():
        yield {"type": "answer", "content": "当前还没有足够的对话内容来生成总结。请先提几个问题吧！"}
        return

    system_prompt = f"""{SUMMARY_PROMPT}

以下是对话历史：
{history_text}
"""
    if ctx.memory_context:
        system_prompt += f"\n\n{ctx.memory_context}"
    user_message = "请生成学习小结。"
    answer_parts: list[str] = []
    await _LLM_SEMAPHORE.acquire()
    try:
        async for token in chat_stream(
            system_prompt=system_prompt,
            history=[],
            user_message=user_message,
            image_path=None,
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    finally:
        _LLM_SEMAPHORE.release()

    yield {"type": "answer", "content": "".join(answer_parts)}


async def _stream_vision_events(ctx: UnifiedContext) -> AsyncGenerator[dict, None]:
    """结合课程知识流式分析上传的图片。"""
    yield {"type": "thinking", "content": "正在分析图片内容..."}

    system_prompt = await get_course_prompt(ctx.course_id)
    if ctx.memory_context:
        system_prompt += f"\n\n{ctx.memory_context}"
    user_message = ctx.user_message or "请描述这张图片。"
    answer_parts: list[str] = []
    await _LLM_SEMAPHORE.acquire()
    try:
        async for token in chat_stream(
            system_prompt=system_prompt,
            history=ctx.conversation_history[-10:],
            user_message=user_message,
            attachments=ctx.attachments or [],
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    finally:
        _LLM_SEMAPHORE.release()

    yield {"type": "answer", "content": "".join(answer_parts)}
