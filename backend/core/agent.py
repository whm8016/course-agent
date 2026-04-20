"""Legacy agent module — kept for backwards compatibility.

The main orchestration logic now lives in core.orchestrator.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from core.llm import chat_stream
from core.prompts import get_course_prompt
from core.rag import retrieve_texts

logger = logging.getLogger(__name__)


def _build_rag_context(chunks: list[str]) -> str:
    if not chunks:
        return ""
    context = "\n\n---\n\n".join(chunks)
    return (
        f"\n\n【参考资料】以下是从课程知识库中检索到的相关内容，请结合这些资料回答：\n\n{context}\n\n---\n"
    )


async def handle_chat(
    course_id: str,
    message: str,
    history: list[dict],
    image_path: str | None = None,
) -> AsyncGenerator[str, None]:
    """Simple single-pass chat — no multi-agent orchestration."""
    logger.info("handle_chat (legacy) course=%s", course_id)
    system_prompt = await get_course_prompt(course_id)

    if not image_path and message.strip():
        try:
            chunks = retrieve_texts(course_id, message)
        except Exception:
            logger.exception("RAG retrieve failed")
            chunks = []
        if chunks:
            system_prompt += _build_rag_context(chunks)

    async for token in chat_stream(
        system_prompt=system_prompt,
        history=history,
        user_message=message,
        image_path=image_path,
    ):
        yield token
