from __future__ import annotations

import base64
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from openai import AsyncOpenAI

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, TEXT_MODEL, VISION_MODEL

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
if os.getenv("LANGSMITH_TRACING", "").strip().lower() in ("1", "true", "yes") and (
    os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
):
    try:
        from langsmith import wrappers

        _client = wrappers.wrap_openai(_client, chat_name="course_agent_chat")
    except Exception:
        logger.exception("LangSmith wrap_openai failed; using raw AsyncOpenAI client")
client = _client


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    suffix = path.suffix.lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    mime_type = mime.get(suffix, "image/png")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime_type};base64,{b64}"


def _build_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    image_path: str | None = None,
) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    if image_path:
        data_url = _image_to_data_url(image_path)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": user_message or "请赏析这张邮票图片。"},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_message})

    return messages


async def chat_stream(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    image_path: str | None = None,
) -> AsyncGenerator[str, None]:
    model = VISION_MODEL if image_path else TEXT_MODEL
    messages = _build_messages(system_prompt, history, user_message, image_path)

    logger.info(
        "LLM stream start model=%s msg_count=%d has_image=%s user_msg=「%s」",
        model, len(messages), bool(image_path), user_message[:80],
    )
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2048,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
