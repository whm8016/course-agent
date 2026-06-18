"""OpenAI-compatible provider for DashScope and other OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import string
from typing import Any

import json_repair
from openai import AsyncOpenAI

from .base import LLMProvider, LLMResponse, ToolCallRequest

logger = logging.getLogger(__name__)

_ALLOWED_MSG_KEYS = frozenset(
    {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content", "extra_content"}
)
_ALNUM = string.ascii_letters + string.digits


def _short_tool_id() -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value if value else None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict) and dumped:
            return dumped
    return None


class OpenAICompatProvider(LLMProvider):
    """Unified provider for all OpenAI-compatible APIs (DashScope, DeepSeek, etc.)."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "qwen-plus",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers

        self._client = AsyncOpenAI(**client_kwargs)

    def get_default_model(self) -> str:
        return self.default_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        sanitized = self._sanitize_empty_content(messages)
        sanitized = self._sanitize_request_messages(sanitized, _ALLOWED_MSG_KEYS)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error("OpenAI-compat API error: %s", e)
            return LLMResponse(content=f"Error calling LLM: {e}", finish_reason="error")

        choice = response.choices[0] if response.choices else None
        if not choice:
            return LLMResponse(content="No response from model", finish_reason="error")

        message = choice.message
        content = message.content
        reasoning_content = getattr(message, "reasoning_content", None)

        tool_calls: list[ToolCallRequest] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tc_id = tc.id or _short_tool_id()
                func = tc.function
                raw_args = func.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    try:
                        args = json_repair.loads(raw_args)
                    except Exception:
                        args = {"raw": raw_args}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(ToolCallRequest(id=tc_id, name=func.name, arguments=args))

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
