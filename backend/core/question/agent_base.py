"""
出题 Agent 基类：DashScope OpenAI 兼容流式调用（替代 deeptutor BaseAgent 的最小子集）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from config import TEXT_MODEL
from core.llm import client as _openai_client

TraceCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class QuestionAgentBase:
    """提供 IdeaAgent / Generator 所需的 get_prompt、stream_llm、trace。"""

    def __init__(
        self,
        module_name: str,
        agent_name: str,
        language: str = "zh",
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (api_key, base_url, api_version, kwargs)
        self.module_name = module_name
        self.agent_name = agent_name
        self.language = language
        self.logger = logging.getLogger(f"{module_name}.{agent_name}")
        self._trace_callback: TraceCallback | None = None

    def set_trace_callback(self, callback: TraceCallback | None) -> None:
        self._trace_callback = callback

    async def _emit_trace_event(self, data: dict[str, Any]) -> None:
        if self._trace_callback is None:
            return
        try:
            res = self._trace_callback(data)
            if hasattr(res, "__await__"):
                await res
        except Exception:
            pass

    def get_prompt(self, name: str, default: str = "") -> str:
        """DeepTutor 用 YAML；此处先返回 default，由子类内联模板补齐。"""
        return default

    async def stream_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        messages: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        stage: str | None = None,
        attachments: list[Any] | None = None,
        trace_meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        _ = (stage, attachments, trace_meta, kwargs)
        m = model or TEXT_MODEL
        t = 0.7 if temperature is None else temperature
        mt = 4096 if max_tokens is None else max_tokens

        if messages:
            msgs = messages
        else:
            msgs = [
                {"role": "system", "content": system_prompt or "You are a helpful teaching assistant."},
                {"role": "user", "content": user_prompt},
            ]

        req: dict[str, Any] = {
            "model": m,
            "messages": msgs,
            "stream": True,
            "temperature": t,
            "max_tokens": mt,
        }
        if response_format is not None:
            req["response_format"] = response_format

        stream = await _openai_client.chat.completions.create(**req)
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if delta and delta.content:
                yield delta.content