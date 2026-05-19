"""
ResearchBaseAgent - Faithful port of DeepTutor BaseAgent.

Preserves the full interface (call_llm / stream_llm / get_prompt nested /
set_trace_callback / _emit_trace_event / _track_tokens) while routing LLM
calls to this project's DashScope-compatible OpenAI client.
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable

import yaml

from config import TEXT_MODEL
from core.llm.llm import client as _openai_client


TraceCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class BaseAgent(ABC):
    """
    Unified base class for all research-module agents.

    Mirrors DeepTutor's BaseAgent interface:
    - call_llm (non-streaming, accumulates stream internally)
    - stream_llm (async generator, yields chunks)
    - get_prompt (nested section.field and simple key modes)
    - set_trace_callback / _emit_trace_event
    - _track_tokens (token-tracker interface stub)
    - get_model / get_temperature / get_max_tokens / get_max_retries
    - refresh_config / is_enabled / has_prompts
    """

    TraceCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

    def __init__(
        self,
        module_name: str,
        agent_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_version: str | None = None,
        language: str = "zh",
        binding: str | None = None,
        config: dict[str, Any] | None = None,
        token_tracker: Any | None = None,
        log_dir: str | None = None,
    ):
        self.module_name = module_name
        self.agent_name = agent_name
        self.language = language
        self._trace_callback: BaseAgent.TraceCallback | None = None

        # config is always a plain dict
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        else:
            self.config = {}

        # LLM settings – fall back to env-vars matching project convention
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
        self.base_url = base_url or os.getenv("LLM_HOST")
        self.model = model or TEXT_MODEL
        self.api_version = api_version
        self.binding = binding or "openai"

        # Per-agent config block (mirrors DeepTutor agents.yaml lookup)
        self.agent_config: dict[str, Any] = self.config.get("agents", {}).get(agent_name, {})
        llm_cfg = self.config.get("llm", {})
        self.llm_config: dict[str, Any] = llm_cfg if isinstance(llm_cfg, dict) else {}

        self.enabled: bool = self.agent_config.get("enabled", True)
        self.token_tracker = token_tracker

        self.logger = logging.getLogger(f"research.{module_name}.{agent_name}")

        # Load prompts via the same YAML layout as DeepTutor PromptManager
        try:
            self.prompts = self._load_prompts()
            if self.prompts:
                self.logger.debug("Prompts loaded: %s (%s)", agent_name, language)
        except Exception as exc:
            self.prompts = None
            self.logger.warning("Failed to load prompts for %s: %s", agent_name, exc)

    # ------------------------------------------------------------------ #
    # Prompt loading – mirrors DeepTutor PromptManager.load_prompts()
    # ------------------------------------------------------------------ #

    def _load_prompts(self) -> dict[str, Any] | None:
        prompt_path = (
            Path(__file__).parent
            / "prompts"
            / self.language
            / f"{self.agent_name}.yaml"
        )
        if not prompt_path.exists():
            self.logger.warning("Prompt file not found: %s", prompt_path)
            return None
        with open(prompt_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None

    def get_prompt(
        self,
        section_or_type: str = "system",
        field_or_fallback: str | None = None,
        fallback: str = "",
    ) -> str | None:
        """
        DeepTutor BaseAgent.get_prompt interface:

        - get_prompt("system", "role")           → prompts["system"]["role"]
        - get_prompt("process", "rephrase", "")  → prompts["process"]["rephrase"]
        - get_prompt("system")                   → prompts["system"]  (simple key)
        """
        if not self.prompts:
            return (
                fallback
                if fallback
                else (
                    field_or_fallback
                    if isinstance(field_or_fallback, str) and field_or_fallback
                    else None
                )
            )

        section_value = self.prompts.get(section_or_type)

        if isinstance(section_value, dict) and field_or_fallback is not None:
            result = section_value.get(field_or_fallback)
            if result is not None:
                return result
            return fallback if fallback else None
        else:
            if section_value is not None:
                return section_value
            return field_or_fallback if field_or_fallback else (fallback if fallback else None)

    def has_prompts(self) -> bool:
        return self.prompts is not None

    # ------------------------------------------------------------------ #
    # Model / parameter getters – same priority chain as DeepTutor
    # ------------------------------------------------------------------ #

    def get_model(self) -> str:
        if self.agent_config.get("model"):
            return self.agent_config["model"]
        if self.llm_config.get("model"):
            return self.llm_config["model"]
        if self.model:
            return self.model
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            return env_model
        raise ValueError(
            f"Model not configured for agent {self.agent_name}. "
            "Please set TEXT_MODEL in config."
        )

    def get_temperature(self) -> float:
        return float(self.agent_config.get("temperature", 0.7))

    def get_max_tokens(self) -> int:
        return int(self.agent_config.get("max_tokens", 8192))

    def get_max_retries(self) -> int:
        return int(self.agent_config.get("max_retries", 2))

    def is_enabled(self) -> bool:
        return self.enabled

    def refresh_config(self) -> None:
        """Interface stub – kept for DeepTutor API compatibility."""

    # ------------------------------------------------------------------ #
    # Trace
    # ------------------------------------------------------------------ #

    def set_trace_callback(self, callback: TraceCallback | None) -> None:
        """Register a trace callback that receives structured LLM call events."""
        self._trace_callback = callback

    async def _emit_trace_event(self, payload: dict[str, Any]) -> None:
        callback = self._trace_callback
        if callback is None:
            return
        try:
            result = callback(payload)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            self.logger.debug("Trace callback failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Token tracking – stub that forwards to external tracker if provided
    # ------------------------------------------------------------------ #

    def _track_tokens(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response: str,
        stage: str | None = None,
    ) -> None:
        stage_label = stage or self.agent_name
        if self.token_tracker:
            try:
                self.token_tracker.add_usage(
                    agent_name=self.agent_name,
                    stage=stage_label,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_text=response,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # LLM interface – identical signatures to DeepTutor BaseAgent
    # ------------------------------------------------------------------ #

    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        messages: list[dict[str, Any]] | None = None,
        response_format: dict[str, str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        verbose: bool = True,
        stage: str | None = None,
        attachments: list[Any] | None = None,
        trace_meta: dict[str, Any] | None = None,
    ) -> str:
        """Non-streaming LLM call (accumulates chunks from stream_llm)."""
        chunks: list[str] = []
        async for chunk in self.stream_llm(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            stage=stage,
            attachments=attachments,
            trace_meta=trace_meta,
        ):
            chunks.append(chunk)
        return "".join(chunks)

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
    ) -> AsyncGenerator[str, None]:
        """Streaming LLM call – yields text chunks."""
        m = model or self.get_model()
        t = self.get_temperature() if temperature is None else temperature
        mt = self.get_max_tokens() if max_tokens is None else max_tokens
        stage_label = stage or self.agent_name

        if messages:
            msgs = messages
        else:
            msgs = [
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
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

        trace_base = {
            "event": "llm_call",
            "state": "running",
            "agent_name": self.agent_name,
            "stage": stage_label,
            "model": m,
            "temperature": t,
            "max_tokens": mt,
            "streaming": True,
            **(trace_meta or {}),
        }
        await self._emit_trace_event(trace_base)

        self.logger.debug(
            "LLM stream input %s:%s model=%s system_chars=%d user_chars=%d",
            self.agent_name,
            stage_label,
            m,
            len(system_prompt or ""),
            len(user_prompt or ""),
        )

        start = time.time()
        full_response = ""
        try:
            stream = await _openai_client.chat.completions.create(**req)
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta
                if delta and delta.content:
                    full_response += delta.content
                    await self._emit_trace_event(
                        {**trace_base, "state": "streaming", "chunk": delta.content}
                    )
                    yield delta.content

            self._track_tokens(m, system_prompt or "", user_prompt or "", full_response, stage_label)

            call_duration = time.time() - start
            await self._emit_trace_event(
                {
                    **trace_base,
                    "state": "complete",
                    "response": full_response,
                    "duration": call_duration,
                }
            )
            self.logger.debug(
                "LLM stream output %s:%s chars=%d duration=%.2fs",
                self.agent_name,
                stage_label,
                len(full_response),
                call_duration,
            )

        except Exception as exc:
            await self._emit_trace_event({**trace_base, "state": "error", "response": str(exc)})
            self.logger.error("LLM streaming failed: %s", exc)
            raise

    # ------------------------------------------------------------------ #
    # Abstract
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Main processing logic (must be implemented by subclasses)."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"module={self.module_name}, "
            f"name={self.agent_name}, "
            f"enabled={self.enabled})"
        )


__all__ = ["BaseAgent"]
