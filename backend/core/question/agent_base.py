"""
出题 Agent 基类：DashScope OpenAI 兼容流式调用（替代 deeptutor BaseAgent 的最小子集）。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from settings import get_settings
TEXT_MODEL = get_settings().llm.text_model
from core.llm.llm import _is_stream_options_unsupported, client as _openai_client
from core.observability.cost import usage_from_response_chunk

TraceCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class QuestionAgentBase:
    """提供 Generator 所需的 get_prompt、stream_llm、trace（出题 agent 基类）。"""

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
        """从 prompts/{language}/{agent_name}.yaml 读取指定 prompt，找不到则返回 default。"""
        try:
            import yaml
            from pathlib import Path
            prompt_path = (
                Path(__file__).parent          # core/question/
                / "prompts"
                / self.language                # zh
                / f"{self.agent_name}.yaml"    # generator.yaml
            )
            if not prompt_path.exists():
                self.logger.warning(f"prompt file not found: {prompt_path}")
                return default
            with open(prompt_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                self.logger.warning(f"prompt file is not a dictionary: {prompt_path}")
                return default
            return str(data.get(name, default) or default)
        except Exception as exc:
            self.logger.warning(f"get_prompt failed ({self.agent_name}/{name}): {exc}")
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
        track: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """流式出题/追问 LLM 调用。

        track 传入时启用 usage 采集（stream_options 末块），流结束经
        track_llm_usage 埋 Prometheus + 用量明细；不传则不带 stream_options，
        行为与原先逐字节一致（评测/脚本路径零变化）。

        已知取舍：本函数是 async generator，消费方中途弃置（如 WS 断线路由 return）
        会跳过流末尾的 usage 块与 post-loop 埋点——该次 usage 丢失，可接受（best-effort）。
        """
        _ = (stage, attachments, trace_meta, kwargs)
        m = model or TEXT_MODEL
        t = 0.7 if temperature is None else temperature
        mt = 8192 if max_tokens is None else max_tokens

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
        if track is not None:
            req["stream_options"] = {"include_usage": True}

        stream = await self._create_stream(req)
        usage = None
        async for chunk in stream:
            # usage 末块 choices 为空，必须在下方 `if not choice: continue` 之前取
            # （loop._one_round 同款教训，见 cost.usage_from_response_chunk docstring）
            if track is not None:
                _u = usage_from_response_chunk(chunk)
                if _u is not None:
                    usage = _u
            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue
            delta = choice.delta
            if delta and delta.content:
                yield delta.content

        if track is not None and usage is not None:
            from core.analytics.token_usage import track_llm_usage

            await track_llm_usage(usage, model=m, **track)

    async def _create_stream(self, req: dict[str, Any]):
        """create + stream_options 安全降级：端点不支持时报 400 而非静默忽略，
        剥掉重试一次（丢本轮 usage 但保流式不挂，与 llm._create_with_image_fallback
        同套路；此处直连 client 享受不到那层的降级，本地复用同一判定函数）。"""
        try:
            return await _openai_client.chat.completions.create(**req)
        except Exception as exc:
            if "stream_options" not in req or not _is_stream_options_unsupported(exc):
                raise
            req.pop("stream_options", None)
            self.logger.info("quiz followup：端点不支持 stream_options，已剥离重试（本轮无 usage）")
            return await _openai_client.chat.completions.create(**req)