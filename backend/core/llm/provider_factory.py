"""LLM client factory — 按 binding/backend 构造正确的 LLM client。

对外暴露两个构造入口：
 - get_llm_client()              —— 启动期按 ProviderSpec.backend 构造固定 client
 - get_llm_client_for_profile()  —— 运行期按 catalog profile 取（缓存的）client

两者返回与 openai.AsyncOpenAI 接口兼容的 client：
 - openai_compat  → openai.AsyncOpenAI（覆盖 openai/deepseek/dashscope/zhipuai/
                     moonshot/groq/gemini/openrouter/siliconflow/ollama/vllm/lm_studio）
 - anthropic      → AnthropicAdapter（原生 SDK 包装为 OpenAI 兼容接口）
 - azure_openai   → openai.AsyncAzureOpenAI
 - 未知 binding   → 按 openai_compat 处理，使用用户提供的 base_url
"""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from settings import get_settings
LLM_BINDING = get_settings().llm.binding
DASHSCOPE_API_KEY = get_settings().llm.api_key.get_secret_value()
LLM_TIMEOUT_SEC = get_settings().llm.timeout_sec
from core.llm.provider_registry import find_by_name, find_by_model

logger = logging.getLogger(__name__)


def get_llm_client(
    *,
    binding: str | None = None,
    api_key: str,
    base_url: str | None = None,
    api_version: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> Any:
    """构造并返回与 openai.AsyncOpenAI 接口兼容的 LLM client。

    参数：
        binding:     供应商标识（provider_registry.ProviderSpec.name），如 "deepseek"。
                     为 None 或未知时，尝试从 model 名推断；均失败则按 openai_compat 处理。
        api_key:     对应供应商的 API key。
        base_url:    覆盖端点 URL；为空时使用 ProviderSpec.default_api_base。
        api_version: Azure OpenAI 专用。
        model:       辅助推断 binding（binding 为空时使用）。
        timeout:     HTTP 超时秒数。

    返回：
        AsyncOpenAI / AsyncAzureOpenAI / AnthropicAdapter，
        均暴露 client.chat.completions.create(**kwargs) 接口。
    """
    spec = find_by_name(binding)
    if spec is None and model:
        spec = find_by_model(model)

    backend = spec.backend if spec else "openai_compat"

    # ---- Anthropic 原生 SDK ------------------------------------------------
    if backend == "anthropic":
        from core.llm.providers.anthropic_adapter import AnthropicAdapter

        logger.info("LLMFactory: anthropic backend (binding=%r model=%r)", binding, model)
        return AnthropicAdapter(
            api_key=api_key or None,
            base_url=base_url or (spec.default_api_base if spec else None),
        )

    # ---- Azure OpenAI ------------------------------------------------------
    if backend == "azure_openai":
        if not base_url:
            raise ValueError(
                "Azure OpenAI 需要配置 base_url（azure_endpoint），"
                "请在 model_catalog.json 或 .env 中填写。"
            )
        logger.info("LLMFactory: azure_openai backend (endpoint=%r)", base_url)
        return AsyncAzureOpenAI(
            api_key=api_key or "sk-no-key-required",
            azure_endpoint=base_url,
            api_version=api_version or "2024-02-01",
        )

    # ---- OpenAI 兼容（默认）-----------------------------------------------
    effective_base = base_url or (spec.default_api_base if spec else None)
    logger.info(
        "LLMFactory: openai_compat backend (binding=%r base_url=%r)",
        binding or (spec.name if spec else "unknown"),
        effective_base,
    )
    return AsyncOpenAI(
        api_key=api_key or "sk-no-key-required",
        base_url=effective_base,
        timeout=timeout,
    )


# ============================================================
# 运行期 profile 动态 client
# ============================================================
# 按 profile 指纹缓存：同一 (binding, key, base_url, api_version, timeout) 复用同一
# client，避免 per-request 频繁 new。profile 字段为空时回退 config 启动常量（.env
# 兜底），使 active=default（catalog key/base_url 通常空）也能正确构造——因此「不选
# profile」与「选 active profile」走同一 client，行为一致。
_CLIENT_CACHE: dict[str, Any] = {}


def _profile_fingerprint(
    binding: str, api_key: str, base_url: str | None, api_version: str | None, timeout: int
) -> str:
    return json.dumps(
        {
            "binding": binding,
            "key": api_key,
            "base": base_url,
            "ver": api_version,
            "to": timeout,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def get_llm_client_for_profile(
    profile: dict[str, Any] | None,
    *,
    timeout: int = LLM_TIMEOUT_SEC,
) -> Any:
    """按 catalog profile 构造 / 取缓存的 LLM client（与 get_llm_client 同接口）。

    profile 字段为空时回退 config 启动常量（active=default 的 catalog 通常 key/base_url
    为空，靠 .env 兜底）。
    """
    profile = profile or {}
    binding = (profile.get("binding") or "").strip() or LLM_BINDING
    api_key = (profile.get("api_key") or "").strip() or DASHSCOPE_API_KEY
    base_url = (profile.get("base_url") or "").strip() or None
    api_version = (profile.get("api_version") or "").strip() or None

    fp = _profile_fingerprint(binding, api_key, base_url, api_version, timeout)
    cached = _CLIENT_CACHE.get(fp)
    if cached is not None:
        return cached

    client = get_llm_client(
        binding=binding,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        timeout=timeout,
    )
    # profile client 接入 LangSmith wrap（在缓存前应用，避免缓存的是未 wrap 实例）。
    # 非 OpenAI 兼容实例（AnthropicAdapter）或未启用时原样返回。
    from core.observability.langsmith_trace import wrap_openai_client

    client = wrap_openai_client(client, chat_name=f"profile:{binding}")
    _CLIENT_CACHE[fp] = client
    return client


def clear_llm_client_cache() -> None:
    """清空 profile client 缓存（运维 / 测试用；正常路径按指纹自动正确，无需手动清）。"""
    _CLIENT_CACHE.clear()


__all__ = ["get_llm_client", "get_llm_client_for_profile", "clear_llm_client_cache"]
