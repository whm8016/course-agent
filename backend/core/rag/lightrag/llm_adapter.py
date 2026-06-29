"""LightRAG LLM/Embedding 适配器。

从 lightrag_engine.py 提取的 LLM 调用和 Embedding 函数，
负责：
- LLM 调用适配（DashScope/OpenAI 兼容 API）
- Embedding 函数适配
- LLM 错误收集
- 可用性检查
"""
from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    LIGHTRAG_EMBEDDING_DIM,
    LIGHTRAG_ENABLED,
    LIGHTRAG_LLM_SYSTEM_MAX_CHARS,
    TEXT_MODEL,
)
from core.observability.langsmith_trace import safe_traceable

logger = logging.getLogger(__name__)

# ── LightRAG 导入（延迟，处理不可用情况）───────────────────────────────────────

try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import wrap_embedding_func_with_attrs

    LIGHTRAG_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    openai_complete_if_cache = None  # type: ignore[assignment]
    openai_embed = None  # type: ignore[assignment]
    wrap_embedding_func_with_attrs = None  # type: ignore[assignment]
    LIGHTRAG_IMPORT_ERROR = exc


# ── LLM 错误收集（LightRAG 内部会吞掉异常，这里在抛出前记录）─────────────────

_llm_error_log: list[Exception] = []


def take_llm_errors() -> list[Exception]:
    """取出并清空已记录的 LLM 错误列表（每批插入后调用）。"""
    errors = _llm_error_log.copy()
    _llm_error_log.clear()
    return errors


def clear_llm_errors() -> None:
    """清空错误缓冲（开始新索引前调用）。"""
    _llm_error_log.clear()


def _is_fatal_llm_error(exc: Exception) -> bool:
    """判断是否为致命错误（账户余额/权限问题），不可重试。"""
    s = str(exc).lower()
    fatal_keywords = (
        "access denied", "account", "unauthorized", "authentication",
        "bad request", "quota", "insufficient",
    )
    if any(kw in s for kw in fatal_keywords):
        return True
    m = re.search(r"error code[:\s]+(\d+)", s)
    if m and int(m.group(1)) in (400, 401, 403):
        return True
    return False


# ── 可用性检查 ────────────────────────────────────────────────────────────────


def is_lightrag_available() -> tuple[bool, str]:
    """检查 LightRAG 是否可用。

    Returns:
        (is_available, error_message) 元组
    """
    if not LIGHTRAG_ENABLED:
        return False, "LIGHTRAG_ENABLED 未开启"
    if LIGHTRAG_IMPORT_ERROR is not None:
        return False, f"LightRAG 依赖不可用: {LIGHTRAG_IMPORT_ERROR}"
    if not DASHSCOPE_API_KEY:
        return False, "缺少 DASHSCOPE_API_KEY"
    return True, ""


# ── LLM 调用函数 ───────────────────────────────────────────────────────────────


@safe_traceable(name="lightrag.llm", run_type="llm")
async def _llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str:
    """LightRAG LLM 调用适配函数。

    使用 DashScope/OpenAI 兼容 API 完成 LLM 调用。
    """
    assert openai_complete_if_cache is not None, "LightRAG LLM 不可用"

    max_sys_chars = LIGHTRAG_LLM_SYSTEM_MAX_CHARS
    safe_system_prompt = system_prompt
    if max_sys_chars > 0 and safe_system_prompt and len(safe_system_prompt) > max_sys_chars:
        safe_system_prompt = safe_system_prompt[:max_sys_chars]

    try:
        return await openai_complete_if_cache(
            TEXT_MODEL,
            prompt,
            system_prompt=safe_system_prompt,
            history_messages=history_messages or [],
            # DeepSeek API 不支持 response_format 结构化输出。
            # LightRAG 已用 json_repair.loads() 解析返回文本，无需 structured output。
            keyword_extraction=False,
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            **kwargs,
        )
    except Exception as exc:
        # LightRAG 内部会捕获此异常并继续，但我们在此先记录
        _llm_error_log.append(exc)
        logger.error("LLM 调用失败（已记录）: %s", exc)
        raise


# ── Embedding 函数 ─────────────────────────────────────────────────────────────


if wrap_embedding_func_with_attrs is not None and openai_embed is not None:

    @wrap_embedding_func_with_attrs(
        embedding_dim=LIGHTRAG_EMBEDDING_DIM,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
    )
    async def _embedding_func(texts: list[str]) -> np.ndarray:
        """LightRAG Embedding 适配函数。"""
        return await openai_embed.func(
            texts,
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

else:

    async def _embedding_func(texts: list[str]) -> np.ndarray:  # pragma: no cover
        """Embedding 函数不可用时的占位。"""
        raise RuntimeError("LightRAG embedding function unavailable")


__all__ = [
    "is_lightrag_available",
    "_llm_model_func",
    "_embedding_func",
    "take_llm_errors",
    "clear_llm_errors",
    "_is_fatal_llm_error",
    "LIGHTRAG_IMPORT_ERROR",
]