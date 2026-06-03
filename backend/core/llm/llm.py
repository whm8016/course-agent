from __future__ import annotations

import base64
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Callable

from openai import AsyncOpenAI

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    LLM_TIMEOUT_SEC,
    TEXT_MODEL,
    VISION_MODEL,
)
from core.llm.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    LLMRetryError,
    RetryConfig,
    get_llm_circuit_breaker,
    with_retry_and_circuit,
)

logger = logging.getLogger(__name__)

# ============================================================
# LLM 客户端配置与初始化
# ============================================================

_client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
    timeout=LLM_TIMEOUT_SEC,
)

# LangSmith tracing 配置
_tracing_flag = os.getenv("LANGSMITH_TRACING", "").strip().lower()
_has_ls_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
if _tracing_flag in ("1", "true", "yes") and _has_ls_key:
    try:
        from langsmith import wrappers

        _client = wrappers.wrap_openai(_client, chat_name="course_agent_chat")
        logger.info(
            "LangSmith: OpenAI client wrapped; runs go to project=%r",
            os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "(default)",
        )
    except Exception:
        logger.exception("LangSmith wrap_openai failed; using raw AsyncOpenAI client")
else:
    logger.info(
        "LangSmith: disabled (LANGSMITH_TRACING=%r, has_api_key=%s). "
        "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in backend/.env, restart backend.",
        os.getenv("LANGSMITH_TRACING", ""),
        _has_ls_key,
    )

client = _client

# ============================================================
# 熔断器配置
# ============================================================

# 获取全局 LLM 熔断器
_llm_circuit_breaker = get_llm_circuit_breaker("default")

# 重试配置
_retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
)


def _image_to_data_url(image_path: str) -> str:
    """将图片转换为 data URL"""
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
    """构建消息列表"""
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


# ============================================================
# 带重试和熔断的 LLM 调用
# ============================================================

async def _make_chat_completion(
    model: str,
    messages: list[dict],
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> dict:
    """
    执行 LLM 调用（带熔断和重试）

    原理：
    1. 通过熔断器执行调用
    2. 如果熔断器拒绝（OPEN 状态），抛出 CircuitOpenError
    3. 如果调用失败，根据 RetryConfig 重试
    4. 使用指数退避策略避免 API 限流
    """
    async def _call():
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return await with_retry_and_circuit(
        _call,
        retry_config=_retry_config,
        circuit_breaker=_llm_circuit_breaker,
    )


async def chat_stream(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    image_path: str | None = None,
    use_reliability: bool = True,
) -> AsyncGenerator[str, None]:
    """
    流式 LLM 对话（带可靠性增强）

    参数：
        system_prompt: 系统提示词
        history: 对话历史
        user_message: 用户消息
        image_path: 图片路径（可选）
        use_reliability: 是否使用重试和熔断机制

    原理：
    - 构造消息后调用 LLM
    - 使用流式响应逐字返回
    - 失败时使用指数退避重试
    """
    model = VISION_MODEL if image_path else TEXT_MODEL
    messages = _build_messages(system_prompt, history, user_message, image_path)

    logger.info(
        "LLM stream start model=%s msg_count=%d has_image=%s user_msg=「%s」",
        model, len(messages), bool(image_path), user_message[:80],
    )

    try:
        if use_reliability:
            stream = await _make_chat_completion(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=8192,
            )
        else:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=8192,
            )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except CircuitOpenError as e:
        logger.error(f"LLM circuit breaker OPEN: {e}")
        yield "⚠️ AI 服务暂时不可用（服务繁忙），请稍后重试。"
    except LLMRetryError as e:
        logger.error(f"LLM retry failed after all retries: {e}")
        yield "⚠️ AI 服务暂时不可用，请稍后重试。"
    except Exception as e:
        logger.exception("LLM stream error")
        yield "⚠️ AI 服务发生错误，请稍后重试。"


async def chat_complete(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """
    非流式 LLM 调用（带可靠性增强）

    参数：
        system_prompt: 系统提示词
        history: 对话历史
        user_message: 用户消息
        model: 模型名称（可选，默认使用 TEXT_MODEL）
        temperature: 温度参数
        max_tokens: 最大 token 数

    返回：
        LLM 生成的完整回复
    """
    model = model or TEXT_MODEL
    messages = _build_messages(system_prompt, history, user_message)

    try:
        response = await _make_chat_completion(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    except CircuitOpenError as e:
        logger.error(f"LLM circuit breaker OPEN: {e}")
        raise RuntimeError("AI 服务暂时不可用，请稍后重试。") from e
    except LLMRetryError as e:
        logger.error(f"LLM retry failed: {e}")
        raise RuntimeError("AI 服务暂时不可用，请稍后重试。") from e


# ============================================================
# 熔断器状态查询
# ============================================================

def get_llm_circuit_state() -> str:
    """获取 LLM 熔断器当前状态"""
    return _llm_circuit_breaker.get_state().value


def reset_llm_circuit_breaker():
    """重置 LLM 熔断器（用于运维操作）"""
    _llm_circuit_breaker.reset()
    logger.info("LLM circuit breaker reset")
