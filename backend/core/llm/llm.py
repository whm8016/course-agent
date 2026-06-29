from __future__ import annotations

import base64
import logging
import os
from collections.abc import AsyncGenerator
from pathlib import Path

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    FALLBACK_API_KEY,
    FALLBACK_BASE_URL,
    FALLBACK_MODEL,
    LLM_API_VERSION,
    LLM_BINDING,
    LLM_TIMEOUT_SEC,
    TEXT_MODEL,
)
from core.llm.provider_factory import get_llm_client
from core.llm.reliability import (
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

_client = get_llm_client(
    binding=LLM_BINDING,
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL or None,
    api_version=LLM_API_VERSION or None,
    model=TEXT_MODEL,
    timeout=LLM_TIMEOUT_SEC,
)

# LangSmith tracing：统一经 wrap_openai_client（对 AsyncOpenAI/AsyncAzureOpenAI 生效，
# Anthropic 适配器自动跳过）。fallback/profile client 也走同一函数（见下方 / provider_factory）。
from core.observability.langsmith_trace import is_tracing_enabled, wrap_openai_client

from openai import AsyncOpenAI as _AsyncOpenAI  # noqa: E402

_was_openai = isinstance(_client, _AsyncOpenAI)
_client = wrap_openai_client(_client, chat_name="course_agent_chat")
if is_tracing_enabled():
    if _was_openai:
        logger.info(
            "LangSmith: OpenAI client wrapped; runs go to project=%r",
            os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "(default)",
        )
    else:
        logger.info(
            "LangSmith: skipped (binding=%r is not AsyncOpenAI, wrapping not supported)",
            LLM_BINDING,
        )
else:
    logger.info(
        "LangSmith: disabled (LANGSMITH_TRACING=%r, has_api_key=%s). "
        "Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in backend/.env, restart backend.",
        os.getenv("LANGSMITH_TRACING", ""),
        bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
    )

client = _client

# Fallback LLM 客户端（主模型熔断时兜底；走 DashScope 等独立端点，binding 勿跟主 LLM 混用）
_fallback_client: object | None = None
if FALLBACK_API_KEY:
    _fallback_binding = "dashscope"
    if FALLBACK_BASE_URL and "deepseek" in FALLBACK_BASE_URL.lower():
        _fallback_binding = "deepseek"
    _fallback_client = get_llm_client(
        binding=_fallback_binding,
        api_key=FALLBACK_API_KEY,
        base_url=FALLBACK_BASE_URL or None,
        api_version=LLM_API_VERSION or None,
        model=FALLBACK_MODEL,
        timeout=LLM_TIMEOUT_SEC,
    )
    # fallback client 也接入 LangSmith wrap（主模型熔断兜底调用同样上 trace）
    _fallback_client = wrap_openai_client(_fallback_client, chat_name="course_agent_fallback")
    logger.info(
        "Fallback LLM client initialized (binding=%s model=%s base=%s)",
        _fallback_binding, FALLBACK_MODEL, FALLBACK_BASE_URL,
    )

# ============================================================
# 熔断器配置
# ============================================================

# 获取全局 LLM 熔断器
_llm_circuit_breaker = get_llm_circuit_breaker("default")

# 重试配置（从 settings 读取；settings 不可用时回退默认值，避免循环导入阻断启动）
try:
    from settings.base import get_settings as _get_settings

    _s = _get_settings()
    _retry_config = RetryConfig(
        max_retries=_s.llm_retry_max,
        base_delay=_s.llm_retry_base_delay,
        max_delay=_s.llm_retry_max_delay,
        exponential_base=_s.llm_retry_exponential_base,
    )
    del _s
except Exception:  # pragma: no cover - 启动期 settings 缺失的兜底
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
) -> list[dict]:
    """构建纯文本消息列表（system + history + user）。

    图片注入由调用方经 prepare_multimodal_messages 完成（对标 DeepTutor 两步式）。
    """
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


# ============================================================
# 带重试和熔断的 LLM 调用
# ============================================================

async def _create_with_image_fallback(
    llm_client: object,
    create_kwargs: dict,
    binding: str,
    model: str,
):
    """单次 create 调用，失败时若模型不支持图片则剥图重试（Stage-2 降级）。

    对标 DeepTutor agent_loop._safe_create 的 image fallback 分支：模型不支持
    vision（异常命中 image/vision/multimodal 等关键词）时，剥掉图片用**同一模型**
    重试纯文本。剥图 inplace 改 create_kwargs['messages']，后续 retry 也用纯文本。
    """
    from core.llm.multimodal import (
        is_image_input_unsupported,
        should_degrade_to_text,
        strip_image_parts_inplace,
    )
    try:
        return await llm_client.chat.completions.create(**create_kwargs)
    except Exception as exc:
        msgs = create_kwargs.get("messages") or []
        if is_image_input_unsupported(exc) and should_degrade_to_text(binding, model, msgs):
            strip_image_parts_inplace(create_kwargs["messages"])
            logger.warning("Stage-2 降级：模型 %s 不支持图片输入，剥图后用同模型重试纯文本", model)
            return await llm_client.chat.completions.create(**create_kwargs)
        raise


async def _make_chat_completion(
    model: str,
    messages: list[dict],
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    binding: str = LLM_BINDING,
) -> dict:
    """
    执行 LLM 调用（带熔断和重试 + Stage-2 图片降级）

    原理：
    1. 通过熔断器执行调用
    2. 如果熔断器拒绝（OPEN 状态），抛出 CircuitOpenError
    3. 如果调用失败，根据 RetryConfig 重试
    4. 单次失败若因模型不支持图片 → 剥图重试（降级在 _call 内，retry 复用已剥图的 messages）
    5. 使用指数退避策略避免 API 限流
    """
    async def _call():
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return await _create_with_image_fallback(client, kwargs, binding, model)

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
    attachments: list | None = None,
    use_reliability: bool = True,
    model_override: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    流式 LLM 对话（带可靠性增强）

    参数：
        system_prompt: 系统提示词
        history: 对话历史
        user_message: 用户消息
        image_path: 图片路径（可选，向后兼容旧单图入口）
        attachments: 附件列表（Attachment，支持多图；优先于 image_path）
        use_reliability: 是否使用重试和熔断机制
        model_override: 指定模型（不传则用 TEXT_MODEL，不因有图自动切换 vision 模型）

    原理：
    - 构造纯文本消息后，经 prepare_multimodal_messages 注入图片
    - 使用流式响应逐字返回
    - 失败时使用指数退避重试
    """
    from core.attachment import from_image_path
    from core.llm.multimodal import prepare_multimodal_messages

    # 合并附件来源：attachments 列表优先，回退旧 image_path 单图
    all_attachments = list(attachments or [])
    if image_path and not any(a.is_image() for a in all_attachments):
        all_attachments.append(from_image_path(image_path))
    has_image = any(a.is_image() for a in all_attachments)

    # 始终用 chat 主模型（TEXT_MODEL）——对标 DeepTutor，不因有图硬切 VISION_MODEL。
    # 图片乐观注入；模型若不支持 vision，由 create 层 Stage-2 降级剥图重试。
    model = model_override or TEXT_MODEL
    messages = _build_messages(system_prompt, history, user_message)
    if all_attachments:
        prepare_multimodal_messages(
            messages, all_attachments, LLM_BINDING,
            fallback_text=user_message or "请描述这张图片",
        )

    logger.info(
        "LLM stream start model=%s msg_count=%d has_image=%s user_msg=「%s」",
        model, len(messages), has_image, user_message[:80],
    )

    try:
        if use_reliability:
            stream = await _make_chat_completion(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=8192,
                binding=LLM_BINDING,
            )
        else:
            stream = await _create_with_image_fallback(
                client,
                {
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 8192,
                },
                LLM_BINDING,
                model,
            )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    except (CircuitOpenError, LLMRetryError) as e:
        logger.warning(f"Primary LLM failed: {e}, trying fallback...")
        if _fallback_client:
            try:
                fb_model = FALLBACK_MODEL
                fb_stream = await _fallback_client.chat.completions.create(
                    model=fb_model,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=8192,
                )
                async for chunk in fb_stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content
                return
            except Exception as fb_e:
                logger.error(f"Fallback LLM also failed: {fb_e}")
        yield "⚠️ AI 服务暂时不可用（服务繁忙），请稍后重试。"
    except Exception:
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

    except (CircuitOpenError, LLMRetryError) as e:
        logger.warning(f"Primary LLM failed: {e}, trying fallback...")
        if _fallback_client:
            try:
                fb_resp = await _fallback_client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return fb_resp.choices[0].message.content or ""
            except Exception as fb_e:
                logger.error(f"Fallback LLM also failed: {fb_e}")
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
