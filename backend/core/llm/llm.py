from __future__ import annotations

import base64
import logging
import os
import random
from pathlib import Path

from settings import get_settings
DASHSCOPE_API_KEY = get_settings().llm.api_key.get_secret_value()
DASHSCOPE_BASE_URL = get_settings().llm.base_url
FALLBACK_API_KEY = get_settings().fallback.api_key.get_secret_value()
FALLBACK_BASE_URL = get_settings().fallback.base_url
FALLBACK_MODEL = get_settings().fallback.model
LLM_API_VERSION = get_settings().llm.api_version
LLM_BINDING = get_settings().llm.binding
LLM_TIMEOUT_SEC = get_settings().llm.timeout_sec
TEXT_MODEL = get_settings().llm.text_model
from core.llm.provider_factory import get_llm_client
from core.llm.reliability import (
    CircuitBreaker,
    CircuitOpenError,
    LLMRetryError,
    RetryConfig,
    get_llm_circuit_breaker,
    with_retry_and_circuit,
)
from core.llm.loadtest_mock import maybe_loadtest_mock_stream

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

# "default" 熔断器：仅 binding 解析失败 / 未走 profile 的兜底调用使用。主链路按 binding
# （dashscope/openai/...）各自注册独立实例（见 _create_with_image_fallback）。保留此全局
# 引用供 get_llm_circuit_state/reset_llm_circuit_breaker 默认行为向后兼容。
_llm_circuit_breaker = get_llm_circuit_breaker("default")

# 重试配置（从 settings.llm 读取；settings 模块缺失时回退默认值，避免循环导入阻断启动）。
# ⚠️ 路径必须是 _s.llm.retry_max（嵌套），不是旧扁平 _s.llm_retry_max —— config 嵌套重构后
# 扁平名已不存在，读它会 AttributeError。曾因下方 except Exception 把这个错误静默吞掉，
# 导致 retry 配置永远走兜底默认值、面板配置不生效。except 已收窄为 ImportError（仅兜底
# settings 模块缺失），配置路径错误会直接抛出暴露问题。
try:
    from settings.base import get_settings as _get_settings

    _llm_cfg = _get_settings().llm
    _retry_config = RetryConfig(
        max_retries=_llm_cfg.retry_max,
        base_delay=_llm_cfg.retry_base_delay,
        max_delay=_llm_cfg.retry_max_delay,
        exponential_base=_llm_cfg.retry_exponential_base,
    )
    del _llm_cfg
except ImportError:  # pragma: no cover - 仅 settings 模块本身缺失（循环导入兜底）
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

    图片注入由调用方经 prepare_multimodal_messages 完成。
    """
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def _is_stream_options_unsupported(exc: Exception) -> bool:
    """识别「端点不支持 stream_options」类错误（用于成本采集降级判断）。

    部分 OpenAI 兼容端点（老版本 DashScope/某些私有部署）会因 stream_options 报 400
    而非静默忽略。此时降级：剥掉 stream_options 用同模型重试一次（丢 usage 但保 turn），
    与 Stage-2 剥图降级同套路。匹配信息关键字，兼容各家错误文案差异。
    """
    msg = str(exc).lower()
    return "stream_options" in msg or "include_usage" in msg


# ============================================================
# 带重试和熔断的 LLM 调用
# ============================================================

async def _create_with_image_fallback(
    llm_client: object,
    create_kwargs: dict,
    binding: str,
    model: str,
    circuit_breaker: CircuitBreaker | None = _llm_circuit_breaker,
):
    """带 retry+熔断的 create 调用；模型拒图时在闭包内剥图重试（Stage-2 降级）。

    reliability（指数退避重试 + 熔断器）下沉到此——主 Agent 路径（loop._one_round）与
    chat_complete 经此统一获得保护。

    image fallback 放进 with_retry_and_circuit 的 _call 内部：剥图重试与首次调用对熔断器
    原子（成功算 1 次 success，failure_count 不增），避免"模型不支持图片"这类确定性业务
    错误污染服务可用性熔断计数，HALF_OPEN 探测期也能正常剥图。模型不支持 vision（异常命中
    image/vision/multimodal 等关键词）时，剥掉图片用同一模型重试纯文本。

    LLM fallback（M-17 + M-18）：主路径重试耗尽 / 熔断 OPEN 后，若配置了 fallback client
    且当前是全局默认路径（circuit_breaker 非 None），切到 fallback 端点再用 reliability
    跑一次（不再裸 create）。此前 fallback 只挂在 chat_complete，生产对话走 run_agent_loop
    时主模型抖动直接抛错给用户；现在 loop 主路径也能兜底。自配路径（circuit_breaker=None）
    不兜底，符合「自配是用户私有资源」语义（H-16）。
    """
    from core.llm.multimodal import (
        is_image_input_unsupported,
        should_degrade_to_text,
        strip_image_parts_inplace,
    )

    def _make_call(client: object, kwargs: dict, bnd: str, mdl: str):
        async def _call_with_image_fallback():
            # 压测概率分流（LOAD_TEST_MOCK_LLM=1）：stream 请求按 LOAD_TEST_REAL_RATIO 概率
            # 真打、其余走假流式 mock。env 不设时整段跳过，行为与生产逐字节一致。mock 不抛
            # 异常 → reliability 计 success；FORCE_FAIL 走 mock 内部分支触发熔断复验（H-11）。
            if os.getenv("LOAD_TEST_MOCK_LLM") == "1" and kwargs.get("stream"):
                _real_ratio = float(os.getenv("LOAD_TEST_REAL_RATIO", "0.15") or "0.15")
                if random.random() >= _real_ratio:
                    _ttft = float(os.getenv("LOAD_TEST_MOCK_TTFT_MS", "600")) / 1000
                    _total = int(os.getenv("LOAD_TEST_MOCK_TOTAL_CHARS", "240"))
                    return await maybe_loadtest_mock_stream(_ttft, _total)
            try:
                return await client.chat.completions.create(**kwargs)
            except Exception as exc:
                msgs = kwargs.get("messages") or []
                if is_image_input_unsupported(exc) and should_degrade_to_text(bnd, mdl, msgs):
                    strip_image_parts_inplace(kwargs["messages"])
                    logger.warning(
                        "Stage-2 降级：模型 %s 不支持图片输入，剥图后用同模型重试纯文本", mdl
                    )
                    return await client.chat.completions.create(**kwargs)
                # 成本采集降级：端点不支持 stream_options（报 400 而非静默忽略）→ 剥掉重试一次。
                # 丢本轮 usage（cost 可观测性降级），但保 turn 不挂——比盲传 stream_options 打断
                # 整轮对话安全。发生在 create 层、reliability 之前，成功算 1 次 success 不污染熔断。
                if _is_stream_options_unsupported(exc) and "stream_options" in kwargs:
                    kwargs.pop("stream_options", None)
                    logger.info(
                        "成本采集降级：%s 端点不支持 stream_options，已剥离重试（本轮无 usage）", mdl
                    )
                    return await client.chat.completions.create(**kwargs)
                raise
        return _call_with_image_fallback

    # 熔断器策略：circuit_breaker=None 时不熔断（with_retry_and_circuit 仍重试，只是不计
    # failure）。默认 _llm_circuit_breaker（全局 "default"）保护平台共享的 DashScope——它
    # 是所有未自配用户共用的下游，挂了要快速失败防雪崩；自配 client 路径由调用方传 None
    # 关闭：自配供应商是用户私有资源，平台不替它兜底，让真实错误冒给用户，也避免自配失败
    # 把全局熔断器打 OPEN 误伤他人（含同 binding 不同 key 的自配用户）。
    try:
        return await with_retry_and_circuit(
            _make_call(llm_client, create_kwargs, binding, model),
            retry_config=_retry_config,
            circuit_breaker=circuit_breaker,
        )
    except (CircuitOpenError, LLMRetryError) as exc:
        # M-17/M-18：主路径彻底失败 → fallback 端点兜底（仅全局默认路径）。
        # fallback 经 with_retry_and_circuit 重跑，瞬时抖动能自愈，不再裸 create。
        if _fallback_client is None or circuit_breaker is None:
            raise  # 无 fallback / 自配路径：原样抛给上层
        logger.warning(
            "Primary LLM path failed (%s: %s); falling back to %s",
            type(exc).__name__, exc, FALLBACK_MODEL,
        )
        fb_kwargs = dict(create_kwargs)
        fb_kwargs["model"] = FALLBACK_MODEL
        try:
            return await with_retry_and_circuit(
                _make_call(_fallback_client, fb_kwargs, _fallback_binding, FALLBACK_MODEL),
                retry_config=_retry_config,
                # fallback 是独立端点，不接入主链路熔断器（避免 fallback 故障连累主路径判定）
                circuit_breaker=None,
            )
        except (CircuitOpenError, LLMRetryError) as fb_exc:
            logger.error("Fallback LLM also failed: %s", fb_exc)
            raise


async def _make_chat_completion(
    model: str,
    messages: list[dict],
    stream: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    binding: str = LLM_BINDING,
) -> dict:
    """执行 LLM 调用（retry+熔断+图片降级已下沉到 _create_with_image_fallback）。

    本函数为 chat_complete 等历史调用方保留签名薄封装；可靠性不再在此层包装，
    否则会与 _create_with_image_fallback 内的 with_retry_and_circuit 双重叠加
    （retry 套 retry、failure 双重计数）。
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return await _create_with_image_fallback(client, kwargs, binding, model)


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
        # _make_chat_completion → _create_with_image_fallback 内部已含主路径 retry+熔断
        # 与 fallback 兜底（M-17/M-18：fallback 下沉到该层，chat_complete 不再重复兜底，
        # 否则双重 fallback 且绕过 reliability）。此处仅在主+fallback 均失败时转友好提示。
        response = await _make_chat_completion(
            model=model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    except (CircuitOpenError, LLMRetryError) as e:
        logger.warning("chat_complete: primary+fallback 均失败: %s", e)
        raise RuntimeError("AI 服务暂时不可用，请稍后重试。") from e


# ============================================================
# 熔断器状态查询
# ============================================================

def get_llm_circuit_state(name: str = "default") -> str:
    """获取指定 binding 的 LLM 熔断器状态（默认 "default"）。

    按 binding 拆分后主链路用供应商名注册（dashscope/openai/...），看全量状态用
    get_all_llm_circuit_states。
    """
    return get_llm_circuit_breaker(name).get_state().value


def get_all_llm_circuit_states() -> dict[str, str]:
    """返回所有已注册 LLM 熔断器的状态快照 {binding: state}（运维健康检查用）。"""
    from core.llm.reliability import all_llm_circuit_states
    return all_llm_circuit_states()


def reset_llm_circuit_breaker(name: str = "default") -> None:
    """重置指定 binding 的 LLM 熔断器（运维操作）。"""
    get_llm_circuit_breaker(name).reset()
    logger.info("LLM circuit breaker reset (name=%s)", name)


def reset_all_llm_circuit_breakers() -> int:
    """重置所有已注册的 LLM 熔断器，返回重置数量（运维一键恢复）。"""
    from core.llm.reliability import reset_all_llm_circuit_breakers as _reset_all
    n = _reset_all()
    logger.info("LLM circuit breakers reset (count=%d)", n)
    return n
