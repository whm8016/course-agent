"""语音转写（ASR）调用层：settings.asr 解析 + 独立熔断 + 用量埋点（B 线语音输入）。

对标 LibreChat STTService 的 provider 分发形状；调用点是本机 venv openai SDK 2.44
的 ``client.audio.transcriptions.create``（openai 兼容通道，paraformer/whisper 系）。
文件以 (filename, BytesIO, mime) 元组直传 SDK——免临时文件免 finally-unlink
（LibreChat 走临时文件的更简替代）。

计费口径：whisper/paraformer 按音频分钟计费，model_pricing.json 的 per-1M-token
结构装不下——本期只 track 明细（mode="asr"，有 usage 的 provider 自动落账），
按分钟估成本留 TODO。
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from settings import get_settings

logger = logging.getLogger(__name__)


class TranscribeError(Exception):
    """转写失败（provider 异常 / 熔断 / 空文本）。端点转 502。"""


async def resolve_asr_runtime(user_id: str = "") -> tuple[Any, str | None]:
    """解析 (asr_client, model)。model 空 = 功能关闭 → (None, None)（端点 503）。

    凭证回退链：asr.api_key/base_url 空 → embedding 凭证——dashscope 一把钥匙
    通吃 embedding+ASR（vision 同款设计动机；vision 实际未做回退，这里真做）。
    用户自配（asr_*四字段）留二期（B1c）。
    """
    settings = get_settings()
    model = (settings.asr.model or "").strip()
    if not model:
        return None, None
    key = (
        settings.asr.api_key.get_secret_value()
        or settings.embedding.api_key.get_secret_value()
    )
    base_url = (settings.asr.base_url or "").strip() or (
        settings.embedding.base_url or ""
    ).strip()
    if not key or not base_url:
        logger.warning(
            "[asr] 未配置凭证（ASR__API_KEY/ASR__BASE_URL，或 embedding 回退），功能不可用"
        )
        return None, None
    try:
        from openai import AsyncOpenAI

        return (
            # max_retries=0：重试收归 reliability.with_retry_and_circuit 单层，
            # 避免 SDK 内置重试与外层重试相乘（最坏 4 波×3 次三层叠加）
            AsyncOpenAI(
                api_key=key,
                base_url=base_url,
                timeout=settings.asr.timeout_sec,
                max_retries=0,
            ),
            model,
        )
    except Exception:
        logger.exception("[asr] 构造 ASR client 失败")
        return None, None


async def transcribe_audio(
    client: Any,
    model: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    language: str | None = None,
    track: dict[str, str] | None = None,
) -> str:
    """转写一段音频字节，返回文本。失败抛 TranscribeError（调用方按 502 处理）。

    Args:
        track: 可选 usage 埋点维度（user_id/course_id/session_id），传入则 LLM 用量
            落 Prometheus + 明细（mode 固定 asr）；不传零变化。
    """
    from core.llm.reliability import (
        CircuitOpenError,
        LLMRetryError,
        RetryConfig,
        get_llm_circuit_breaker,
        with_retry_and_circuit,
    )

    try:
        # ASR 专属熔断器（按 model 注册，与对话主链路各 binding 隔离——M-21 同款：
        # 辅助链路失败不打开对话熔断器；并入全局注册表 → 统一吃 settings、
        # /health/detailed 可见、运维 reset 覆盖）
        resp = await with_retry_and_circuit(
            client.audio.transcriptions.create,
            model=model,
            file=(filename, BytesIO(file_bytes), mime_type),
            language=language or None,
            retry_config=RetryConfig(),
            circuit_breaker=get_llm_circuit_breaker(f"asr:{model}"),
        )
    except (CircuitOpenError, LLMRetryError) as exc:
        raise TranscribeError(f"ASR provider 失败（熔断/重试耗尽）：{exc}") from exc
    except Exception as exc:
        raise TranscribeError(f"ASR 调用异常：{exc}") from exc

    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise TranscribeError("转写结果为空（音频无声或格式无法解码）")

    if track is not None:
        try:
            from core.analytics.token_usage import track_llm_response

            await track_llm_response(resp, model=model, mode="asr", **track)
        except Exception:
            # best-effort：无 usage 的转写响应会被静默跳过，埋点失败不影响转写
            logger.debug("[asr] usage 埋点失败（best-effort）", exc_info=True)
    return text


__all__ = ["TranscribeError", "resolve_asr_runtime", "transcribe_audio"]
