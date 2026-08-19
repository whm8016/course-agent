"""语音转写端点（B 线）：浏览器录音 multipart → ASR → 文字回填输入框。不落盘。

前端 VoiceInput（MediaRecorder webm/opus，iOS 为 mp4/m4a）→ 本端点转写 →
文字返回，由前端塞输入框（不自动发送）。未配置 ASR → 503（前端首探后隐藏
麦克风入口，优雅降级）。

MIME 白名单显式映射，未知 MIME → 400 拒绝——绝不带着错误扩展名静默进 SDK
（upload.py `_safe_ext` 未知扩展回退 .png 的坑，同类问题不二犯）。
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.requests import Request

from api.auth import get_current_user
from core.db.limiter import limiter
from core.llm.asr import TranscribeError, resolve_asr_runtime, transcribe_audio
from settings import get_settings

router = APIRouter(tags=["transcribe"])

# MIME → 扩展名（显式映射）。video/webm 必须放行：MediaRecorder opus 在 Chrome
# 常报 video/webm 而非 audio/webm。
_MIME_EXT = {
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
}


def _max_audio_bytes() -> int:
    return get_settings().asr.max_audio_mb * 1024 * 1024


class TranscribeResponse(BaseModel):
    text: str
    model: str
    duration_ms: int


@router.post("/transcribe", response_model=TranscribeResponse)
@limiter.limit("10/minute")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """转写一段录音。限流 10/min（per-user，TESTING 环境关闭）。"""
    mime = (file.content_type or "").strip().lower()
    ext = _MIME_EXT.get(mime)
    if ext is None:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式：{mime or '未知'}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="音频内容为空")
    max_bytes = _max_audio_bytes()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"音频超过 {max_bytes // (1024 * 1024)}MB 上限",
        )

    settings = get_settings()
    client, model = await resolve_asr_runtime(str(user["id"]))
    if client is None or not model:
        # 未配置 ASR：503 让前端隐藏入口（麦克风按钮消失，其余功能不受影响）
        raise HTTPException(status_code=503, detail="未配置语音识别模型")

    started = time.monotonic()
    try:
        text = await transcribe_audio(
            client,
            model,
            data,
            f"speech.{ext}",
            mime,
            language=(settings.asr.language or "").strip() or None,
            track={"user_id": str(user["id"])},
        )
    except TranscribeError as exc:
        raise HTTPException(status_code=502, detail=f"语音识别失败：{exc}") from exc
    return TranscribeResponse(
        text=text,
        model=model,
        duration_ms=int((time.monotonic() - started) * 1000),
    )
