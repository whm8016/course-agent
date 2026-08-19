"""B2 语音转写测试：asr 调用层 + 转写端点（MIME 白名单/大小/未配置/鉴权）。

Mock 边界：ASR provider 一律 mock（fake client 的 audio.transcriptions.create）；
端点不落盘 uploads（转写完即弃）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── transcribe_audio（调用层） ───────────────────────────────────────────────


class _FakeTranscription:
    def __init__(self, text: str):
        self.text = text


def _fake_client(text: str = "你好") -> MagicMock:
    c = MagicMock(name="asr_client")
    c.audio.transcriptions.create = AsyncMock(return_value=_FakeTranscription(text))
    return c


async def test_transcribe_audio_returns_text():
    from core.llm.asr import transcribe_audio

    client = _fake_client("光合作用是……")
    text = await transcribe_audio(
        client, "paraformer-v2", b"fake-bytes", "speech.webm", "audio/webm",
        language="zh", track={"user_id": "u1"},
    )
    assert text == "光合作用是……"
    # 文件以 (filename, BytesIO, mime) 元组传给 SDK（免临时文件）
    sent = client.audio.transcriptions.create.await_args.kwargs.get("file")
    assert sent[0] == "speech.webm"
    assert sent[2] == "audio/webm"
    assert client.audio.transcriptions.create.await_args.kwargs.get("model") == "paraformer-v2"


async def test_transcribe_audio_empty_text_raises():
    from core.llm.asr import TranscribeError, transcribe_audio

    with pytest.raises(TranscribeError):
        await transcribe_audio(_fake_client(""), "m", b"x", "s.webm", "audio/webm")


async def test_transcribe_audio_provider_failure_raises():
    from core.llm.asr import TranscribeError, transcribe_audio

    client = _fake_client()
    client.audio.transcriptions.create = AsyncMock(side_effect=RuntimeError("provider down"))
    # with_retry_and_circuit 对意外异常透传（非 LLMRetryError），TranscribeError 兜住
    with pytest.raises(TranscribeError):
        await transcribe_audio(client, "m", b"x", "s.webm", "audio/webm")


# ── resolve_asr_runtime（三级解析） ─────────────────────────────────────────


async def test_resolve_asr_unconfigured_returns_none():
    from core.llm.asr import resolve_asr_runtime

    client, model = await resolve_asr_runtime()
    assert client is None and model is None  # 默认 settings.asr.model="" = 功能关


async def test_resolve_asr_falls_back_to_embedding_creds(monkeypatch):
    """api_key/base_url 空 → 回退 embedding 凭证（dashscope 一把钥匙通吃）。"""
    from types import SimpleNamespace

    from core.llm import asr as asr_mod

    fake_settings = SimpleNamespace(
        asr=SimpleNamespace(
            model="paraformer-v2", language="zh", timeout_sec=60, max_audio_mb=25,
            api_key=type("S", (), {"get_secret_value": lambda self: ""})(),
            base_url="",
        ),
        embedding=SimpleNamespace(
            model="text-embedding-v3",
            api_key=type("S", (), {"get_secret_value": lambda self: "sk-emb"})(),
            base_url="https://dashscope.example/compatible-mode/v1",
        ),
    )
    monkeypatch.setattr(asr_mod, "get_settings", lambda: fake_settings)
    with patch("openai.AsyncOpenAI") as mock_openai:
        client, model = await asr_mod.resolve_asr_runtime()
    mock_openai.assert_called_once_with(
        api_key="sk-emb",
        base_url="https://dashscope.example/compatible-mode/v1",
        timeout=60,
    )
    assert model == "paraformer-v2"


# ── POST /api/transcribe（端点） ─────────────────────────────────────────────


async def test_endpoint_unconfigured_503(client, auth_headers):
    r = await client.post(
        "/api/transcribe",
        files={"file": ("s.webm", b"x", "audio/webm")},
        headers=auth_headers,
    )
    assert r.status_code == 503  # 默认测试环境未配 ASR


async def test_endpoint_happy_path(client, auth_headers):
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("今天讲什么"), "paraformer-v2"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"fake-audio", "audio/webm")},
            headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "今天讲什么"
    assert body["model"] == "paraformer-v2"
    assert "duration_ms" in body


async def test_endpoint_video_webm_allowed(client, auth_headers):
    """MediaRecorder opus 在 Chrome 常报 video/webm——必须放行。"""
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("ok"), "m"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"x", "video/webm")},
            headers=auth_headers,
        )
    assert r.status_code == 200


async def test_endpoint_mime_with_codec_param_allowed(client, auth_headers):
    """带参数的 content-type（audio/webm;codecs=opus，浏览器 File 的真实报法）
    归一化后放行——精确匹配裸键会把合法录音拒成 400（复审 P0-2）。"""
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("ok"), "m"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("speech.webm", b"x", "audio/webm;codecs=opus")},
            headers=auth_headers,
        )
    assert r.status_code == 200


async def test_endpoint_unknown_mime_400_no_fallback(client, auth_headers):
    """未知 MIME 显式拒绝——绝不带着错误扩展名静默进 SDK（_safe_ext 坑的教训）。"""
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("x"), "m"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.txt", b"x", "text/plain")},
            headers=auth_headers,
        )
    assert r.status_code == 400


async def test_endpoint_empty_file_400(client, auth_headers):
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("x"), "m"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"", "audio/webm")},
            headers=auth_headers,
        )
    assert r.status_code == 400


async def test_endpoint_oversize_413(client, auth_headers, monkeypatch):
    import api.transcribe as t

    monkeypatch.setattr(t, "_max_audio_bytes", lambda: 4)  # 4 字节上限，发 10 字节
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("x"), "m"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"0123456789", "audio/webm")},
            headers=auth_headers,
        )
    assert r.status_code == 413


async def test_endpoint_provider_failure_502(client, auth_headers):
    from core.llm.asr import TranscribeError

    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("x"), "m"))), \
         patch("api.transcribe.transcribe_audio",
               AsyncMock(side_effect=TranscribeError("provider down"))):
        r = await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"x", "audio/webm")},
            headers=auth_headers,
        )
    assert r.status_code == 502


async def test_endpoint_requires_auth(client):
    r = await client.post(
        "/api/transcribe",
        files={"file": ("s.webm", b"x", "audio/webm")},
    )
    assert r.status_code == 401


async def test_endpoint_no_persistence(client, auth_headers, tmp_path):
    """转写不落盘 uploads——纯字节流过境（LibreChat 临时文件 finally-unlink 的更简版）。"""
    from settings import get_settings

    upload_dir = get_settings().paths.upload_dir
    with patch("api.transcribe.resolve_asr_runtime",
               AsyncMock(return_value=(_fake_client("x"), "m"))):
        await client.post(
            "/api/transcribe",
            files={"file": ("s.webm", b"x", "audio/webm")},
            headers=auth_headers,
        )
    import os
    if upload_dir and os.path.isdir(upload_dir):
        leftovers = [f for f in os.listdir(upload_dir) if f.endswith((".webm", ".mp3", ".m4a"))]
        assert leftovers == []
