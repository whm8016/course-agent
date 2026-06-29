"""Chat API happy-path tests: mock TurnRuntimeManager, verify SSE response."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


async def _empty_subscribe(turn_id, after_seq=0):
    """Async generator that yields nothing (turn finishes immediately)."""
    return
    yield  # make it an async generator


def _make_mock_trm():
    """Build a TurnRuntimeManager mock that starts/subscribes/cancels without side effects."""
    trm = MagicMock()
    trm.start_turn = AsyncMock(return_value="test-turn-id")
    trm.subscribe_turn = _empty_subscribe
    trm.cancel_turn = AsyncMock()
    return trm


@pytest.mark.asyncio
async def test_chat_sse_ok(client: AsyncClient, admin_headers: dict):
    """已登录且有课程权限（admin 绕过），POST /api/chat 返回 200 + text/event-stream。"""
    with patch("api.chat.get_turn_runtime_manager", return_value=_make_mock_trm()):
        r = await client.post(
            "/api/chat",
            headers=admin_headers,
            json={"course_id": "any", "message": "hello"},
        )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_chat_long_message_truncated(client: AsyncClient, admin_headers: dict):
    """超过 2000 字符的消息会被截断，不返回 4xx。"""
    long_msg = "x" * 3000
    with patch("api.chat.get_turn_runtime_manager", return_value=_make_mock_trm()):
        r = await client.post(
            "/api/chat",
            headers=admin_headers,
            json={"course_id": "any", "message": long_msg},
        )
    assert r.status_code == 200
