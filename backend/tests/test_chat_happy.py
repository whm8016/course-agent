"""Chat API happy-path tests: mock run_agent_stream, verify SSE response."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import AsyncClient


async def _empty_stream(*args, **kwargs):
    """空异步生成器：模拟 LLM 不返回任何事件。"""
    return
    yield  # 使函数成为 async generator


@pytest.mark.asyncio
async def test_chat_sse_ok(client: AsyncClient, admin_headers: dict):
    """已登录且有课程权限（admin 绕过），POST /api/chat 返回 200 + text/event-stream。"""
    with patch("api.chat.run_agent_stream", _empty_stream):
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
    with patch("api.chat.run_agent_stream", _empty_stream):
        r = await client.post(
            "/api/chat",
            headers=admin_headers,
            json={"course_id": "any", "message": long_msg},
        )
    assert r.status_code == 200
