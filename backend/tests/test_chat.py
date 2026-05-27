"""Chat endpoint: auth + course access gate."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_requires_auth(client: AsyncClient):
    r = await client.post(
        "/api/chat",
        json={"course_id": "stamp", "message": "hi", "history": []},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_chat_denies_unenrolled_course(client: AsyncClient, auth_headers: dict):
    r = await client.post(
        "/api/chat",
        headers=auth_headers,
        json={"course_id": "stamp", "message": "hi", "history": []},
    )
    assert r.status_code == 403
