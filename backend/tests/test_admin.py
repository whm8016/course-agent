"""Admin route authorization."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_student_cannot_access_admin(client: AsyncClient, auth_headers: dict):
    r = await client.get("/api/admin/info", headers=auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_access_admin_info(client: AsyncClient, admin_headers: dict):
    r = await client.get("/api/admin/info", headers=admin_headers)
    assert r.status_code == 200
