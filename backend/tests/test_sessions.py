"""Session API: course access + ownership isolation."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_session_requires_course_access(client: AsyncClient, auth_headers: dict):
    r = await client.post(
        "/api/sessions",
        headers=auth_headers,
        json={"course_id": "nonexistent_course", "title": "t"},
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_session_ownership_isolation(client: AsyncClient):
    u1 = f"sess_u1_{__import__('os').urandom(3).hex()}"
    u2 = f"sess_u2_{__import__('os').urandom(3).hex()}"

    r1 = await client.post(
        "/api/auth/register",
        json={"username": u1, "password": "pass1234"},
    )
    r2 = await client.post(
        "/api/auth/register",
        json={"username": u2, "password": "pass1234"},
    )
    h1 = {"Authorization": f"Bearer {r1.json()['token']}"}
    h2 = {"Authorization": f"Bearer {r2.json()['token']}"}

    # Admin can create session on builtin course (seeded stamp/circuit)
    admin_reg = await client.post(
        "/api/auth/register",
        json={"username": "admin", "password": "adminpass123"},
    )
    if admin_reg.status_code == 409:
        admin_login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "adminpass123"},
        )
        admin_token = admin_login.json()["token"]
    else:
        admin_token = admin_reg.json()["token"]
    admin_h = {"Authorization": f"Bearer {admin_token}"}

    created = await client.post(
        "/api/sessions",
        headers=admin_h,
        json={"course_id": "stamp", "title": "admin session"},
    )
    assert created.status_code == 200
    session_id = created.json()["id"]

    forbidden = await client.get(f"/api/sessions/{session_id}", headers=h1)
    assert forbidden.status_code == 403
