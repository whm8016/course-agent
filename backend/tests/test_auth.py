"""Auth API integration tests."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_login_me(client: AsyncClient):
    username = "auth_user_1"
    reg = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "pass1234", "display_name": "U1"},
    )
    assert reg.status_code == 200
    assert "token" in reg.json()

    login = await client.post(
        "/api/auth/login",
        json={"username": username, "password": "pass1234"},
    )
    assert login.status_code == 200
    token = login.json()["token"]

    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["user"]["username"] == username


@pytest.mark.asyncio
async def test_me_requires_auth(client: AsyncClient):
    r = await client.get("/api/auth/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    username = "auth_user_2"
    await client.post(
        "/api/auth/register",
        json={"username": username, "password": "pass1234"},
    )
    r = await client.post(
        "/api/auth/login",
        json={"username": username, "password": "wrong"},
    )
    assert r.status_code == 401
