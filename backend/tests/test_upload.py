"""Upload validation and authenticated file access."""
from __future__ import annotations

import io

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_requires_auth(client: AsyncClient):
    files = {"file": ("x.png", io.BytesIO(b"fake"), "image/png")}
    r = await client.post("/api/upload", files=files)
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_type(client: AsyncClient, auth_headers: dict):
    # upload 现支持图片 + PDF/Word/文本；真正不支持的类型（如可执行文件）才拒绝
    files = {"file": ("x.exe", io.BytesIO(b"hello"), "application/x-msdownload")}
    r = await client.post("/api/upload", files=files, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_and_fetch_own_file(client: AsyncClient, auth_headers: dict):
    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    files = {"file": ("t.png", io.BytesIO(png_header), "image/png")}
    up = await client.post("/api/upload", files=files, headers=auth_headers)
    assert up.status_code == 200
    data = up.json()
    assert data["path"].startswith("/api/uploads/")
    filename = data["filename"]
    assert auth_headers["Authorization"]  # noqa: S101

    user_id = (await client.get("/api/auth/me", headers=auth_headers)).json()["user"]["id"]
    assert filename.startswith(f"{user_id}_")

    get_r = await client.get(f"/api/uploads/{filename}", headers=auth_headers)
    assert get_r.status_code == 200


@pytest.mark.asyncio
async def test_cannot_fetch_other_user_upload(client: AsyncClient, auth_headers: dict):
    other = f"other_{__import__('os').urandom(3).hex()}"
    reg = await client.post(
        "/api/auth/register",
        json={"username": other, "password": "pass1234"},
    )
    other_headers = {"Authorization": f"Bearer {reg.json()['token']}"}

    png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    files = {"file": ("t.png", io.BytesIO(png_header), "image/png")}
    up = await client.post("/api/upload", files=files, headers=other_headers)
    filename = up.json()["filename"]

    r = await client.get(f"/api/uploads/{filename}", headers=auth_headers)
    assert r.status_code == 403
