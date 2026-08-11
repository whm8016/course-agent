"""P0 IDOR 回归：mem0 记忆改删越权（api/memory.py PUT/DELETE）。

攻击者视角：B 用户拿到 A 的 memory_id（可枚举/猜测），调 PUT/DELETE 改删 A 的记忆。
- 修复前：memory.py 只认 memory_id 不校验归属 -> B 可改删 A 的记忆。
- 修复后：改删前先 m.get(id) 校验 user_id 归属，不属于当前用户 -> 404（不泄露存在性），
  且 update/delete 根本不被调用。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


async def _my_id(client, auth_headers) -> str:
    """取 auth_headers 对应的当前用户 id（构造 owner 合法用例用）。"""
    me = await client.get("/api/auth/me", headers=auth_headers)
    assert me.status_code == 200, me.text
    return str(me.json()["user"]["id"])


@pytest.mark.asyncio
async def test_put_memory_rejects_non_owner(client, auth_headers):
    """B 用户改 A 的记忆 -> 404，update 不被调用。"""
    fake_mem = SimpleNamespace(
        get=AsyncMock(return_value={"id": "mem1", "user_id": "user_A", "memory": "A 的记忆"}),
        update=AsyncMock(return_value={"message": "ok"}),
    )
    with patch("api.memory.get_memory", return_value=fake_mem):
        r = await client.put(
            "/api/memory/mem1",
            headers=auth_headers,
            json={"content": "被篡改"},
        )
    assert r.status_code == 404, r.text
    fake_mem.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_memory_rejects_non_owner(client, auth_headers):
    """B 用户删 A 的记忆 -> 404，delete 不被调用。"""
    fake_mem = SimpleNamespace(
        get=AsyncMock(return_value={"id": "mem1", "user_id": "user_A", "memory": "A 的记忆"}),
        delete=AsyncMock(),
    )
    with patch("api.memory.get_memory", return_value=fake_mem):
        r = await client.delete("/api/memory/mem1", headers=auth_headers)
    assert r.status_code == 404, r.text
    fake_mem.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_put_memory_rejects_missing_memory(client, auth_headers):
    """memory_id 不存在（get 返回 None）-> 404，不泄露存在性。"""
    fake_mem = SimpleNamespace(
        get=AsyncMock(return_value=None),
        update=AsyncMock(),
    )
    with patch("api.memory.get_memory", return_value=fake_mem):
        r = await client.put(
            "/api/memory/nope",
            headers=auth_headers,
            json={"content": "x"},
        )
    assert r.status_code == 404, r.text
    fake_mem.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_put_memory_allows_owner(client, auth_headers):
    """owner 改自己的记忆 -> 200（回归：归属校验不误伤合法操作）。"""
    my_id = await _my_id(client, auth_headers)
    fake_mem = SimpleNamespace(
        get=AsyncMock(return_value={"id": "mem1", "user_id": my_id, "memory": "旧内容"}),
        update=AsyncMock(return_value={"message": "ok"}),
    )
    with patch("api.memory.get_memory", return_value=fake_mem):
        r = await client.put(
            "/api/memory/mem1",
            headers=auth_headers,
            json={"content": "新内容"},
        )
    assert r.status_code == 200, r.text
    fake_mem.update.assert_awaited_once()
