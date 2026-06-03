"""Health endpoint smoke test."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_db_ok(client: AsyncClient):
    """GET /api/health 返回 db=ok；redis/llm 在测试环境可能降级，不强断言。"""
    r = await client.get("/api/health")
    # 200 = all ok, 503 = degraded（测试环境 redis memory:// 可能报错）
    assert r.status_code in (200, 503)
    data = r.json()
    assert "checks" in data
    assert data["checks"].get("db") == "ok"
