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


@pytest.mark.asyncio
async def test_health_detailed_requires_admin(client: AsyncClient):
    """detailed 带 DB/Redis 异常详情（连接串/内部主机名）→ admin 鉴权收口。

    未认证 401/403（不再裸奔回显）；admin token 可看。基础存活探测仍走无鉴权
    /api/health（compose healthcheck 依赖它）。
    """
    r = await client.get("/api/health/detailed")
    assert r.status_code in (401, 403)

    # 注册普通用户也不行（必须是 admin）
    reg = await client.post(
        "/api/auth/register",
        json={"username": "health_stu", "password": "pass1234", "display_name": "S"},
    )
    token = reg.json()["token"]
    r2 = await client.get(
        "/api/health/detailed", headers={"Authorization": f"Bearer {token}"}
    )
    assert r2.status_code in (401, 403)
