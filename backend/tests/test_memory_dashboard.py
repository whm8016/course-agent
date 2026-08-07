"""仪表盘 /memory/dashboard 止血测试：mem0 与 load_graphs 并行 + 短 TTL 读缓存。

止血范围（学情分析四模块设计 §模块四 P1）：
- 并行化：mem0.get_all 与 load_graphs 互相独立，asyncio.gather 砍串行叠加；
  该性质由 gather 两独立 awaitable 的结构保证，无需脆弱的时序断言。
- 缓存：cache-aside，二次打开命中缓存，get_all 不重复调用。

注：cache_get/cache_set 用 dict 替身注入（同 test_llm_catalog 做法）——
memory:// 的 async redis pool 在测试里不可用（cache 层静默降级为 always-miss），
生产真 Redis 才生效；故这里只验「端点侧 cache-aside 编排逻辑」。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_dashboard_cache_aside_hits_on_second_call(client, auth_headers, monkeypatch):
    """首次 MISS 现算并写缓存，二次 HIT 直接返回缓存（get_all 全程只调一次）。"""
    import api.memory as mem_mod

    _store: dict = {}

    async def fake_get(key):
        return _store.get(key)

    async def fake_set(key, value, ttl=60):
        _store[key] = value

    monkeypatch.setattr(mem_mod, "cache_get", fake_get)
    monkeypatch.setattr(mem_mod, "cache_set", fake_set)

    fake_get_all = AsyncMock(return_value={"results": [{"memory": "记住了欧姆定律"}]})
    fake_mem = SimpleNamespace(get_all=fake_get_all)

    fake_kg = {"nodes": [{"id": "k1", "status": "active", "risk": 0.9}]}
    fake_eg = {"nodes": [{"id": "e1", "error_count": 3}]}

    with patch("api.memory.get_memory", return_value=fake_mem), \
         patch("api.memory.load_graphs", new=AsyncMock(return_value=(fake_kg, fake_eg))):
        r1 = await client.get("/api/memory/dashboard", headers=auth_headers)
        r2 = await client.get("/api/memory/dashboard", headers=auth_headers)

    assert r1.status_code == 200, r1.text
    assert r2.status_code == 200, r2.text
    body = r1.json()
    # 写进缓存的证据：二次返回与首次完全一致
    assert r1.json() == r2.json()
    assert body["knowledge_node_count"] == 1
    assert body["error_node_count"] == 1
    assert body["high_risk_points"][0]["id"] == "k1"
    assert body["frequent_errors"][0]["id"] == "e1"
    # 关键：get_memory().get_all 只被调用一次 —— 第二次命中缓存，未现算
    assert fake_get_all.call_count == 1
