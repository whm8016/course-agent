"""P0-C：mem0 build_memory_context 的 search_threshold 传递 + TypeError 自适应降级。

不连真实 PG/mem0：mock get_memory 返回假 AsyncMemory，验证 threshold 透传、默认 0 不传、
mem0 不支持 threshold 时降级重试不阻塞。recency_decay_lambda/threshold 真实生效待 Docker。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.memory import mem0_client
from settings import get_settings


def _mem_settings(monkeypatch, **overrides):
    """配置 mem0 settings：默认关冲突检测（简化测试），可覆盖 search_threshold/time_decay。"""
    cfg = get_settings().mem0
    monkeypatch.setattr(cfg, "conflict_detect_enabled", False)
    monkeypatch.setattr(cfg, "time_decay_enabled", overrides.get("time_decay_enabled", False))
    monkeypatch.setattr(cfg, "search_threshold", overrides.get("search_threshold", 0.0))
    return cfg


def _mock_memory(search_fn):
    m = MagicMock()
    m.search = search_fn
    return m


@pytest.mark.asyncio
async def test_threshold_passed_when_positive(monkeypatch):
    _mem_settings(monkeypatch, search_threshold=0.3)
    captured: dict = {}

    async def fake_search(query, **kwargs):
        captured["kwargs"] = kwargs
        return {"results": [{"memory": "事实A"}]}

    monkeypatch.setattr(mem0_client, "get_memory", lambda: _mock_memory(fake_search))
    out = await mem0_client.build_memory_context("u1", "查询", top_k=5)

    assert "事实A" in out
    assert captured["kwargs"].get("threshold") == 0.3


@pytest.mark.asyncio
async def test_threshold_not_passed_when_zero(monkeypatch):
    """threshold=0（默认）→ 不传 threshold，行为等价旧实现。"""
    _mem_settings(monkeypatch, search_threshold=0.0)
    captured: dict = {}

    async def fake_search(query, **kwargs):
        captured["kwargs"] = kwargs
        return {"results": [{"memory": "事实B"}]}

    monkeypatch.setattr(mem0_client, "get_memory", lambda: _mock_memory(fake_search))
    await mem0_client.build_memory_context("u1", "查询")

    assert "threshold" not in captured["kwargs"]


@pytest.mark.asyncio
async def test_threshold_typeerror_falls_back(monkeypatch):
    """mem0 不支持 threshold（TypeError）→ 降级重试（第二次无 threshold），仍返回结果。"""
    _mem_settings(monkeypatch, search_threshold=0.3)
    calls = {"n": 0}

    async def fake_search(query, **kwargs):
        calls["n"] += 1
        if "threshold" in kwargs:
            raise TypeError("unexpected keyword argument 'threshold'")
        return {"results": [{"memory": "事实C"}]}

    monkeypatch.setattr(mem0_client, "get_memory", lambda: _mock_memory(fake_search))
    out = await mem0_client.build_memory_context("u1", "查询")

    assert calls["n"] == 2          # 首次 TypeError → 降级第二次
    assert "事实C" in out


@pytest.mark.asyncio
async def test_empty_user_or_query_returns_empty(monkeypatch):
    """空 user_id / query → 直接返回空串，不调 search。"""
    _mem_settings(monkeypatch)
    called = {"n": 0}

    async def fake_search(query, **kwargs):
        called["n"] += 1
        return {"results": []}

    monkeypatch.setattr(mem0_client, "get_memory", lambda: _mock_memory(fake_search))
    assert await mem0_client.build_memory_context("", "查询") == ""
    assert await mem0_client.build_memory_context("u1", "") == ""
    assert called["n"] == 0
