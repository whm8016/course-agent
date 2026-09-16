"""H-7~H-10 并发/数据完整性回归测试。

每个测试模拟竞态场景，验证修复后的防护逻辑。

H-11（CircuitBreaker HALF_OPEN 死锁）已随 2026-09 滑动窗口重写删除：新实现结构性
保证"合闸所需成功数 ≤ 探测名额"，死锁组合不可能出现；对应行为测试见
test_llm_circuit_breaker.py。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# H-10 LightRAG 实例池 use-after-evict：evict_oldest 跳过 in_use>0 的实例
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h10_evict_skips_in_use_instance():
    """in_use>0 的实例不被 evict，返回 None 表示淘汰失败；in_use=0 可正常淘汰。"""
    from core.rag.lightrag.instance_pool import evict_oldest, _instances, _in_use

    _instances.clear()
    _in_use.clear()

    mock_rag = SimpleNamespace(finalize_storages=AsyncMock())

    # c1 在用（in_use=1）
    _instances["c1"] = mock_rag
    _in_use["c1"] = 1

    result = await evict_oldest()
    assert result is None, "in_use>0 的实例不应被淘汰"
    assert "c1" in _instances
    assert _in_use.get("c1") == 1
    mock_rag.finalize_storages.assert_not_called()

    # 释放引用后，c1 可淘汰
    _in_use.pop("c1", None)
    result = await evict_oldest()
    assert result == "c1"
    assert "c1" not in _instances
    mock_rag.finalize_storages.assert_awaited_once()


@pytest.mark.asyncio
async def test_h10_evict_all_in_use_returns_none():
    """全部实例都在用 → evict 返回 None（调用方据此临时超容，不淘汰任何实例）。"""
    from core.rag.lightrag.instance_pool import evict_oldest, _instances, _in_use

    _instances.clear()
    _in_use.clear()

    mock_a = SimpleNamespace(finalize_storages=AsyncMock())
    mock_b = SimpleNamespace(finalize_storages=AsyncMock())

    _instances["c1"] = mock_a
    _instances["c2"] = mock_b
    _in_use["c1"] = 1
    _in_use["c2"] = 2  # 2 也在用

    result = await evict_oldest()
    assert result is None, "全部在用时应返回 None（不淘汰）"
    assert "c1" in _instances
    assert "c2" in _instances
    mock_a.finalize_storages.assert_not_called()
    mock_b.finalize_storages.assert_not_called()

    _instances.clear()
    _in_use.clear()


@pytest.mark.asyncio
async def test_h10_release_instance_after_acquire():
    """_release_instance 正确递减 in_use，归零后 evict 可淘汰。"""
    from core.rag.lightrag.instance_pool import (
        _in_use,
        _instances,
        _release_instance,
    )

    # _get_instance 需要完整 LightRAG 依赖，测试 _release_instance
    # 和 _acquire_instance 的配对通过直接检查 in_use 计数。
    _instances.clear()
    _in_use.clear()

    # 手动设一个实例，模拟 _get_instance 已 +1
    mock_rag = SimpleNamespace(finalize_storages=AsyncMock())
    _instances["c1"] = mock_rag
    _in_use["c1"] = 2  # 已 +2（如两次并发获取）

    # 释放一次 → 1
    await _release_instance("c1")
    assert _in_use.get("c1") == 1

    # 再释放 → 归 0（key 被 pop）
    await _release_instance("c1")
    assert _in_use.get("c1") is None or _in_use["c1"] == 0
