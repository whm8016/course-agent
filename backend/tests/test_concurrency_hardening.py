"""H-7~H-11 并发/数据完整性回归测试。

每个测试模拟竞态场景，验证修复后的防护逻辑。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.llm.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


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


# ---------------------------------------------------------------------------
# H-11 CircuitBreaker HALF_OPEN 死锁：half_open_calls 释放语义 + 配置守卫
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h11_half_open_two_probes_close():
    """success_threshold=2，half_open_max_calls=1：串行两次探测成功，熔断器应关闭。

    旧实现：half_open_calls 只增不减，第一次探测成功后名额耗尽，第二次探测
    永远被拒 → success_count 卡在 1，熔断器永久卡在 HALF_OPEN。
    """
    cb = CircuitBreaker(
        "h11_test",
        config=CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            open_timeout=0,  # 立即转 HALF_OPEN
            half_open_max_calls=1,
        ),
    )

    async def always_fail():
        raise ValueError("模拟后端不可用")

    async def always_ok():
        return "ok"

    # Step 1: 触发 OPEN（需 2 次失败：failure_threshold 被 worker 缩放至 2）
    with pytest.raises(ValueError, match="模拟后端不可用"):
        await cb.call(always_fail)
    with pytest.raises(ValueError, match="模拟后端不可用"):
        await cb.call(always_fail)
    assert cb.state == CircuitState.OPEN

    # Step 2: 第一次探测（OPEN → HALF_OPEN，放行）
    r1 = await cb.call(always_ok)
    assert r1 == "ok"
    # 新实现：success_count=1 < 2，仍 HALF_OPEN
    # 旧实现：此处会永久卡死（half_open_calls 不释放）
    assert cb.state == CircuitState.HALF_OPEN

    # Step 3: 第二次探测（新实现：half_open_calls 已释放，应能放行）
    r2 = await cb.call(always_ok)
    assert r2 == "ok"
    assert cb.state == CircuitState.CLOSED, "两次成功后应关闭熔断器"


@pytest.mark.asyncio
async def test_h11_half_open_probe_failure_reopens():
    """半开探测失败 → half_open_calls 释放 → 状态回到 OPEN（不残留死锁）。"""
    cb = CircuitBreaker(
        "h11_test_reopen",
        config=CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            open_timeout=0,  # 立即转 HALF_OPEN
            half_open_max_calls=1,
        ),
    )

    async def always_fail():
        raise ValueError("模拟后端不可用")

    async def sometimes_ok():
        return "ok"

    # Step 1: 触发 OPEN（需 2 次失败：failure_threshold 被 worker 缩放至 2）
    with pytest.raises(ValueError, match="模拟后端不可用"):
        await cb.call(always_fail)
    with pytest.raises(ValueError, match="模拟后端不可用"):
        await cb.call(always_fail)
    assert cb.state == CircuitState.OPEN

    # Step 2: 第一次探测成功（HALF_OPEN，success_count=1）
    await cb.call(sometimes_ok)
    assert cb.state == CircuitState.HALF_OPEN

    # Step 3: 第二次探测失败 → OPEN
    with pytest.raises(ValueError, match="模拟后端不可用"):
        await cb.call(always_fail)
    assert cb.state == CircuitState.OPEN, "探测失败应回到 OPEN"
    assert cb.half_open_calls == 0, "失败后 half_open_calls 应释放"


def test_h11_config_guard_against_deadlock():
    """success_threshold > half_open_max_calls → __init__ 自动抬升 half_open_max_calls。"""
    cb = CircuitBreaker(
        "h11_guard",
        config=CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=5,       # > half_open_max_calls(1)
            open_timeout=10.0,
            half_open_max_calls=1,
        ),
    )
    assert cb.config.half_open_max_calls >= cb.config.success_threshold, \
        "守卫应抬升 half_open_max_calls 至少等于 success_threshold"
