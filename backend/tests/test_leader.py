"""Leader 选举单测：竞选者循环 + CAS 续约 + 状态翻转回调。

用 mock redis 控制 SETNX / eval 返回值，验证：
  - 非 leader 竞选 loop 在锁可用后接管（on_gain 恰好一次）
  - 续约 CAS 返回 0（锁被别人抢）→ on_lose + 切回 follower
  - 续约 Redis 异常 → loop 不崩、不丢锁（lease 语义保持 leader 身份）
  - _become_leader 重入保护（on_gain 只调一次）
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# 确认 backend 在 path 上
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.leader as leader  # noqa: E402


async def _wait_for(pred, timeout: float = 1.5) -> bool:
    """轮询等待条件成立（兼容 CI 抖动）。"""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


@pytest.fixture(autouse=True)
async def _reset_leader_state():
    """每个测试前后重置 leader 模块全局状态；隔离 prometheus 上报。"""
    with patch.object(leader, "set_leader_status"):
        leader._is_leader = False
        leader._redis_client = None
        leader._on_gain = None
        leader._on_lose = None
        if leader._active_task and not leader._active_task.done():
            leader._active_task.cancel()
        leader._active_task = None

        yield

        if leader._active_task and not leader._active_task.done():
            leader._active_task.cancel()
            try:
                await asyncio.wait_for(leader._active_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        leader._active_task = None
        leader._is_leader = False
        leader._redis_client = None


def _make_redis_mock() -> AsyncMock:
    """构造 mock redis client：set/get/eval/delete/aclose 均可控。"""
    client = AsyncMock()
    client.set.return_value = None
    client.get.return_value = None
    client.eval.return_value = 1  # CAS 续约默认成功
    client.delete.return_value = 1
    return client


async def test_campaign_takes_over_when_lock_becomes_free():
    """非 leader 持续竞选：首次抢失败，下一周期锁可用时接管 + on_gain 恰好一次。"""
    redis_mock = _make_redis_mock()
    redis_mock.set.side_effect = [None, True]  # 启动抢锁失败 → 竞选首次成功
    redis_mock.get.return_value = leader._worker_id  # 二次确认通过

    on_gain = AsyncMock()
    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=on_gain, on_lose=on_lose)

    with patch.object(leader, "_CAMPAIGN_INTERVAL", 0.01), patch.object(
        leader, "aioredis"
    ) as fake_aioredis:
        fake_aioredis.from_url.return_value = redis_mock
        became = await leader.try_become_leader()

        assert became is False
        assert leader.is_leader() is False
        # 竞选 loop：sleep 0.01 → set=True → become_leader → on_gain
        assert await _wait_for(lambda: leader.is_leader())

    assert leader.is_leader() is True
    on_gain.assert_awaited_once()
    on_lose.assert_not_awaited()


async def test_renew_loss_triggers_on_lose():
    """续约 CAS 返回 0（锁被别人抢）→ on_lose 调用、切回 follower。"""
    redis_mock = _make_redis_mock()
    redis_mock.set.side_effect = [True, None]  # 首次抢到 → 丢锁后竞选抢不回
    redis_mock.get.return_value = leader._worker_id
    redis_mock.eval.return_value = 0  # CAS 续约失败

    on_gain = AsyncMock()
    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=on_gain, on_lose=on_lose)

    with patch.object(leader, "_RENEW_INTERVAL", 0.01), patch.object(
        leader, "_CAMPAIGN_INTERVAL", 0.01
    ), patch.object(leader, "aioredis") as fake_aioredis:
        fake_aioredis.from_url.return_value = redis_mock
        became = await leader.try_become_leader()

        assert became is True
        on_gain.assert_awaited_once()
        # renew loop：sleep 0.01 → eval=0 → lose_leader → on_lose
        assert await _wait_for(lambda: on_lose.await_count > 0)

    assert leader.is_leader() is False
    on_lose.assert_awaited_once()


async def test_renew_redis_error_keeps_leadership():
    """续约时 Redis 异常 → loop 不崩、不丢锁（lease 语义保持 leader 身份）。"""
    redis_mock = _make_redis_mock()
    redis_mock.set.return_value = True
    redis_mock.get.return_value = leader._worker_id
    redis_mock.eval.side_effect = RuntimeError("redis connection lost")

    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=on_lose)

    with patch.object(leader, "_RENEW_INTERVAL", 0.01), patch.object(
        leader, "aioredis"
    ) as fake_aioredis:
        fake_aioredis.from_url.return_value = redis_mock
        await leader.try_become_leader()

        # 让 renew loop 跑几轮（每轮 eval 抛异常）
        await asyncio.sleep(0.15)

    assert leader.is_leader() is True  # 异常不丢锁
    on_lose.assert_not_awaited()
    assert redis_mock.eval.await_count >= 2  # 持续重试续约


async def test_become_leader_idempotent():
    """_become_leader 重入保护：第二次直接 return，on_gain 只调一次。"""
    leader._redis_client = _make_redis_mock()
    on_gain = AsyncMock()
    leader.register_leader_callbacks(on_gain=on_gain, on_lose=AsyncMock())

    with patch.object(leader, "_RENEW_INTERVAL", 99):  # 抑制 renew loop 干扰
        await leader._become_leader()
        await leader._become_leader()  # 重入

    assert leader.is_leader() is True
    on_gain.assert_awaited_once()


async def test_follower_shutdown_does_not_release_lock():
    """follower shutdown 不误删锁：_is_leader=False 时 CAS eval/delete 均不调用。

    回归 Bug 1：旧实现 shutdown_leader 在 if _is_leader 之外无条件 delete(_LEADER_KEY)，
    follower 正常重启/滚动更新时会删掉真 leader 的锁 → leader 下次续约 CAS 返回 0 误判
    丢锁 → on_lose 误触发、单例空窗。
    """
    redis_mock = _make_redis_mock()
    leader._redis_client = redis_mock
    leader._is_leader = False  # follower

    await leader.shutdown_leader()

    # follower 不触碰锁（既不 CAS 删，也不无条件 delete）
    redis_mock.eval.assert_not_awaited()
    redis_mock.delete.assert_not_awaited()
    assert leader._redis_client is None  # client 已关闭


async def test_leader_shutdown_releases_lock_then_on_lose():
    """leader shutdown：CAS 删锁（eval）→ on_lose 停单例。

    锁已被新 leader 抢走（eval 返回 0）时仍执行 on_lose——本地单例清理不依赖 Redis 锁状态。
    回归 Bug 1：旧实现无条件 delete 会误删新 leader 的锁；CAS 返回 0 时安全跳过 del。
    """
    redis_mock = _make_redis_mock()
    redis_mock.eval.return_value = 0  # CAS 删锁返回 0：锁已被别人持
    leader._redis_client = redis_mock
    leader._is_leader = True

    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=on_lose)

    await leader.shutdown_leader()

    # 走 CAS 删锁（eval），不是无条件 delete
    redis_mock.eval.assert_awaited_once()
    redis_mock.delete.assert_not_awaited()
    # 锁虽已被别人抢，本地 on_lose 仍执行（停 Cron/Bot/MCP 单例）
    on_lose.assert_awaited_once()
    assert leader._is_leader is False
    assert leader._redis_client is None
