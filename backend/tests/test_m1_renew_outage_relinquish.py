"""M-1：Redis 中断 > TTL 时短暂双 leader 的回归测试。

根因（AUDIT M-1，``leader.py`` 原 renew loop）：续约 Redis 异常时**只 log 重试、
保持 leader 身份**（lease 语义）。但若中断持续超过一个 lease TTL，锁在 Redis 侧
早已过期被别的 worker 抢走，本 worker 却仍 ``_is_leader=True``、继续当 leader →
双 leader 窗口（Cron/Bot/MCP 跑两份、单例状态错乱）。

修法：续约 loop 引入失败时间窗 ``fail_since``。自首次失败（异常或无 client）起，
若累积时长 >= TTL 即主动 ``_lose_leader``——锁必然已不在我们名下，主动让位比
「装作还是 leader」安全。**短抖动（< TTL）仍保持身份**，不误伤既有 lease 语义
（见 test_leader.py::test_renew_redis_error_keeps_leadership）。

interleaving（monotonic 时钟为 t）：
  - 正常：eval 成功 → ``fail_since=None``，永不丢锁。
  - 短抖动：t0 异常 → fail_since=t0；t1(<t0+TTL) 恢复成功 → fail_since=None，保持。
  - 长中断（本测试）：t0 异常 → fail_since=t0；t1,t2 持续异常；
    t_k=t0+TTL 仍异常 → ``_lose_leader`` 被调，``_is_leader=False``。
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.leader as leader  # noqa: E402


@pytest.fixture(autouse=True)
async def _reset_leader_state():
    """每个测试前后重置 leader 模块全局状态；隔离 prometheus 上报。"""
    with patch.object(leader, "set_leader_status"):
        leader._is_leader = False
        leader._redis_client = None
        leader._on_gain = None
        leader._on_lose = None
        leader._shutting_down = False
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
    client = AsyncMock()
    client.set.return_value = True
    client.get.return_value = leader._worker_id
    client.eval.return_value = 1
    return client


async def test_renew_outage_longer_than_ttl_relinquishes():
    """续约持续失败 > TTL → 主动 _lose_leader，不再装作 leader（消除双主窗口）。"""
    redis_mock = _make_redis_mock()
    redis_mock.eval.side_effect = RuntimeError("redis connection lost")  # 每轮续约都失败

    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=on_lose)

    # TTL 压到 0.05s、续约周期 0.01s：约 1-2 个周期后累积失败 > TTL 即丢锁。
    with patch.object(leader, "_RENEW_INTERVAL", 0.01), patch.object(
        leader, "_LEASE_TTL", 0.05
    ), patch.object(leader, "aioredis") as fake_aioredis:
        fake_aioredis.from_url.return_value = redis_mock
        became = await leader.try_become_leader()
        assert became is True

        # 让 renew loop 跑足够多周期（> TTL）
        await asyncio.sleep(0.3)

    assert leader.is_leader() is False  # 长中断 > TTL → 让位，不再双 leader
    on_lose.assert_awaited_once()
    assert redis_mock.eval.await_count >= 2  # 确实尝试了续约


async def test_renew_outage_shorter_than_ttl_keeps_leadership():
    """续约短抖动（< TTL）→ 保持 leader 身份（lease 语义不误伤）。"""
    redis_mock = _make_redis_mock()
    redis_mock.eval.side_effect = RuntimeError("transient blip")

    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=on_lose)

    # TTL=30s（默认量级），仅抖动 0.1s，远不到丢锁阈值。
    with patch.object(leader, "_RENEW_INTERVAL", 0.01), patch.object(
        leader, "_LEASE_TTL", 30
    ), patch.object(leader, "aioredis") as fake_aioredis:
        fake_aioredis.from_url.return_value = redis_mock
        await leader.try_become_leader()

        await asyncio.sleep(0.1)

    assert leader.is_leader() is True  # 短抖动保持身份
    on_lose.assert_not_awaited()


async def test_renew_no_client_longer_than_ttl_relinquishes():
    """无 redis client（init 失败）持续 > TTL → 同样主动让位（fail_since 覆盖该分支）。"""
    on_lose = AsyncMock()
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=on_lose)

    def _ensure_returns_none():
        # _ensure_client 内部 get_settings() 可能抛错 → _redis_client 保持 None
        leader._redis_client = None

    with patch.object(leader, "_RENEW_INTERVAL", 0.01), patch.object(
        leader, "_LEASE_TTL", 0.05
    ), patch.object(leader, "_ensure_client", side_effect=_ensure_returns_none):
        # 直接进 leader 态（绕过 try_become_leader 的 _try_acquire，它需要 client）。
        leader._redis_client = None
        await leader._become_leader()

        await asyncio.sleep(0.3)

    assert leader.is_leader() is False
    on_lose.assert_awaited_once()
