"""M-2：shutdown_leader 中 asyncio.shield 可导致 campaign 重启的回归测试。

根因（AUDIT M-2，``leader.py`` 原 ``shutdown_leader``）：
``_active_task.cancel()`` 后用 ``asyncio.wait_for(asyncio.shield(_active_task), 2.0)``。
若 cancel 的瞬间该 task 正在 ``_lose_leader()``（如 CAS 续约刚返回 0），``_lose_leader``
会 ``asyncio.create_task(_campaign_loop())`` 并把新 task 赋给 ``_active_task``。此时
``_active_task`` 已是**新竞选 task**，``shield`` 反而保护它不被取消 → ``wait_for`` 超时后
新 campaign task 仍在跑，shutdown 之后本进程竞选继续、可能重新抢锁变 leader。

修法：去掉 ``shield``；先存旧引用，await 它退出后，再把 cancel 期间被 reassign 出来的
新 task 也取消，确保无残留 loop。

interleaving（cancel 与 _lose_leader reassign 的交错）：
  - 正常：task 无 reassign → cancel 旧 task、``_active_task=None``，干净。
  - 竞态（本测试）：旧 task 在 cancel 前 reassign ``_active_task=新campaign``；
    修复后旧 task 被取消，新 campaign task 也被显式取消、不残留。
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


async def test_shutdown_cancels_leaked_campaign_task():
    """shutdown 期间 _active_task 被 reassign 为新 campaign → 必须一并取消，不残留。

    精确复现 M-2 的真实 interleaving（cancel 与 reassign 的交错）：
      1. 旧 renew task 正 await 在「_lose_leader 内的某处」（这里用一个 event 模拟），
         此刻主协程调 shutdown_leader。
      2. shutdown_leader 先 ``task_to_stop = _active_task``（仍是旧 task），再
         ``task_to_stop.cancel()``——cancel 挂起待处理，旧 task 还在 await，未立即生效。
      3. ``shutdown_leader`` 让出控制权（await wait_for）→ 旧 task 恢复，先完成 reassign：
         ``_active_task = 新campaign task``，再退出。
      4. 旧实现 ``asyncio.shield(_active_task)`` 在步骤 2 求值，拿到的是旧 task；步骤 3
         reassign 后新 task 既未被 cancel、也未被 await → 泄漏，shutdown 后继续抢锁。
      修复后：await 旧 task 退出后，显式检查 ``_active_task`` 是否被 reassign，是则再 cancel。
    """
    redis_mock = AsyncMock()
    redis_mock.eval.return_value = 1  # CAS 删锁成功
    leader._redis_client = redis_mock
    leader._is_leader = True
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=AsyncMock())

    spawned: list = []
    reassign_done = asyncio.Event()  # 旧 task 已 reassign 并让出控制权（_invoke_callback await 处）

    async def inner_campaign_runs():
        # 新竞选 task：长跑。旧 shield 实现下它会在 shutdown 后活下来继续抢锁。
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            spawned.append("campaign_cancelled")
            raise

    async def ghost_renew_in_lose_leader():
        # 复现 _renew_lease_loop 在 _lose_leader 内：reassign（同步 create_task）先发生，
        # 随后 await _invoke_callback(_on_lose)——主协程在此 await 点发起 shutdown。
        # 此时 _active_task 已是新 campaign task；旧 shield 实现若 cancel 作用于旧 task
        # 而 shield 求值到新 task，新 task 可能逃过取消继续抢锁。修复后无论哪个时序，
        # 被 reassign 出来的新 task 都会被显式 cancel。
        new_task = asyncio.create_task(inner_campaign_runs(), name="ghost:campaign")
        leader._active_task = new_task  # reassign 先于 shutdown
        spawned.append(("reassigned", new_task))
        reassign_done.set()
        # await 一拍让主协程 shutdown_leader 抢入（模拟 _invoke_callback await 点）
        await asyncio.sleep(0.5)

    ghost = asyncio.create_task(ghost_renew_in_lose_leader(), name="ghost:renew")
    leader._active_task = ghost
    await reassign_done.wait()  # 旧 task 已 reassign、让出控制权
    # 此时 _active_task 是新 campaign task，ghost 仍在 await（未退出）
    assert leader._active_task is not ghost
    assert not ghost.done()

    await leader.shutdown_leader()

    # M-2 核心：被 reassign 出来的新 campaign task 必须被取消（旧 shield 实现会留下它继续抢锁）
    assert "campaign_cancelled" in spawned
    # _active_task 最终归 None（无残留 loop）
    assert leader._active_task is None
    # shutdown 置位了关停标志
    assert leader.is_shutting_down() is True


async def test_shutdown_clean_when_no_reassign():
    """正常 shutdown：_active_task 未被 reassign → cancel 旧 task、无多余取消，干净退出。"""
    redis_mock = AsyncMock()
    redis_mock.eval.return_value = 0
    leader._redis_client = redis_mock
    leader._is_leader = True
    leader.register_leader_callbacks(on_gain=AsyncMock(), on_lose=AsyncMock())

    async def plain_loop():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(plain_loop(), name="plain:renew")
    leader._active_task = task
    await asyncio.sleep(0)

    await leader.shutdown_leader()

    assert task.cancelled() or task.done()
    assert leader._active_task is None
