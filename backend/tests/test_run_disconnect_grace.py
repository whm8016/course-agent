"""quiz WS 断线重连宽限（服务端半边，api/run.py）。

断开不立刻 cancel turn：挂 15s 延迟处决；subscribe_turn 重连认领时掐掉；
超时无人认领才取消。interleaving 覆盖：正常超时取消 / 重连掐掉 / 重复挂载只留
最新一处决（旧的被 cancel，finally 只清自己的占位）。
"""
from __future__ import annotations

import asyncio

import pytest

from api import run as run_api


class _Trm:
    def __init__(self):
        self.cancelled: list[str] = []

    async def cancel_turn(self, turn_id: str) -> None:
        self.cancelled.append(turn_id)


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch):
    monkeypatch.setattr(run_api, "_RECONNECT_GRACE_S", 0.05)


@pytest.fixture(autouse=True)
def _clean_grace_dict():
    run_api._disconnect_grace.clear()
    yield
    run_api._disconnect_grace.clear()


async def test_grace_cancels_after_timeout_when_no_reconnect():
    """超时无人认领 → 真取消；dict 占位自清（不随 turn 数增长泄漏）。"""
    trm = _Trm()
    run_api._arm_disconnect_grace(trm, "t1")
    assert "t1" in run_api._disconnect_grace
    await asyncio.sleep(0.15)
    assert trm.cancelled == ["t1"]
    assert "t1" not in run_api._disconnect_grace


async def test_grace_disarmed_by_reconnect():
    """宽限内重连认领 → 掐掉处决，turn 继续跑。"""
    trm = _Trm()
    run_api._arm_disconnect_grace(trm, "t2")
    run_api._disarm_disconnect_grace("t2")
    await asyncio.sleep(0.15)
    assert trm.cancelled == []
    assert "t2" not in run_api._disconnect_grace


async def test_rearm_replaces_old_grace_and_cancels_once():
    """重连后又断开再挂：旧处决被 cancel、只有最新的会开火，且只取消一次。"""
    trm = _Trm()
    run_api._arm_disconnect_grace(trm, "t3")
    old_task = run_api._disconnect_grace["t3"]
    run_api._arm_disconnect_grace(trm, "t3")  # 再次断线重挂
    assert run_api._disconnect_grace["t3"] is not old_task
    await asyncio.sleep(0.15)
    assert trm.cancelled == ["t3"]  # 一次，不多不少
    assert "t3" not in run_api._disconnect_grace


async def test_disarm_unknown_turn_is_noop():
    run_api._disarm_disconnect_grace("never-armed")  # 不抛即通过
