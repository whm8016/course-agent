"""C-1/C-3 回归测试：start_singleton_services 与 shutdown 的竞态。

审计 C-1/C-3：worker 刚当选、on_gain(start) 正在执行时 Gunicorn SIGTERM →
shutdown 跑 on_lose(stop)，但 start 尚未置位 _singletons_started，stop 早退
什么都不停；start 恢复后无条件置位 True → Cron/Bot/MCP 在即将死亡的 worker 上
继续运行。

修复后：关停信号统一来自 leader.is_shutting_down()（lifespan shutdown 第一行经
leader.mark_shutting_down() 置位，最早）。start 在每个 await 后检查它，发现已关闭
则回滚已启动服务，且绝不置位 _singletons_started。

测试手法：asyncio 单线程下竞态发生在 await 点的交错。在 manager 的 start 方法
side_effect 里调 leader.mark_shutting_down()，精确复现"start 执行到一半、shutdown
抢入"这一竞态状态，验证 start 的检查点逻辑。
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.leader as leader_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singletons():
    """每个测试前后重置 main._singletons_started 与 leader._shutting_down。"""
    import main as main_mod

    main_mod._singletons_started = False
    leader_mod._shutting_down = False
    yield
    main_mod._singletons_started = False
    leader_mod._shutting_down = False


def _mgr() -> AsyncMock:
    """构造一个单例 manager mock（start/stop 方法齐全）。"""
    return AsyncMock()


def _patch_managers(bot, cron, mcp, tutorbot=True):
    """统一 patch 三个 manager getter（main.py 内部 import，patch 原模块路径）。"""
    return [
        patch.object(__import__("main"), "TUTORBOT_ENABLED", tutorbot),
        patch("core.bot.manager.get_bot_manager", return_value=bot),
        patch("services.cron.service.get_cron_service", return_value=cron),
        patch("core.mcp.manager.get_mcp_manager", return_value=mcp),
    ]


async def test_start_normal_path():
    """正常路径：三个 manager 全部启动，置位 _singletons_started。"""
    import main as main_mod

    bot, cron, mcp = _mgr(), _mgr(), _mgr()
    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.start_singleton_services()
    finally:
        for p in patches:
            p.stop()

    bot.auto_start_bots.assert_awaited_once()
    cron.start.assert_awaited_once()
    mcp.ensure_started.assert_awaited_once()
    assert main_mod._singletons_started is True


async def test_start_skipped_when_already_shutting_down():
    """关停标志已置位时，start 直接跳过，不启动任何服务、不置位。"""
    import main as main_mod

    leader_mod._shutting_down = True  # 关停标志统一来自 leader
    bot, cron, mcp = _mgr(), _mgr(), _mgr()
    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.start_singleton_services()
    finally:
        for p in patches:
            p.stop()

    bot.auto_start_bots.assert_not_awaited()
    cron.start.assert_not_awaited()
    mcp.ensure_started.assert_not_awaited()
    assert main_mod._singletons_started is False


async def test_start_aborts_when_shutdown_races_after_bot():
    """C-1/C-3 核心：bot 启动后、置位前 shutdown 插队。

    模拟 auto_start_bots 的 await 期间 shutdown 协程抢入并 mark_shutting_down()。
    start 恢复后必须：回滚已启动的 bot、不启动 cron/mcp、不置位 _singletons_started。
    """
    import main as main_mod

    bot, cron, mcp = _mgr(), _mgr(), _mgr()

    async def bot_starts_then_shutdown_marks():
        # bot 启动期间（await 点），shutdown 协程插队置位关闭标志
        await asyncio.sleep(0)
        leader_mod.mark_shutting_down()

    bot.auto_start_bots.side_effect = bot_starts_then_shutdown_marks

    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.start_singleton_services()
    finally:
        for p in patches:
            p.stop()

    # 置位未发生（C-3：不再无条件 = True）
    assert main_mod._singletons_started is False
    # cron/mcp 未启动（C-1：不在死亡 worker 上重启）
    cron.start.assert_not_awaited()
    mcp.ensure_started.assert_not_awaited()
    # bot 已启动 → 被回滚
    bot.stop_all.assert_awaited_once()


async def test_start_aborts_when_shutdown_races_after_cron():
    """C-1 另一切点：cron 启动后 shutdown 插队 → 回滚 bot+cron，mcp 不启动。"""
    import main as main_mod

    bot, cron, mcp = _mgr(), _mgr(), _mgr()

    async def cron_starts_then_shutdown_marks():
        await asyncio.sleep(0)
        leader_mod.mark_shutting_down()

    cron.start.side_effect = cron_starts_then_shutdown_marks

    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.start_singleton_services()
    finally:
        for p in patches:
            p.stop()

    assert main_mod._singletons_started is False
    mcp.ensure_started.assert_not_awaited()
    # bot + cron 均已启动 → 回滚
    bot.stop_all.assert_awaited_once()
    cron.stop.assert_awaited_once()


async def test_stop_idempotent_when_not_started():
    """未启动时 stop 是 no-op，不调用任何 manager。"""
    import main as main_mod

    bot, cron, mcp = _mgr(), _mgr(), _mgr()
    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.stop_singleton_services()
    finally:
        for p in patches:
            p.stop()

    bot.stop_all.assert_not_awaited()
    cron.stop.assert_not_awaited()
    mcp.shutdown.assert_not_awaited()
    assert main_mod._singletons_started is False


async def test_stop_tears_down_when_started():
    """已启动时 stop 调用三个 manager 的停止方法并清位。"""
    import main as main_mod

    bot, cron, mcp = _mgr(), _mgr(), _mgr()
    main_mod._singletons_started = True
    patches = _patch_managers(bot, cron, mcp, tutorbot=True)
    for p in patches:
        p.start()
    try:
        await main_mod.stop_singleton_services()
    finally:
        for p in patches:
            p.stop()

    mcp.shutdown.assert_awaited_once()
    cron.stop.assert_awaited_once()
    bot.stop_all.assert_awaited_once()
    assert main_mod._singletons_started is False
