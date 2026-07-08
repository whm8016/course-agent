"""M-3：worker fallback Redis 连接未 close 的回归测试。

根因（AUDIT M-3，``worker.py:194-199,219-224``）：``cron_flush_memory`` 和
``flush_all_pending_job`` 当 ``ctx["redis"]`` 为 None 时 fallback 用
``aioredis.from_url`` 自建连接，**但从不 ``aclose()``** → 每次 cron/flush job 泄漏
一个连接，ARQ worker 长跑会耗尽 Redis 连接池。

修法：抽 ``_resolve_redis(ctx)`` 返回 ``(redis, should_close)``。ARQ 注入的连接
（ctx["redis"]）should_close=False（ARQ 管其生命周期）；fallback 自建的 should_close=True，
调用方用 try/finally 在用完后 ``aclose()``。

interleaving（每个 job 的连接生命周期）：
  - ctx 有 redis：复用 ARQ 连接，不 aclose（不应关掉别人管的连接）。
  - ctx 无 redis：from_url 自建 → scan/flush → finally aclose（本测试验证）。
  - 异常路径：scan/flush 抛错 → finally 仍 aclose（不因异常再泄漏一个）。
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import worker  # noqa: E402


async def test_resolve_redis_uses_ctx_when_present():
    """ctx["redis"] 存在 → 复用、should_close=False（不关 ARQ 管理的连接）。"""
    ctx_redis = MagicMock(name="arq_redis")
    ctx = {"redis": ctx_redis}

    r, should_close = await worker._resolve_redis(ctx)

    assert r is ctx_redis
    assert should_close is False


async def test_resolve_redis_fallback_when_ctx_missing():
    """ctx 无 redis → from_url 自建、should_close=True（用完必须关）。"""
    fallback_redis = MagicMock(name="fallback_redis")
    ctx = {}

    with patch("redis.asyncio.from_url", return_value=fallback_redis) as mock_from_url:
        r, should_close = await worker._resolve_redis(ctx)

    assert r is fallback_redis
    assert should_close is True
    mock_from_url.assert_called_once()


async def test_cron_flush_memory_closes_fallback_redis():
    """fallback 路径：cron_flush_memory 结束后 aclose 自建连接（M-3 核心）。"""
    fallback_redis = AsyncMock(name="fallback_redis")
    ctx = {}  # 无 ctx["redis"] → fallback

    with patch("redis.asyncio.from_url", return_value=fallback_redis), patch(
        "core.memory.flush_manager.scan_and_flush", new=AsyncMock(return_value=3)
    ):
        await worker.cron_flush_memory(ctx)

    fallback_redis.aclose.assert_awaited_once()


async def test_flush_all_pending_closes_fallback_redis():
    """fallback 路径：flush_all_pending_job 结束后 aclose 自建连接。"""
    fallback_redis = AsyncMock(name="fallback_redis")
    ctx = {}

    with patch("redis.asyncio.from_url", return_value=fallback_redis), patch(
        "core.memory.flush_manager.flush_all_pending", new=AsyncMock(return_value=5)
    ):
        await worker.flush_all_pending_job(ctx)

    fallback_redis.aclose.assert_awaited_once()


async def test_cron_flush_does_not_close_ctx_redis():
    """ctx 有 redis → 不 aclose（ARQ 管理其生命周期，关了会破坏后续 job）。"""
    ctx_redis = AsyncMock(name="arq_redis")
    ctx = {"redis": ctx_redis}

    with patch(
        "core.memory.flush_manager.scan_and_flush", new=AsyncMock(return_value=2)
    ):
        await worker.cron_flush_memory(ctx)

    ctx_redis.aclose.assert_not_awaited()


async def test_cron_flush_closes_fallback_even_on_exception():
    """异常路径：scan_and_flush 抛错 → finally 仍 aclose（不因异常再泄漏连接）。"""
    fallback_redis = AsyncMock(name="fallback_redis")
    ctx = {}

    with patch("redis.asyncio.from_url", return_value=fallback_redis), patch(
        "core.memory.flush_manager.scan_and_flush",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        # cron_flush_memory 内部 try/except 吞异常，不向上抛
        await worker.cron_flush_memory(ctx)

    fallback_redis.aclose.assert_awaited_once()
