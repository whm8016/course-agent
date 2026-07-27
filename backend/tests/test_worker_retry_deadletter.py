"""plan 第三批-2：ARQ max_tries 自动重试 + 死信队列回归测试。

验证 ``_push_deadletter_if_terminal``：
- job_try < max_tries（中间失败）→ 不写死信（交由 ARQ 自动重试）
- job_try >= max_tries（终态失败）→ rpush 到 ``arq:deadletter`` + expire，payload 含 job_id/function/error
- ctx 有 redis → 复用、不 aclose；fallback 自建 → 用完 aclose（同 M-3 连接生命周期）
- 死信写入本身失败 → best-effort 不抛（观测不影响主流程）
- WorkerSettings 显式声明 max_tries / retry_jobs，且 max_tries 引用同一 ``_ARQ_MAX_TRIES`` 常量
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import worker  # noqa: E402


async def test_no_deadletter_when_retry_remaining():
    """job_try < max_tries → 不写死信（还有重试机会，交由 ARQ 自动重试）。"""
    ctx_redis = AsyncMock()
    ctx = {"job_try": 1, "job_id": "j1", "redis": ctx_redis}  # < _ARQ_MAX_TRIES(3)
    await worker._push_deadletter_if_terminal(
        ctx, function="run_indexing", error=RuntimeError("x")
    )
    ctx_redis.rpush.assert_not_awaited()


async def test_deadletter_written_on_terminal_failure():
    """job_try >= max_tries → rpush 到 arq:deadletter + expire，并 aclose fallback 连接。"""
    fallback = AsyncMock()
    ctx = {"job_try": worker._ARQ_MAX_TRIES, "job_id": "j2"}
    with patch("redis.asyncio.from_url", return_value=fallback):
        await worker._push_deadletter_if_terminal(
            ctx, function="run_indexing", error=RuntimeError("boom")
        )
    fallback.rpush.assert_awaited_once()
    args = fallback.rpush.call_args
    assert args.args[0] == worker._DEADLETTER_KEY
    payload = json.loads(args.args[1])
    assert payload["function"] == "run_indexing"
    assert payload["job_id"] == "j2"
    assert payload["job_try"] == worker._ARQ_MAX_TRIES
    assert "boom" in payload["error"]
    fallback.expire.assert_awaited_once_with(worker._DEADLETTER_KEY, worker._DEADLETTER_TTL)
    fallback.aclose.assert_awaited_once()  # fallback 自建连接用完关


async def test_deadletter_does_not_close_ctx_redis():
    """ctx 有 redis → 复用 ARQ 连接，死信写完不 aclose（不关别人管的连接）。"""
    ctx_redis = AsyncMock()
    ctx = {"job_try": worker._ARQ_MAX_TRIES, "job_id": "j3", "redis": ctx_redis}
    await worker._push_deadletter_if_terminal(
        ctx, function="run_llamaindex_build", error=RuntimeError("y")
    )
    ctx_redis.rpush.assert_awaited_once()
    ctx_redis.aclose.assert_not_awaited()


async def test_deadletter_best_effort_on_redis_error():
    """死信写入本身失败 → 不抛（观测不影响主流程），finally 仍 aclose fallback。"""
    fallback = AsyncMock()
    fallback.rpush.side_effect = RuntimeError("redis down")
    ctx = {"job_try": worker._ARQ_MAX_TRIES, "job_id": "j4"}
    with patch("redis.asyncio.from_url", return_value=fallback):
        await worker._push_deadletter_if_terminal(  # 不应抛
            ctx, function="run_indexing", error=RuntimeError("orig")
        )
    fallback.aclose.assert_awaited_once()


def test_worker_settings_declares_retry():
    """WorkerSettings 显式声明 max_tries / retry_jobs，且 max_tries 引用 _ARQ_MAX_TRIES 常量。"""
    assert worker.WorkerSettings.max_tries == worker._ARQ_MAX_TRIES
    assert worker.WorkerSettings.retry_jobs is True
