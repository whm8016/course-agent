"""第四批-1：成本配额（per-user/course/日 Redis 计数 + 软降级）回归测试。

覆盖：
- accrue_cost：incrbyfloat+expire 累加、cost<=0/空 user 跳过、Redis 异常静默吞
- check_quota：未超/已超阈值判定、空 user/零预算放行、Redis 异常放行（绝不误伤）
- settings.cost_quota：默认关（enabled=False，行为零变化）
- loop 累加门控：enabled 关时不调 accrue、开时调用
- accrue→check 端到端：累计过阈值后 check 返回 over=True
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# FakeRedis：记录 incrbyfloat/expire/get 调用，内存态可读回
# ---------------------------------------------------------------------------
class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.calls: list[str] = []

    async def incrbyfloat(self, key, amount):
        self.calls.append(f"incr:{key}:{amount}")
        cur = float(self.store.get(key, "0")) + float(amount)
        self.store[key] = repr(cur)
        return cur

    async def expire(self, key, ttl):
        self.calls.append(f"expire:{key}:{ttl}")
        return True

    async def get(self, key):
        self.calls.append(f"get:{key}")
        return self.store.get(key)


class _BoomRedis:
    async def incrbyfloat(self, *a, **k):
        raise RuntimeError("redis down")

    async def expire(self, *a, **k):
        raise RuntimeError("redis down")

    async def get(self, *a, **k):
        raise RuntimeError("redis down")


@pytest.fixture
def budget(monkeypatch):
    """固定 daily_budget_usd=1.0（settings 是 lru_cache 单例，必须 monkeypatch 自动还原）。"""
    from settings import get_settings
    monkeypatch.setattr(get_settings().cost_quota, "daily_budget_usd", 1.0)
    return 1.0


# ---------------------------------------------------------------------------
# accrue_cost
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_accrue_increments_and_sets_ttl(monkeypatch):
    from core.db import cache
    from core.quota.cost_quota import accrue_cost
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_pool", lambda: fake)

    await accrue_cost("u1", "c1", 0.123)

    # incrbyfloat + expire 都被调，key 含 user/course/当日日期段
    assert any(c.startswith("incr:ca:costquota:u1:c1:") for c in fake.calls)
    assert any(c.startswith("expire:ca:costquota:u1:c1:") for c in fake.calls)


@pytest.mark.asyncio
async def test_accrue_skips_zero_cost_and_empty_user(monkeypatch):
    from core.db import cache
    from core.quota.cost_quota import accrue_cost
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_pool", lambda: fake)

    await accrue_cost("u1", "c1", 0.0)       # cost<=0 → 跳过
    await accrue_cost("u1", "c1", -1.0)      # cost<0 → 跳过
    await accrue_cost("", "c1", 0.5)         # 空 user → 跳过

    assert fake.calls == []  # 一次 Redis 都没碰


@pytest.mark.asyncio
async def test_accrue_swallows_redis_error(monkeypatch):
    from core.db import cache
    from core.quota.cost_quota import accrue_cost
    monkeypatch.setattr(cache, "_get_pool", lambda: _BoomRedis())

    # Redis 挂了也不抛（best-effort，绝不阻塞业务）
    await accrue_cost("u1", "c1", 0.5)


# ---------------------------------------------------------------------------
# check_quota
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_check_under_budget(monkeypatch, budget):  # noqa: ARG001
    from core.db import cache
    from core.quota.cost_quota import check_quota, accrue_cost
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_pool", lambda: fake)

    await accrue_cost("u2", "c2", 0.3)
    over, used, bud = await check_quota("u2", "c2")
    assert over is False
    assert used == pytest.approx(0.3, abs=1e-6)
    assert bud == budget


@pytest.mark.asyncio
async def test_check_over_budget(monkeypatch, budget):  # noqa: ARG001
    from core.db import cache
    from core.quota.cost_quota import check_quota, accrue_cost
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_pool", lambda: fake)

    # 累计 1.5（预算 1.0）→ over=True
    await accrue_cost("u3", "c3", 1.0)
    await accrue_cost("u3", "c3", 0.5)
    over, used, _ = await check_quota("u3", "c3")
    assert over is True
    assert used == pytest.approx(1.5, abs=1e-6)


@pytest.mark.asyncio
async def test_check_redis_error_returns_not_over(monkeypatch):
    from core.db import cache
    from core.quota.cost_quota import check_quota
    monkeypatch.setattr(cache, "_get_pool", lambda: _BoomRedis())

    # Redis 异常 → 放行（over=False），绝不因配额模块故障误伤业务
    over, used, _ = await check_quota("u4", "c4")
    assert over is False
    assert used == 0.0


@pytest.mark.asyncio
async def test_check_empty_user(monkeypatch):
    from core.db import cache
    from core.quota.cost_quota import check_quota
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "_get_pool", lambda: fake)

    over, _, _ = await check_quota("", "c5")  # 空 user → 放行
    assert over is False


# ---------------------------------------------------------------------------
# settings 默认
# ---------------------------------------------------------------------------
def test_cost_quota_disabled_by_default():
    from settings import get_settings
    cfg = get_settings().cost_quota
    assert cfg.enabled is False          # 默认关：check/accrue 全短路，行为零变化
    assert cfg.daily_budget_usd > 0
    assert cfg.degrade_model is True


# ---------------------------------------------------------------------------
# loop 累加门控：enabled 关→不调 accrue；开→调用
# ---------------------------------------------------------------------------
async def _async_iter(items):
    for it in items:
        yield it


def _make_chunk(content: str = ""):
    """匹配 _one_round 实际读取的 streaming chunk 字段（同 test_agent_loop）。"""
    delta = MagicMock()
    delta.content = content or None
    delta.tool_calls = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


@pytest.mark.asyncio
async def test_loop_accrue_gated_by_enabled(monkeypatch):
    """enabled=False → loop 结束不碰 accrue；enabled=True → 调一次 accrue。"""
    from core.agentic.loop import run_agent_loop
    from core.context import UnifiedContext
    from core.stream_bus import StreamBus
    from settings import get_settings

    async def _run_case(enabled: bool):
        monkeypatch.setattr(get_settings().cost_quota, "enabled", enabled)
        calls = []
        monkeypatch.setattr(
            "core.quota.cost_quota.accrue_cost",
            AsyncMock(side_effect=lambda *a, **k: calls.append(a)),
        )
        # 模型一轮直接作答（不调工具）
        stream_obj = MagicMock()
        stream_obj.__aiter__ = lambda self: _async_iter([_make_chunk("ok 答案")])
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream_obj)
        monkeypatch.setattr("core.agentic.loop._default_client", mock_client)

        ctx = UnifiedContext(
            course_id="qc", user_id="qu", user_message="问", mode="chat",
        )
        await run_agent_loop(
            context=ctx, stream=StreamBus(),
            system_prompt="sys", tool_schemas=None,
        )
        return calls

    c_off = await _run_case(False)
    assert c_off == []          # 关 → 零 accrue 调用
    c_on = await _run_case(True)
    assert len(c_on) == 1       # 开 → 恰好一次 accrue（单 loop）
    # accrue 调用参数：(user_id, course_id, loop_cost)
    assert c_on[0][0] == "qu" and c_on[0][1] == "qc"
