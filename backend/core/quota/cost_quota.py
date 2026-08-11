"""每用户/课程/自然日 LLM 成本配额（第四批）。

软预算 + 只降级不拒绝：超 ``daily_budget_usd`` 时把对话模型降到便宜档 fast_model
（在 chat_pipeline 解析 runtime 后切换），而非返回 4xx 阻断。Redis 不可用→放行 +
静默跳过，绝不阻塞业务。

【调研依据】
- 软限流（token/leaky bucket 的「降级而非拒绝」变体）：硬拒绝会造成「上一轮刚好超限
  →下一轮被锁死」的死锁式体验；软降级（cheaper model）在控成本的同时保证可用性，
  是 API 网关（Kong/APISIX）与 LLM 网关（LiteLLM proxy 的 budget router）的主流做法。
- 计数键按日滚动 + TTL 自清理：避免长尾 key 堆积（Redis best practice），跨进程共享
  （gunicorn -w4 下任一 worker accrue，任一 worker check 都读到同一累计值）。

【滞后性】成本在 loop 结束后按 estimate_cost 累加（loop.py），故「推过预算的那一轮」
本身不降级，下一轮才降级——quota 的固有滞后，符合软限流惯例，非 bug。

复用 ``core.db.cache._get_pool``（decode_responses=True + 2s 连接超时），不自建连接池。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from settings import get_settings

logger = logging.getLogger(__name__)

# 键 TTL（2 天）：日键次日即不再写入，留 1 天余量防时区/边界漂移导致提前过期。
_KEY_TTL = 2 * 24 * 3600


def _day_key(user_id: str, course_id: str) -> str:
    """当日计数键：ca:costquota:{user}:{course}:{YYYYMMDD}。UTC 日，避免本地时区漂移。"""
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"ca:costquota:{user_id}:{course_id}:{day}"


async def accrue_cost(user_id: str, course_id: str, cost_usd: float) -> None:
    """累加本轮花费到当日 Redis 计数（best-effort）。

    loop.py 在 ``estimate_cost`` 后调用。``cost_usd<=0`` 或 user_id 空 或 Redis 不可用→
    静默跳过。incrbyfloat 原子累加，跨 worker 共享同一累计值。
    """
    if not user_id or cost_usd <= 0:
        return
    try:
        # 复用 cache.py 的共享连接池（唯一池访问器，下划线名但同项目内有意复用）
        from core.db.cache import _get_pool
        key = _day_key(user_id, course_id)
        r = _get_pool()
        await r.incrbyfloat(key, cost_usd)
        await r.expire(key, _KEY_TTL)
    except Exception:
        logger.debug(
            "cost_quota accrue failed user=%s course=%s", user_id, course_id, exc_info=True
        )


async def check_quota(user_id: str, course_id: str) -> tuple[bool, float, float]:
    """返回 ``(over_budget, used_usd, budget_usd)``。

    读当日已用花费与 ``settings.cost_quota.daily_budget_usd`` 比对。Redis 不可用→
    ``(False, 0.0, budget)`` 即放行（配额模块自身的故障绝不误伤业务）。
    """
    budget = float(get_settings().cost_quota.daily_budget_usd)
    if not user_id or budget <= 0:
        return (False, 0.0, budget)
    try:
        from core.db.cache import _get_pool
        raw = await _get_pool().get(_day_key(user_id, course_id))
        used = float(raw) if raw else 0.0
    except Exception:
        logger.debug(
            "cost_quota check failed user=%s course=%s", user_id, course_id, exc_info=True
        )
        return (False, 0.0, budget)
    return (used >= budget, round(used, 6), budget)


async def reconcile_quota_from_db() -> int:
    """启动时从 llm_usage_daily 回填当日 Redis 配额键（SETNX，不覆盖已有值）。

    防 Redis 失联后配额静默重置：Redis 丢数据后当日 key 缺失，check_quota 读到 0 -> 超预算
    用户被错误放行。本函数从 DB 日汇总（rollup_usage 每小时重算，滞后≤1h）回填缺失键，使
    配额近似恢复。已存在的键不覆盖（Redis 是正常运行时的权威源）。best-effort，返回回填条数。
    """
    if not get_settings().cost_quota.enabled:
        return 0
    try:
        from core.db.cache import _get_pool
        from core.db.database import AsyncSessionLocal, LlmUsageDaily
        from sqlalchemy import func, select

        r = _get_pool()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(
                    LlmUsageDaily.user_id,
                    LlmUsageDaily.course_id,
                    func.sum(LlmUsageDaily.cost_usd).label("spent"),
                )
                .where(LlmUsageDaily.day == day)
                .group_by(LlmUsageDaily.user_id, LlmUsageDaily.course_id)
            )).all()
        n = 0
        for row in rows:
            if not row.user_id or not (row.spent or 0) > 0:
                continue
            # nx=True：仅当 key 不存在时回填，绝不覆盖 Redis 已累积的权威值
            if await r.set(_day_key(row.user_id, row.course_id or ""),
                           str(round(float(row.spent), 6)),
                           ex=_KEY_TTL, nx=True):
                n += 1
        return n
    except Exception:
        logger.debug("cost_quota reconcile failed", exc_info=True)
        return 0
