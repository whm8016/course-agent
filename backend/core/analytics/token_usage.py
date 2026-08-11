"""LLM 用量统计：明细落库 + 日汇总（读模型）+ 聚合查询。

两级存储（对标 LiteLLM SpendLogs/DailyUserSpend + Langfuse 摄取时算成本）：
- ``record_llm_usage`` 写 ``llm_usage_records`` 明细（每个 run_agent_loop 一行，best-effort）。
- ``rollup_daily`` 把指定天从明细「删后重算」灌进 ``llm_usage_daily``，天然幂等。
- ``query_usage`` 只读 ``llm_usage_daily`` 聚合表，SQL 形状照搬 LiteLLM ``/spend/report``
  （``SUM(...) GROUP BY <维度> ORDER BY cost DESC``）。

**为什么写库全 best-effort**：用量统计是横切账单关注点，绝不能阻塞对话主链路。异常只
``logger.warning``，对齐项目已有的 ``record_learning_event``（events.py）。独立 ``AsyncSessionLocal``
会话，与请求主会话隔离，写失败不影响事务。

**为什么 rollup 用 epoch 边界而非 SQL 日期函数**：明细 ``created_at`` 是 epoch 秒，把 epoch→
"YYYYMMDD" 在 PG（``to_timestamp``）与 SQLite（``datetime(x,'unixepoch')``）上语法不同。
改为在 Python 算出当天的 ``[start, end)`` epoch 区间、用 ``WHERE created_at >= start AND
created_at < end`` 过滤，方言无关且天然正确。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import AsyncSessionLocal, LlmUsageDaily, LlmUsageRecord
from core.observability.cost import TokenUsage

logger = logging.getLogger(__name__)

# query_usage 的合法分组维度 → daily 表列（白名单，杜绝任意列注入）
_DIM_COLUMNS = {
    "day": LlmUsageDaily.day,
    "user": LlmUsageDaily.user_id,
    "course": LlmUsageDaily.course_id,
    "model": LlmUsageDaily.model,
}
_DAY_SECONDS = 86400.0


def _day_range(day: str) -> tuple[float, float]:
    """"YYYYMMDD" → 该 UTC 自然日的 ``[start_epoch, end_epoch)``（rollup 扫明细用）。"""
    start = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    return start, start + _DAY_SECONDS


def _today_yesterday() -> list[str]:
    """今日 + 昨日（UTC "YYYYMMDD"）：cron 重算两天的聚合，覆盖跨日边界的滞后写入。"""
    now = datetime.now(timezone.utc)
    return [now.strftime("%Y%m%d"), (now - timedelta(days=1)).strftime("%Y%m%d")]


async def record_llm_usage(
    *,
    user_id: str,
    course_id: str,
    session_id: str,
    turn_id: str,
    mode: str,
    model: str,
    usage: TokenUsage,
    cost_usd: float,
    rounds: int,
) -> None:
    """插一行 LLM 用量明细。best-effort：DB 不可用只记日志，绝不抛给主链路。

    cost_usd 在摄取时按当时价目表快照落库（对齐 Langfuse：日后改价目表不篡改历史账）。
    """
    try:
        async with AsyncSessionLocal() as db:
            db.add(
                LlmUsageRecord(
                    user_id=str(user_id or ""),
                    course_id=str(course_id or ""),
                    session_id=str(session_id or ""),
                    turn_id=str(turn_id or ""),
                    mode=str(mode or "chat"),
                    model=str(model or ""),
                    input_tokens=int(usage.input_tokens),
                    output_tokens=int(usage.output_tokens),
                    cache_read_tokens=int(usage.cache_read_tokens),
                    cost_usd=float(cost_usd or 0.0),
                    rounds=int(rounds or 0),
                )
            )
            await db.commit()
    except Exception:
        logger.warning(
            "record_llm_usage failed model=%s mode=%s", model, mode, exc_info=True
        )


async def rollup_daily(days: list[str]) -> int:
    """把指定天从明细重算灌进日汇总：每天先删 daily 旧行，再 GROUP BY 重插。

    幂等：整天重算，连跑两次结果一致（避开 PG/SQLite 双方言 ON CONFLICT）。
    返回新插入的聚合行数；失败 best-effort 返回 0。
    """
    inserted = 0
    try:
        async with AsyncSessionLocal() as db:
            for day in days:
                start, end = _day_range(day)
                # 先删该天已有聚合行（整删整插 = 幂等）
                await db.execute(delete(LlmUsageDaily).where(LlmUsageDaily.day == day))
                # 从明细按 (user, course, model) 聚合（SQL 形状同 LiteLLM /spend/report）
                rows = (
                    await db.execute(
                        select(
                            LlmUsageRecord.user_id,
                            LlmUsageRecord.course_id,
                            LlmUsageRecord.model,
                            func.sum(LlmUsageRecord.input_tokens).label("input_tokens"),
                            func.sum(LlmUsageRecord.output_tokens).label("output_tokens"),
                            func.sum(LlmUsageRecord.cache_read_tokens).label("cache_read_tokens"),
                            func.sum(LlmUsageRecord.cost_usd).label("cost_usd"),
                            func.count().label("call_count"),
                        )
                        .where(
                            LlmUsageRecord.created_at >= start,
                            LlmUsageRecord.created_at < end,
                        )
                        .group_by(
                            LlmUsageRecord.user_id,
                            LlmUsageRecord.course_id,
                            LlmUsageRecord.model,
                        )
                    )
                ).all()
                now = time.time()
                for row in rows:
                    db.add(
                        LlmUsageDaily(
                            day=day,
                            user_id=row.user_id,
                            course_id=row.course_id,
                            model=row.model,
                            input_tokens=row.input_tokens or 0,
                            output_tokens=row.output_tokens or 0,
                            cache_read_tokens=row.cache_read_tokens or 0,
                            cost_usd=row.cost_usd or 0.0,
                            call_count=row.call_count or 0,
                            updated_at=now,
                        )
                    )
                    inserted += 1
            await db.commit()
    except Exception:
        logger.warning("rollup_daily failed days=%s", days, exc_info=True)
        return 0
    return inserted


async def purge_old_records(retention_days: int) -> int:
    """删过期明细（默认 90 天），日汇总永久保留（账单历史）。返回删除行数；失败返回 0。"""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - _DAY_SECONDS * retention_days
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                delete(LlmUsageRecord).where(LlmUsageRecord.created_at < cutoff)
            )
            await db.commit()
            return int(result.rowcount or 0)
    except Exception:
        logger.warning("purge_old_records failed", exc_info=True)
        return 0


async def query_usage(
    db: AsyncSession,
    *,
    start: str,
    end: str,
    group_by: list[str],
    course_id: str | None = None,
    user_id: str | None = None,
    limit: int = 100,
) -> dict:
    """读 llm_usage_daily 聚合表，按维度分组求和。返回 ``{total, rows, latest_day}``。

    - ``start`` / ``end``：含闭区间，"YYYYMMDD" 日串（字典序比较 == 日期比较）。
    - ``group_by``：维度名列表，子集 {day, user, course, model}（白名单）。
    - ``course_id`` / ``user_id``：可选过滤（teacher 端强制传 course_id 做归属隔离）。
    - ``total``：全范围（不限 limit）的汇总，便于看板显示「合计」与占比。
    - ``latest_day``：当前数据覆盖到的最近一天（max(day)），供前端标「数据截至」。
    """
    dims = [g for g in group_by if g in _DIM_COLUMNS] or ["day"]
    dim_labels = [_DIM_COLUMNS[g].label(g) for g in dims]

    base_filters = [
        LlmUsageDaily.day >= start,
        LlmUsageDaily.day <= end,
    ]
    if course_id is not None:
        base_filters.append(LlmUsageDaily.course_id == course_id)
    if user_id is not None:
        base_filters.append(LlmUsageDaily.user_id == user_id)
    rows = (
        await db.execute(
            select(
                *dim_labels,
                func.sum(LlmUsageDaily.input_tokens).label("input_tokens"),
                func.sum(LlmUsageDaily.output_tokens).label("output_tokens"),
                func.sum(LlmUsageDaily.cache_read_tokens).label("cache_read_tokens"),
                func.sum(LlmUsageDaily.cost_usd).label("cost_usd"),
                func.sum(LlmUsageDaily.call_count).label("call_count"),
            )
            .where(*base_filters)
            .group_by(*dim_labels)
            .order_by(func.sum(LlmUsageDaily.cost_usd).desc())
            .limit(limit)
        )
    ).all()

    label_names = dims
    row_dicts = []
    for row in rows:
        d: dict = {}
        for name in label_names:
            d[name] = getattr(row, name)
        d.update(
            input_tokens=row.input_tokens or 0,
            output_tokens=row.output_tokens or 0,
            cache_read_tokens=row.cache_read_tokens or 0,
            cost_usd=round(float(row.cost_usd or 0.0), 6),
            call_count=row.call_count or 0,
        )
        row_dicts.append(d)

    # 全范围合计（不受 limit / group_by 影响）
    total_row = (
        await db.execute(
            select(
                func.sum(LlmUsageDaily.input_tokens).label("input_tokens"),
                func.sum(LlmUsageDaily.output_tokens).label("output_tokens"),
                func.sum(LlmUsageDaily.cache_read_tokens).label("cache_read_tokens"),
                func.sum(LlmUsageDaily.cost_usd).label("cost_usd"),
                func.sum(LlmUsageDaily.call_count).label("call_count"),
                func.max(LlmUsageDaily.day).label("latest_day"),
            ).where(*base_filters)
        )
    ).one()

    return {
        "total": {
            "input_tokens": total_row.input_tokens or 0,
            "output_tokens": total_row.output_tokens or 0,
            "cache_read_tokens": total_row.cache_read_tokens or 0,
            "cost_usd": round(float(total_row.cost_usd or 0.0), 6),
            "call_count": total_row.call_count or 0,
        },
        "rows": row_dicts,
        "latest_day": total_row.latest_day,
    }
