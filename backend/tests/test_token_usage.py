"""LLM 用量统计测试：明细落库（best-effort）+ rollup 幂等 + 聚合查询 + teacher 越权 403。

覆盖 token_usage.py 的四级契约：record_llm_usage 字段正确且 DB 故障不抛、rollup_daily
删后重算幂等、query_usage 各 group_by 维度求和正确、teacher 端点归属隔离（非 owner → 403）。
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select


@pytest.mark.asyncio
async def test_record_llm_usage_persists_fields(client):
    """record_llm_usage 落一行明细，字段（含 usage/rounds/turn_id）正确持久化。"""
    from core.analytics.token_usage import record_llm_usage
    from core.db.database import AsyncSessionLocal, LlmUsageRecord
    from core.observability.cost import TokenUsage

    await record_llm_usage(
        user_id="ru1", course_id="rc1", session_id="rs1", turn_id="rt1",
        mode="quiz", model="deepseek-chat",
        usage=TokenUsage(input_tokens=120, output_tokens=40, cache_read_tokens=15),
        cost_usd=0.05, rounds=4,
    )
    async with AsyncSessionLocal() as db:
        rec = (await db.execute(
            select(LlmUsageRecord).where(LlmUsageRecord.user_id == "ru1")
        )).scalar_one()
    assert rec.course_id == "rc1"
    assert rec.session_id == "rs1"
    assert rec.turn_id == "rt1"
    assert rec.mode == "quiz"
    assert rec.model == "deepseek-chat"
    assert rec.input_tokens == 120
    assert rec.output_tokens == 40
    assert rec.cache_read_tokens == 15
    assert rec.cost_usd == pytest.approx(0.05)
    assert rec.rounds == 4


@pytest.mark.asyncio
async def test_record_llm_usage_best_effort_no_raise(monkeypatch):
    """AsyncSessionLocal 抛异常时 record_llm_usage 不向主链路抛（best-effort 契约）。

    用量统计是横切账单关注点，绝不能阻塞对话主链路——DB 故障只记日志。
    """
    from core.analytics import token_usage as tu
    from core.observability.cost import TokenUsage

    class _BoomSession:
        def __init__(self, *a, **kw):
            raise RuntimeError("db down")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(tu, "AsyncSessionLocal", _BoomSession)
    # 不抛异常即通过
    await tu.record_llm_usage(
        user_id="u", course_id="c", session_id="s", turn_id="t", mode="chat",
        model="m", usage=TokenUsage(1, 1, 0), cost_usd=0.01, rounds=1,
    )


@pytest.mark.asyncio
async def test_rollup_idempotent_and_query_sums(client):
    """rollup_daily 删后重算幂等；query_usage 各维度求和正确、按 cost 降序、范围/课程过滤生效。"""
    from core.analytics.token_usage import query_usage, rollup_daily, _day_range
    from core.db.database import AsyncSessionLocal, LlmUsageRecord

    day = "20260115"
    day_start = _day_range(day)[0]
    # 两用户 × 两模型，控 created_at 落在指定日（直接插明细以确定性控日）
    async with AsyncSessionLocal() as db:
        db.add_all([
            LlmUsageRecord(user_id="u1", course_id="c1", mode="chat", model="m1",
                           input_tokens=100, output_tokens=50, cache_read_tokens=20,
                           cost_usd=0.10, rounds=2, created_at=day_start + 10),
            LlmUsageRecord(user_id="u1", course_id="c1", mode="chat", model="m2",
                           input_tokens=200, output_tokens=60, cache_read_tokens=0,
                           cost_usd=0.20, rounds=1, created_at=day_start + 20),
            LlmUsageRecord(user_id="u2", course_id="c1", mode="quiz", model="m1",
                           input_tokens=300, output_tokens=70, cache_read_tokens=30,
                           cost_usd=0.30, rounds=3, created_at=day_start + 30),
        ])
        await db.commit()

    # rollup：3 个 (user,course,model) 组 → 3 行
    assert await rollup_daily([day]) == 3
    # 幂等：再跑一次仍是 3 行（先删后插，不翻倍）
    assert await rollup_daily([day]) == 3

    # 按 user 聚合
    async with AsyncSessionLocal() as db:
        by_user = await query_usage(db, start=day, end=day, group_by=["user"])
    users = {r["user"]: r for r in by_user["rows"]}
    assert users["u1"]["input_tokens"] == 300        # 100 + 200
    assert users["u1"]["output_tokens"] == 110       # 50 + 60
    assert users["u1"]["call_count"] == 2
    assert users["u2"]["input_tokens"] == 300
    # total 反映全范围（不受 limit/group_by 影响）
    assert by_user["total"]["input_tokens"] == 600    # 100+200+300
    assert by_user["total"]["call_count"] == 3
    assert by_user["total"]["cache_read_tokens"] == 50  # 20+0+30
    assert by_user["latest_day"] == day

    # 按 model 聚合 + cost 降序（m1=0.40 > m2=0.20 → m1 在前）
    async with AsyncSessionLocal() as db:
        by_model = await query_usage(db, start=day, end=day, group_by=["model"])
    assert by_model["rows"][0]["model"] == "m1"
    models = {r["model"]: r for r in by_model["rows"]}
    assert models["m1"]["input_tokens"] == 400        # 100 + 300
    assert models["m2"]["input_tokens"] == 200

    # course_id 过滤：命中 c1 / 不命中 other
    async with AsyncSessionLocal() as db:
        scoped = await query_usage(db, start=day, end=day, group_by=["user"], course_id="c1")
        miss = await query_usage(db, start=day, end=day, group_by=["user"], course_id="other")
    assert scoped["total"]["input_tokens"] == 600
    assert miss["total"]["input_tokens"] == 0
    assert miss["rows"] == []


@pytest.mark.asyncio
async def test_admin_usage_summary_endpoint(client, admin_headers):
    """admin /usage/summary 返回 200 + {total, rows, latest_day} 结构。"""
    from core.analytics.token_usage import rollup_daily, _day_range
    from core.db.database import AsyncSessionLocal, LlmUsageRecord

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    day_start = _day_range(day)[0]
    async with AsyncSessionLocal() as db:
        db.add(LlmUsageRecord(user_id="au1", course_id="ac1", mode="chat", model="am",
                              input_tokens=10, output_tokens=5, cost_usd=0.01,
                              rounds=1, created_at=day_start + 5))
        await db.commit()
    await rollup_daily([day])

    r = await client.get(
        "/api/admin/usage/summary",
        headers=admin_headers,
        params={"group_by": "course", "start": day, "end": day},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"]["input_tokens"] == 10
    assert body["rows"][0]["course"] == "ac1"
    assert body["latest_day"] == day


@pytest.mark.asyncio
async def test_teacher_usage_owner_forbidden(client, admin_headers, course_with_code):
    """非 owner 教师访问他人课程用量 → 403（与其它 analytics 端点同归属隔离）。"""
    course_id = course_with_code["course_id"]
    other_teacher = await _make_teacher(client)

    r = await client.get(
        f"/api/teacher/courses/{course_id}/analytics/usage",
        headers=other_teacher,
    )
    assert r.status_code == 403


async def _make_teacher(client) -> dict:
    """注册一名教师（升级 role=teacher）并返回 auth headers（同 test_teacher_academic_api）。"""
    import os

    from core.db.database import AsyncSessionLocal, User

    username = f"te_{os.urandom(3).hex()}"
    r = await client.post(
        "/api/auth/register",
        json={"username": username, "password": "testpass123", "display_name": "T"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    from sqlalchemy import update
    async with AsyncSessionLocal() as db:
        await db.execute(update(User).where(User.username == username).values(role="teacher"))
        await db.commit()
    return {"Authorization": f"Bearer {token}"}
