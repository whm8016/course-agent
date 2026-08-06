"""Phase 4a: L3 mastery 层——追加观测不覆盖 + course 隔离 + 读时衰减 + consolidate 双写。

覆盖：append_mastery（新建/追加/course 隔离）、get_mastery_context（空/薄弱点/低风险过滤）、
consolidate 双写（mem0 + graph + mastery 单次抽取喂三者）。
"""
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import mastery


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


# ── append_mastery ──────────────────────────────────────────────────────────


async def test_append_mastery_creates_row(db):
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    kps = [{"label": "导数", "entity_id": "ent1", "mastery_delta": -0.1, "risk_delta": 0.2}]
    async with AsyncSessionLocal() as s:
        n = await mastery.append_mastery(s, "u1", "c1", kps, "ep1")
    assert n == 1
    async with AsyncSessionLocal() as s:
        row = (
            (await s.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == "u1",
                    KnowledgeMastery.course_id == "c1",
                    KnowledgeMastery.kp_id == "ent1",
                )
            )).scalars().first()
        )
    assert row is not None
    assert row.observation_count == 1
    assert row.mastery < 0.5  # 0.5 + (-0.1)
    assert row.risk > 0.5  # 0.5 + 0.2
    assert row.evidence_episode_ids == ["ep1"]


async def test_append_mastery_appends_observation_not_overwrite(db):
    """同 kp 第二次观测：count 累加、evidence 追加（不覆盖）。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    kps = [{"label": "导数", "entity_id": "ent1", "mastery_delta": 0.1, "risk_delta": 0.1}]
    async with AsyncSessionLocal() as s:
        await mastery.append_mastery(s, "u1", "c1", kps, "ep1")
        await mastery.append_mastery(s, "u1", "c1", kps, "ep2")
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == "u1", KnowledgeMastery.course_id == "c1"
                )
            )).scalars().all()
        )
    assert len(rows) == 1  # 不新增行（追加观测）
    assert rows[0].observation_count == 2
    assert rows[0].evidence_episode_ids == ["ep1", "ep2"]


async def test_append_mastery_course_isolation(db):
    """同 kp_id、不同 course → 两条独立行（修 users.knowledge_graph 跨课程污染）。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    kps = [{"label": "导数", "entity_id": "ent1", "mastery_delta": 0.0, "risk_delta": 0.0}]
    async with AsyncSessionLocal() as s:
        await mastery.append_mastery(s, "u1", "c1", kps, "ep1")
        await mastery.append_mastery(s, "u1", "c2", kps, "ep2")
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1")))
            .scalars().all()
        )
    assert len(rows) == 2
    assert {r.course_id for r in rows} == {"c1", "c2"}


# ── get_mastery_context ─────────────────────────────────────────────────────


async def test_get_mastery_context_empty(db):
    from core.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        ctx = await mastery.get_mastery_context(s, "u1", "c1")
    assert ctx == ""


async def test_get_mastery_context_lists_weak_points(db):
    """高 risk 知识点进注入文本；低 risk（<0.5）不出现。"""
    from core.db.database import AsyncSessionLocal

    kps_weak = [{"label": "积分", "entity_id": "e1", "mastery_delta": -0.3, "risk_delta": 0.4}]
    kps_strong = [{"label": "加减法", "entity_id": "e2", "mastery_delta": 0.4, "risk_delta": -0.4}]
    async with AsyncSessionLocal() as s:
        await mastery.append_mastery(s, "u1", "c1", kps_weak, "ep1")
        await mastery.append_mastery(s, "u1", "c1", kps_strong, "ep2")
        ctx = await mastery.get_mastery_context(s, "u1", "c1")
    assert "积分" in ctx  # 薄弱点出现
    assert "加减法" not in ctx  # 低风险不出现
    assert len(ctx) <= 300


# ── consolidate 双写（mem0 + graph + mastery 单次抽取）──────────────────────


async def test_consolidate_writes_mastery_and_marks_done(db):
    """consolidate 成功：mem0 升格 + graph/mastery 双写 + episode 标 done。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery, MemoryEpisode
    from sqlalchemy import select

    # seed pending episode
    async with AsyncSessionLocal() as s:
        s.add(MemoryEpisode(
            user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
            user_msg="什么是导数", assistant_msg="变化率", status="pending",
        ))
        await s.commit()

    extracted = {"knowledge_points": [{"label": "导数", "entity_id": "ent1", "mastery_delta": -0.1, "risk_delta": 0.2}], "error_patterns": []}
    with patch("core.memory.mem0_client.get_memory") as gm, \
         patch("core.memory.graph_memory.extract_knowledge", AsyncMock(return_value=extracted)):
        gm.return_value.add = AsyncMock(return_value={"results": []})
        from core.memory.consolidation import consolidate
        async with AsyncSessionLocal() as s:
            result = await consolidate(s, "u1", "c1")

    assert result == {"claimed": 1, "promoted": 1}
    async with AsyncSessionLocal() as s:
        ep = (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1"))).scalars().first()
        km = (await s.execute(select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1"))).scalars().all()
    assert ep.status == "done"  # mem0 成功 → done
    assert len(km) == 1 and km[0].kp_id == "ent1"  # mastery 双写
