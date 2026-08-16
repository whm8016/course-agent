"""阶段0：读侧断链修复——_read_mastery 只信任落在 course_topic.topic_id 集内的 kp_id。

回归背景：写侧 kp_id 由 entity_id 或 label 哈希生成（恒非空），旧代码 `r.kp_id or
topic_map.get(...)` 的 `or` 短路令 label 兜底永不生效 → mastery_map 键是与 course_topic
对不上的哈希串，门控的前置闭包交集恒空（断链）。修复后：kp_id 不在 topic_id 集 → 走
label 归一化对齐；对齐不上 → 丢弃该行。
"""
import time

import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed_mastery(user_id: str, course_id: str, kp_id: str, label: str,
                        risk: float = 0.8, theta: float | None = None):
    from core.db.database import AsyncSessionLocal, KnowledgeMastery

    async with AsyncSessionLocal() as s:
        s.add(
            KnowledgeMastery(
                user_id=user_id,
                course_id=course_id,
                kp_id=kp_id,
                label=label,
                risk=risk,
                theta=theta if theta is not None else 0.0,
                n_obs=2,
                observation_count=2,
                last_observed_at=time.time(),
            )
        )
        await s.commit()


async def _seed_topic(course_id: str, topic_id: str, label: str):
    from core.db.database import AsyncSessionLocal, CourseTopic

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id=course_id, topic_id=topic_id, label=label))
        await s.commit()


async def _topic_map(course_id: str) -> dict[str, str]:
    from core.db.database import AsyncSessionLocal
    from core.memory.course_topic_store import get_course_topic_map

    async with AsyncSessionLocal() as s:
        return await get_course_topic_map(course_id, s)


async def test_read_mastery_bridges_broken_kp_id_via_label(db):
    """kp_id 是历史哈希串（不在 topic_id 集）+ label 能对上 → mastery_map 键是 topic_id。

    阶段D：risk 由折叠状态算（theta/b_topic），种子 theta=logit(0.2)（risk 0.8）。"""
    import math

    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import _read_mastery

    await _seed_topic("c1", "ohm", "欧姆定律")
    await _seed_mastery(
        "u_broken", "c1", kp_id="kp:deadbeef", label="欧姆定律",
        theta=math.log(0.2 / 0.8),  # logit(0.2) → mastery 0.2 / risk 0.8
    )

    async with AsyncSessionLocal() as s:
        m = await _read_mastery("u_broken", "c1", s, await _topic_map("c1"))
    assert "ohm" in m
    assert "kp:deadbeef" not in m
    assert m["ohm"]["label"] == "欧姆定律"
    assert m["ohm"]["risk"] == pytest.approx(0.8, abs=1e-3)


async def test_read_mastery_keeps_valid_kp_id(db):
    """kp_id 本就是 topic_id（新写侧直写口径）→ 原样保留，不走 label 对齐。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import _read_mastery

    await _seed_topic("c2", "ohm", "欧姆定律")
    await _seed_mastery("u_new", "c2", kp_id="ohm", label="欧姆定律")

    async with AsyncSessionLocal() as s:
        m = await _read_mastery("u_new", "c2", s, await _topic_map("c2"))
    assert set(m) == {"ohm"}


async def test_read_mastery_drops_unalignable_row(db):
    """kp_id 对不上 + label 也对不上（课程外/废知识点）→ 整行丢弃，不产生幽灵键。"""
    from core.db.database import AsyncSessionLocal
    from core.memory.proactive import _read_mastery

    await _seed_topic("c3", "ohm", "欧姆定律")
    await _seed_mastery("u_ghost", "c3", kp_id="ent:weird123", label="量子纠缠")

    async with AsyncSessionLocal() as s:
        m = await _read_mastery("u_ghost", "c3", s, await _topic_map("c3"))
    assert m == {}


async def test_prereq_predecessors_filters_non_prerequisite_relation(db):
    """门控回溯只沿先修类边（prerequisite / prerequisite_inferred）。

    防御契约：未来往同表加 related 等非先修 relation 时，不得被 BFS 当成
    「必须先学」混入前置闭包。"""
    from core.db.database import AsyncSessionLocal, CourseTopicEdge
    from core.memory.course_topic_store import get_prereq_predecessors

    async with AsyncSessionLocal() as s:
        for src, rel in (("a", "prerequisite"), ("c", "prerequisite_inferred"), ("d", "related")):
            s.add(CourseTopicEdge(course_id="c9", src_topic_id=src, dst_topic_id="target", relation=rel))
        await s.commit()

    async with AsyncSessionLocal() as s:
        preds = await get_prereq_predecessors("target", "c9", s)
    assert preds == {"a", "c"}
