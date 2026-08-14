"""阶段4：班级收缩（经验贝叶斯 k=3）——观测少时 risk_used 向全班均值靠。

口径（plan 阶段4验收）：20 人多数 risk=0.2、新生 n=1 risk=0.9 → 收缩后不入 C1；
同人 n=10 → risk_used≈0.74 入 C1。
注：plan 给的 0.375 把班级均值近似成 0.2；精确口径（AVG 含本人，21 人均值≈0.2333）
算得 n=1 → 0.40，两者同样低于 τ 闸，验收性质不变。
"""
import time

import pytest

from core.memory.proactive import decide_stitch

_PREREQ = {"ohm"}


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed_cohort(course_id: str, n_class: int, class_risk: float):
    """n_class 个同学在 ohm 上观测过 class_risk；返回写入函数给主角补行。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, KnowledgeMastery

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id=course_id, topic_id="ohm", label="欧姆定律"))
        now = time.time()
        for i in range(n_class):
            s.add(KnowledgeMastery(
                user_id=f"classmate{i}", course_id=course_id, kp_id="ohm",
                label="欧姆定律", risk=class_risk, observation_count=5,
                first_observed_at=now, last_observed_at=now - 30 * 86400,
            ))
        await s.commit()


async def _seed_target(user_id: str, course_id: str, obs_count: int, risk: float):
    from core.db.database import AsyncSessionLocal, KnowledgeMastery

    now = time.time()
    async with AsyncSessionLocal() as s:
        s.add(KnowledgeMastery(
            user_id=user_id, course_id=course_id, kp_id="ohm", label="欧姆定律",
            risk=risk, observation_count=obs_count,
            first_observed_at=now, last_observed_at=now - 30 * 86400,  # t=30 天
        ))
        await s.commit()


async def _read(user_id: str, course_id: str):
    from core.db.database import AsyncSessionLocal
    from core.memory.course_topic_store import get_course_topic_map
    from core.memory.proactive import _read_mastery

    async with AsyncSessionLocal() as s:
        topic_map = await get_course_topic_map(course_id, s)
        return await _read_mastery(user_id, course_id, s, topic_map)


async def test_cohort_shrink_n1_pulls_to_class_mean(db):
    await _seed_cohort("c1", n_class=20, class_risk=0.2)
    await _seed_target("fresh", "c1", obs_count=1, risk=0.9)
    m = await _read("fresh", "c1")
    # 班级均值含本人 = (20*0.2+0.9)/21 ≈ 0.2333；n=1,k=3 → 0.25*0.9+0.75*0.2333 ≈ 0.40
    assert m["ohm"]["risk_used"] == pytest.approx(0.4, abs=0.01)
    # 不入 C1：p_need = 0.4·p_forget(t=30)≈0.399 < τ≈0.455
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is False
    assert brief.reason == "below_tau"


async def test_cohort_shrink_n10_trusts_personal(db):
    await _seed_cohort("c2", n_class=20, class_risk=0.2)
    await _seed_target("senior", "c2", obs_count=10, risk=0.9)
    m = await _read("senior", "c2")
    # (10/13)*0.9 + (3/13)*0.2333 ≈ 0.746
    assert m["ohm"]["risk_used"] == pytest.approx(0.74, abs=0.02)
    brief = decide_stitch("kcl", _PREREQ, m, set())
    assert brief.should_stitch is True
    assert brief.reason == "prereq_gap"


async def test_cohort_degenerate_only_self(db):
    """该 topic 只有自己观测 → cohort_mean=个人值，公式退化成个人 risk。"""
    from core.db.database import AsyncSessionLocal, CourseTopic

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c3", topic_id="ohm", label="欧姆定律"))
        await s.commit()
    await _seed_target("lonely", "c3", obs_count=3, risk=0.8)
    m = await _read("lonely", "c3")
    assert m["ohm"]["risk_used"] == pytest.approx(0.8, abs=1e-6)
