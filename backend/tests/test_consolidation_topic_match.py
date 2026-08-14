"""阶段1旧口径收口：consolidate 已改走 LLM 判定路径（阶段C），本文件保留核心回归：

- 主题命中后 mastery 行 kp_id=topic_id 直写（经 judge 路径折叠落库）
- 无主题命中 → 不写 mastery、不调判定，episode 照常 done
- 词表/extract_knowledge 已退出掌握度主路径（infer_signal 降级 LF，
  should_route_error_llm 已删——判定器一段一次调用，无第二次可省）
"""
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import consolidation
from core.memory.judge import Judgement, JudgeResult


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed_episode(user_id="u1", course_id="c1", user_msg="", assistant_msg=""):
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id=course_id, topic_id="ohm", label="欧姆定律"))
        s.add(
            MemoryEpisode(
                user_id=user_id, course_id=course_id, session_id="s1", turn_id="t1",
                user_msg=user_msg, assistant_msg=assistant_msg, status="pending",
            )
        )
        await s.commit()


async def _run_consolidate(course_id="c1", user_id="u1"):
    from core.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        return await consolidation.consolidate(s, user_id, course_id)


def _judge_poor():
    return JudgeResult(
        judgements=[Judgement("ohm", "poor", 0.9, "没看懂")],
        abstained=[], error_patterns=[], dropped_invalid_topic=0, dropped_bad_quote=0,
    )


async def test_consolidate_writes_topic_id_via_judge(db):
    """主题命中 + 判定 poor → kp_id=topic_id 直写（观测折叠路径）。"""
    await _seed_episode(user_msg="欧姆定律公式是什么", assistant_msg="U=IR")
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=_judge_poor())),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        await _run_consolidate()

    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        row = (
            (await s.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == "u1",
                    KnowledgeMastery.course_id == "c1",
                )
            )).scalars().first()
        )
    assert row is not None
    assert row.kp_id == "ohm"  # 闭集直写，不再哈希
    assert row.label == "欧姆定律"
    assert row.n_obs == 1 and row.theta < 0  # 一次 poor → theta 下移


async def test_consolidate_error_patterns_closed_ids(db):
    """判定器返回 error_patterns（闭集 related_topic_ids）→ 直写 error_graph。"""
    await _seed_episode(user_msg="不对，应该是串联分压", assistant_msg="你说得对")
    err_result = JudgeResult(
        judgements=[Judgement("ohm", "poor", 0.9, "不对，应该是串联分压")],
        abstained=[],
        error_patterns=[{"label": "分压接反", "related_topic_ids": ["ohm"],
                         "evidence_quote": "不对，应该是串联分压"}],
        dropped_invalid_topic=0, dropped_bad_quote=0,
    )
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=err_result)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.graph_memory.merge_and_save_graphs", AsyncMock()) as mock_merge,
    ):
        await _run_consolidate()

    merged = mock_merge.call_args[0][2]
    assert "error_patterns" in merged and "knowledge_points" not in merged
    assert merged["error_patterns"][0]["related_topic_ids"] == ["ohm"]


async def test_consolidate_no_hits_no_mastery(db):
    """无主题命中 → 不写 mastery、不调判定，episode 照常 done。"""
    await _seed_episode(user_msg="实验报告几点交", assistant_msg="下周五前")
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])),
        patch("core.memory.judge.judge_segment", AsyncMock()) as mock_judge,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run_consolidate()

    assert result["promoted"] == 1
    assert mock_judge.await_count == 0
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1")
        )).scalars().all()
    assert rows == []


def test_should_route_error_llm_removed():
    """should_route_error_llm 已删（判定一段一次调用，无第二次可省）。"""
    from core.memory import topic_match

    assert not hasattr(topic_match, "should_route_error_llm")
