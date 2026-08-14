"""阶段1验收：consolidate 写侧闭集化——kp_id=topic_id 直写、零 LLM 抽取、错题才路由。

口径（plan 阶段1验收）：
- mock match_topics 返回 [("ohm", 0.9)] → KnowledgeMastery.kp_id == "ohm" 且
  extract_knowledge 未被调用（中性对话零 LLM）
- 纠正语（"不对，应该是…"）→ extract_knowledge 被调用一次（仅 error 路由）
"""
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import consolidation


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


async def test_consolidate_writes_topic_id_no_llm(db):
    """中性对话 + 主题命中 → kp_id=topic_id 直写，extract_knowledge 零调用。"""
    await _seed_episode(user_msg="欧姆定律公式是什么", assistant_msg="U=IR")
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.graph_memory.extract_knowledge", AsyncMock()) as mock_extract,
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
    assert mock_extract.await_count == 0  # 中性轮零 LLM


async def test_consolidate_routes_error_llm_on_correction(db):
    """纠正语 → extract_knowledge 恰好一次（只取 error_patterns）。"""
    await _seed_episode(user_msg="不对，应该是串联分压", assistant_msg="你说得对")
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch(
            "core.memory.graph_memory.extract_knowledge",
            AsyncMock(return_value={"error_patterns": [{"label": "分压接反"}], "knowledge_points": []}),
        ) as mock_extract,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.graph_memory.merge_and_save_graphs", AsyncMock()) as mock_merge,
    ):
        await _run_consolidate()

    assert mock_extract.await_count == 1
    # merge 收到的只有 error_patterns（knowledge_points 被丢弃，防 id 漂移）
    merged = mock_merge.call_args[0][2]
    assert "error_patterns" in merged and "knowledge_points" not in merged


async def test_consolidate_no_hits_no_mastery(db):
    """无主题命中 → 不写 mastery、不调 LLM，episode 照常 done。"""
    await _seed_episode(user_msg="实验报告几点交", assistant_msg="下周五前")
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])),
        patch("core.memory.graph_memory.extract_knowledge", AsyncMock()) as mock_extract,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run_consolidate()

    assert result["promoted"] == 1
    assert mock_extract.await_count == 0
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1")
        )).scalars().all()
    assert rows == []
