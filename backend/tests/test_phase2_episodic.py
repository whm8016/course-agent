"""Phase 2: L3 episodic 层——record_episode 落库 + 幂等 + importance 启发式。

episodic 表取代 Redis buffer：每轮 turn 完成后 INSERT 一行（永不删除，兼巩固 outbox），
(session_id, turn_id) 唯一保证幂等。importance 用零 LLM 启发式供巩固排序/触发。
"""
import pytest

from core.memory import episodic


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


# ── estimate_importance（纯函数，零 LLM）─────────────────────────────────────


def test_importance_short_acknowledgment_low():
    """短确认「好的」：只有长度分，importance 低（巩固时会跳过）。"""
    score = episodic.estimate_importance(user_message="好的", agent_output="不客气")
    assert score < 0.1


def test_importance_question_signal_removed():
    """问句不再单独加分：课场景几乎句句是问句，+0.25 恒真曾让单轮打满巩固阈值。"""
    long_ans = "导数是变化率" * 50
    base = episodic.estimate_importance(user_message="导数定义如下", agent_output=long_ans)
    q = episodic.estimate_importance(user_message="什么是导数？", agent_output=long_ans)
    assert q == base  # 同长度下问句与陈述句同分


def test_importance_typical_rag_turn_scale():
    """刻度回归：典型实质检索问答（问句 + rag + 中等长度回答）单轮 ≈0.25~0.5，
    不再一两轮打满 1.2 巩固阈值（攒 3~4 轮 promote 一次）。"""
    score = episodic.estimate_importance(
        user_message="什么是欧姆定律？",
        agent_output="欧姆定律指出导体中电流与电压成正比、与电阻成反比。" * 20,
        tools_used=("rag",),
    )
    assert 0.2 <= score <= 0.5


def test_importance_error_signal_boost():
    """纠错/困惑词加 0.25（掌握度关键证据）。"""
    score = episodic.estimate_importance(
        user_message="我这里算错了，为什么不对", agent_output="." * 100
    )
    assert score >= 0.25


def test_importance_deep_mode_boost():
    """quiz/deep_* 模式比 chat 加 0.1。"""
    chat = episodic.estimate_importance(user_message="x", agent_output="y")
    quiz = episodic.estimate_importance(user_message="x", agent_output="y", mode="quiz")
    assert quiz > chat


def test_importance_tools_used_boost():
    """触发 rag/solve/web_search 加 0.2。"""
    no_tool = episodic.estimate_importance(user_message="m", agent_output="n")
    with_tool = episodic.estimate_importance(
        user_message="m", agent_output="n", tools_used=["rag"]
    )
    assert with_tool > no_tool


def test_importance_single_turn_ceiling():
    """全信号叠满的单轮上限 = 0.85（长度0.3 + 纠错0.25 + 工具0.2 + 深度模式0.1）。

    删问句信号后单轮永远够不到 1.2 巩固阈值（promote 至少 2 轮起步）；
    min(1.0) 封顶保留为防御（未来加信号仍生效）。
    """
    score = episodic.estimate_importance(
        user_message="我这里算错了，需要纠正" + "长" * 3000,
        agent_output="长" * 3000,
        mode="quiz",
        tools_used=["rag", "solve"],
    )
    assert score == 0.85


# ── record_episode（DB 集成）─────────────────────────────────────────────────


async def test_record_episode_inserts_pending(db):
    from sqlalchemy import select

    from core.db.database import AsyncSessionLocal, MemoryEpisode

    ok = await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
        user_msg="什么是导数", assistant_msg="导数是变化率", mode="chat",
    )
    assert ok is True
    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    ep = rows[0]
    assert ep.status == "pending"
    assert ep.session_id == "s1" and ep.turn_id == "t1"
    assert ep.importance > 0  # 长度项（问句信号已删）


async def test_record_episode_idempotent_on_duplicate(db):
    """同一 (session_id, turn_id) 重放只写一次。"""
    from sqlalchemy import func, select

    from core.db.database import AsyncSessionLocal, MemoryEpisode

    await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
        user_msg="q", assistant_msg="a",
    )
    ok = await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
        user_msg="q", assistant_msg="a",
    )
    assert ok is False  # 重复 → 幂等跳过
    async with AsyncSessionLocal() as s:
        n = (
            await s.execute(
                select(func.count())
                .select_from(MemoryEpisode)
                .where(MemoryEpisode.session_id == "s1")
            )
        ).scalar()
    assert n == 1


async def test_record_episode_skips_empty_and_no_user(db):
    assert await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
        user_msg="", assistant_msg="",
    ) is False
    assert await episodic.record_episode(
        user_id="", course_id="c1", session_id="s1", turn_id="t1",
        user_msg="q", assistant_msg="a",
    ) is False


async def test_record_episode_generates_turn_id_when_missing(db):
    """turn_id 为空时自动生成（保证 (session_id, turn_id) 唯一约束不撞空值）。"""
    from sqlalchemy import func, select

    from core.db.database import AsyncSessionLocal, MemoryEpisode

    ok1 = await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="", turn_id="",
        user_msg="q1", assistant_msg="a1",
    )
    ok2 = await episodic.record_episode(
        user_id="u1", course_id="c1", session_id="", turn_id="",
        user_msg="q2", assistant_msg="a2",
    )
    assert ok1 is True and ok2 is True
    async with AsyncSessionLocal() as s:
        n = (
            await s.execute(
                select(func.count())
                .select_from(MemoryEpisode)
                .where(MemoryEpisode.user_id == "u1")
            )
        ).scalar()
    assert n == 2
