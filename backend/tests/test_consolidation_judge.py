"""阶段C验收：consolidate 走 LLM 判定——一段一次调用、观测落库、预算与降级。

口径（plan 阶段C验收）：
- 一段 4 轮同主题 → judge_segment 恰好一次、mastery_observation 按判定条数写入、
  knowledge_mastery 折叠出 theta
- 判定不可用（judge 返回 None）→ 掌握度路径跳过，episode 回 pending 等下轮；
  mem0 不受影响
- 25 段（超预算 20）→ 判定调用 20 次，超出的段 episode 留 pending
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import consolidation


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def _seed(user_id="u1", course_id="c1", episodes=()):
    """episodes: [(turn_id, user_msg, assistant_msg)]"""
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id=course_id, topic_id="ohm", label="欧姆定律"))
        for tid, u, a in episodes:
            s.add(
                MemoryEpisode(
                    user_id=user_id, course_id=course_id, session_id="s1", turn_id=tid,
                    user_msg=u, assistant_msg=a, status="pending",
                )
            )
        await s.commit()


async def _run(user_id="u1", course_id="c1"):
    from core.db.database import AsyncSessionLocal

    async with AsyncSessionLocal() as s:
        return await consolidation.consolidate(s, user_id, course_id)


_JUDGE_OK = json.dumps({
    "judgements": [
        {"topic_id": "ohm", "level": "poor", "confidence": 0.9,
         "evidence_quote": "欧姆定律我没看懂"},
    ],
    "error_patterns": [],
})


async def test_one_segment_four_turns_one_llm_call(db):
    """一段 4 轮同主题 → judge 一次调用、观测写入、折叠出 theta。"""
    await _seed(episodes=[
        (f"t{i}", "欧姆定律我没看懂" if i == 0 else "还是没懂", "U=IR……")
        for i in range(4)
    ])
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=_mk_result())) as mock_judge,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run()

    assert result == {"claimed": 4, "promoted": 4}
    assert mock_judge.await_count == 1  # 一段一次，不是一轮一次

    from core.db.database import AsyncSessionLocal, KnowledgeMastery, MasteryObservation
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        obs = (await s.execute(
            select(MasteryObservation).where(MasteryObservation.user_id == "u1")
        )).scalars().all()
        km = (await s.execute(
            select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1")
        )).scalars().all()
    assert len(obs) == 1  # 一条判定一条观测
    assert obs[0].source == "llm_judge" and obs[0].outcome == 0.0
    assert len(obs[0].episode_ids) == 4  # 段内 4 条 episode 都记证据
    assert len(km) == 1 and km[0].kp_id == "ohm"
    assert km[0].theta < 0 and km[0].n_obs == 1  # 一次 poor → theta 下移


def _mk_result():
    from core.memory.judge import Judgement, JudgeResult

    return JudgeResult(
        judgements=[Judgement("ohm", "poor", 0.9, "欧姆定律我没看懂")],
        abstained=[], error_patterns=[], dropped_invalid_topic=0, dropped_bad_quote=0,
    )


async def test_judge_unavailable_defers_and_keeps_pending(db):
    """判定不可用（None）→ episode 回 pending 等下轮；mem0 调过不受影响。"""
    await _seed(episodes=[("t1", "欧姆定律我没看懂", "U=IR")])
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=None)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()) as mock_promote,
    ):
        result = await _run()

    assert result["claimed"] == 1
    assert result["promoted"] == 0  # 判定不可用 → 不标 done
    assert mock_promote.await_count == 1  # mem0 升格照常

    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        ep = (await s.execute(select(MemoryEpisode))).scalars().first()
    assert ep.status == "pending"  # 等下轮重判


async def test_budget_overflow_25_segments(db):
    """25 段（超 _JUDGE_SEGMENTS_PER_RUN=20）→ 判定 20 次，多的 5 段留 pending。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        s.add(CourseTopic(course_id="c1", topic_id="kcl", label="基尔霍夫电流定律"))
        # 25 段：ohm/kcl 交替 → 每条 episode 自成一段（主题变更切开）
        for i in range(25):
            s.add(MemoryEpisode(
                user_id="u1", course_id="c1", session_id="s1", turn_id=f"t{i}",
                user_msg=f"问题{i}", assistant_msg=f"回答{i}", status="pending",
            ))
        await s.commit()

    async def _fake_match(text, course_id, db, **kw):
        # 按题面轮次交替主题：i 偶 ohm 奇 kcl
        idx = int(text.split("问题")[1].split("\n")[0])
        return [("ohm" if idx % 2 == 0 else "kcl", 0.9)]

    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(side_effect=_fake_match)),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=_mk_result())) as mock_judge,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run()

    assert mock_judge.await_count == consolidation._JUDGE_SEGMENTS_PER_RUN == 20
    assert result["claimed"] == 25
    assert result["promoted"] == 20
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        statuses = (await s.execute(select(MemoryEpisode.status))).scalars().all()
    assert statuses.count("pending") == 5
    assert statuses.count("done") == 20


async def test_mastery_write_failure_defers(db):
    """_write_mastery_observations 写库失败 → episode 回 pending（非 done）。

    旧 bug：写库失败 rollback 后照样标 done，这段对话的掌握度观测永久丢失。
    """
    await _seed(episodes=[("t1", "欧姆定律我没看懂", "U=IR")])
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=_mk_result())),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.consolidation._write_mastery_observations",
              AsyncMock(side_effect=RuntimeError("db down"))),
    ):
        result = await _run()

    assert result == {"claimed": 1, "promoted": 0}
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        ep = (await s.execute(select(MemoryEpisode))).scalars().first()
    assert ep.status == "pending"  # 等下轮重判重写，不是 done


async def test_judge_path_crash_defers_whole_session(db):
    """judge 路径整体抛错（切段/查 meta 崩）→ 该 session 全部 episode 回 pending。"""
    await _seed(episodes=[
        (f"t{i}", "欧姆定律我没看懂" if i == 0 else "还是没懂", "U=IR……")
        for i in range(3)
    ])
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
        patch("core.memory.consolidation.split_topic_segments",
              side_effect=RuntimeError("segment crash")),
    ):
        result = await _run()

    assert result == {"claimed": 3, "promoted": 0}
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        statuses = (await s.execute(select(MemoryEpisode.status))).scalars().all()
    assert statuses.count("pending") == 3


async def test_judge_budget_shared_across_sessions(db):
    """判定段预算整次 consolidate 共享 20（非 per-session）。

    旧 bug：每 session 各判 20 段，200 条 episode × 10 session → 200 次 LLM，
    正是预算要防的 ARQ 超时。现在跨 session 共享：2 个 session 各 15 段 →
    只判 20 次，剩 10 段留 pending。
    """
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        s.add(CourseTopic(course_id="c1", topic_id="kcl", label="基尔霍夫电流定律"))
        # 2 个 session，各 30 轮交替主题 → 各 30 段（每条 episode 自成一段）
        for si in range(2):
            for i in range(30):
                s.add(MemoryEpisode(
                    user_id="u1", course_id="c1", session_id=f"s{si}", turn_id=f"s{si}t{i}",
                    user_msg=f"问题{si}-{i}", assistant_msg="回答", status="pending",
                ))
        await s.commit()

    async def _fake_match(text, course_id, db, **kw):
        # episode 文本 = f"{user_msg}\n{assistant_msg}"，user_msg = "问题{si}-{i}"
        si, i = map(int, text.split("\n")[0].split("问题")[1].split("-"))
        return [("ohm" if (si + i) % 2 == 0 else "kcl", 0.9)]

    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(side_effect=_fake_match)),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=_mk_result())) as mock_judge,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run()

    assert mock_judge.await_count == consolidation._JUDGE_SEGMENTS_PER_RUN == 20
    assert result["claimed"] == 60
    assert result["promoted"] == 20
    from core.db.database import AsyncSessionLocal, MemoryEpisode
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        statuses = (await s.execute(select(MemoryEpisode.status))).scalars().all()
    assert statuses.count("pending") == 40
    assert statuses.count("done") == 20


async def test_no_topic_no_judge(db):
    """无主题段（match_topics 空）不调判定，episode 照常 done。"""
    await _seed(episodes=[("t1", "实验报告几点交", "下周五前")])
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[])),
        patch("core.memory.judge.judge_segment", AsyncMock()) as mock_judge,
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        result = await _run()

    assert result == {"claimed": 1, "promoted": 1}
    assert mock_judge.await_count == 0


async def test_seg_topics_ranked_by_sim_not_alphabetical(db):
    """段内可判主题按 sim 降序取前 3（P0-4）。

    旧 bug：sorted({...})[:3] 是 topic_id 字典序取前三——4 个命中主题时字母序
    最后（sim 最高）的那个被切掉，判定器只能看到无关主题。同主题多轮取最大 sim。
    """
    from core.db.database import AsyncSessionLocal, CourseTopic, MemoryEpisode

    async with AsyncSessionLocal() as s:
        for tid in ("zeta", "alpha", "beta", "gamma"):  # 字母序最后 = zeta，sim 最高
            s.add(CourseTopic(course_id="c1", topic_id=tid, label=tid))
        for i in range(4):
            s.add(MemoryEpisode(
                user_id="u1", course_id="c1", session_id="s1", turn_id=f"t{i}",
                user_msg=f"问题{i}", assistant_msg="回答", status="pending",
            ))
        await s.commit()

    # 单轮 4 命中：zeta sim 最高但字母序最后；另一轮 zeta sim 更低（验证取最大 sim）
    hits_seq = [
        [("alpha", 0.60), ("beta", 0.70), ("gamma", 0.80), ("zeta", 0.95)],
        [("zeta", 0.58)],
        [("alpha", 0.60), ("beta", 0.70), ("gamma", 0.80), ("zeta", 0.95)],
        [("alpha", 0.60), ("beta", 0.70), ("gamma", 0.80), ("zeta", 0.99)],
    ]
    it = iter(hits_seq)

    async def _fake_match(text, course_id, db, **kw):
        return next(it)

    seen_topics: list[list] = []

    async def _fake_judge(dialog, topics, track=None):
        seen_topics.append([t for t, _l, _d in topics])
        return _mk_result()

    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(side_effect=_fake_match)),
        patch("core.memory.judge.judge_segment", AsyncMock(side_effect=_fake_judge)),
        patch("core.memory.consolidation._promote_segment", AsyncMock()),
    ):
        await _run()

    assert len(seen_topics) == 3  # 段按主话题（zeta）切成 3 段：[0,2]同题 / [1]低sim / [3]
    # 字母序截断会选出 alpha/beta/gamma 切掉 zeta；按 sim 应是 zeta/gamma/beta
    assert seen_topics[0] == ["zeta", "gamma", "beta"]
