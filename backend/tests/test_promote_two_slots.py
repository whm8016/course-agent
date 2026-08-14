"""阶段2验收：主题连续段切 event + persona 另写 + 两槽检索 + mem0 失败不拖垮 mastery。

口径（plan 阶段2验收）：
- ohm×3 + thevenin×2 → 2 段，中间插入无主题的不并入两侧
- 一段 4 轮 ohm → m.add event 恰一次且 infer=False
- 「我喜欢分步讲」→ 另一次 add 且 kind=="persona"
- 「好的」→ 不 add event
- mem0 失败 → warning 降级，episode 照常 done、mastery 照常写入
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from core.memory import consolidation
from core.memory.consolidation import _event_line, split_topic_segments


def _ep(i: int, user_msg: str, assistant_msg: str = "答"):
    return SimpleNamespace(
        id=f"ep{i}", user_msg=user_msg, assistant_msg=assistant_msg,
        session_id="s1", turn_id=f"t{i}",
    )


# ── split_topic_segments（纯函数）────────────────────────────────────────────


def test_split_two_topics_two_segments():
    eps = [_ep(1, "q1"), _ep(2, "q2"), _ep(3, "q3"), _ep(4, "q4"), _ep(5, "q5")]
    segs = split_topic_segments(eps, ["ohm", "ohm", "ohm", "thevenin", "thevenin"])
    assert [len(s) for s in segs] == [3, 2]


def test_split_none_segment_stands_alone():
    """无主题（None）插在两主题之间 → 自成一段，不并入两侧。"""
    eps = [_ep(1, "q"), _ep(2, "q"), _ep(3, "q")]
    segs = split_topic_segments(eps, ["ohm", None, "thevenin"])
    assert [len(s) for s in segs] == [1, 1, 1]


def test_split_consecutive_none_grouped():
    segs = split_topic_segments([_ep(1, "q"), _ep(2, "q")], [None, None])
    assert [len(s) for s in segs] == [1] or [len(s) for s in segs] == [2]


def test_split_length_mismatch_raises():
    with pytest.raises(ValueError):
        split_topic_segments([_ep(1, "q")], [])


def test_event_line_format():
    line = _event_line([1, 2, 3, 4], "ohm", "欧姆定律", ["confusion", "confusion", "纠正"])
    assert line == "欧姆定律 · 4轮 · confusion/纠正"


# ── _promote_segment 双槽写入（mock mem0）───────────────────────────────────


class _MemRecorder:
    """伪 mem0：记录每次 add 的 (内容, metadata, infer)。"""

    def __init__(self):
        self.calls: list[dict] = []

    async def add(self, content, *, user_id=None, metadata=None, infer=None, **kw):
        self.calls.append(
            {"content": content, "metadata": metadata or {}, "infer": infer}
        )


async def _run_promote(eps, topic_ids_per_ep, mem):
    with patch("core.memory.mem0_client.get_memory", return_value=mem):
        await consolidation._promote_segment(
            "u1", "c1", eps, db=None, topic_ids_per_ep=topic_ids_per_ep
        )


async def test_promote_one_event_per_segment_infer_false():
    """4 轮 ohm 一段 → event 恰一次，infer=False，文本是压缩行。"""
    mem = _MemRecorder()
    eps = [_ep(i, f"欧姆定律问题{i}") for i in range(1, 5)]
    await _run_promote(eps, ["ohm"] * 4, mem)
    events = [c for c in mem.calls if c["metadata"].get("kind") == "event"]
    assert len(events) == 1
    assert events[0]["infer"] is False
    assert "4轮" in events[0]["content"]
    assert events[0]["metadata"]["topic_ids"] == ["ohm"]
    assert events[0]["metadata"]["segment_id"]


async def test_promote_persona_written_separately():
    """「我喜欢分步讲」→ 独立一次 add，kind=persona，不混进 event。"""
    mem = _MemRecorder()
    eps = [_ep(1, "我喜欢分步讲解欧姆定律")]
    await _run_promote(eps, ["ohm"], mem)
    personas = [c for c in mem.calls if c["metadata"].get("kind") == "persona"]
    assert len(personas) == 1
    assert "我喜欢分步讲解" in personas[0]["content"]
    assert personas[0]["infer"] is False
    # event 也有一条（有主题）
    assert any(c["metadata"].get("kind") == "event" for c in mem.calls)


async def test_promote_no_event_for_topicless():
    """「好的」+ 无主题命中 → 不 add event。"""
    mem = _MemRecorder()
    await _run_promote([_ep(1, "好的", "不客气")], [None], mem)
    assert [c for c in mem.calls if c["metadata"].get("kind") == "event"] == []


async def test_promote_two_segments_two_events():
    """ohm×3 + thevenin×2 → 两条 event。"""
    mem = _MemRecorder()
    eps = [_ep(1, "a"), _ep(2, "a"), _ep(3, "a"), _ep(4, "b"), _ep(5, "b")]
    await _run_promote(eps, ["ohm", "ohm", "ohm", "thevenin", "thevenin"], mem)
    events = [c for c in mem.calls if c["metadata"].get("kind") == "event"]
    assert len(events) == 2
    assert {e["metadata"]["topic_ids"][0] for e in events} == {"ohm", "thevenin"}


# ── mem0 失败不拖垮 mastery（consolidate 层）────────────────────────────────


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


async def test_mem0_failure_does_not_block_mastery(db):
    """mem0 双槽写炸 → warning 降级；episode 照 done、mastery 照写（不回 pending）。

    阶段C：判定路径 mock 为返回 fair 判定（真实走观测折叠写 mastery）；mem0 炸
    只降级 warning，判定与 done 不受影响。"""
    from core.db.database import AsyncSessionLocal, CourseTopic, KnowledgeMastery, MemoryEpisode
    from core.memory.judge import Judgement, JudgeResult
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        s.add(CourseTopic(course_id="c1", topic_id="ohm", label="欧姆定律"))
        s.add(MemoryEpisode(
            user_id="u1", course_id="c1", session_id="s1", turn_id="t1",
            user_msg="欧姆定律公式是什么", assistant_msg="U=IR", status="pending",
        ))
        await s.commit()

    judge_result = JudgeResult(
        judgements=[Judgement("ohm", "fair", 0.8, "欧姆定律公式是什么")],
        abstained=[], error_patterns=[], dropped_invalid_topic=0, dropped_bad_quote=0,
    )
    with (
        patch("core.memory.topic_match.match_topics", AsyncMock(return_value=[("ohm", 0.9)])),
        patch("core.memory.judge.judge_segment", AsyncMock(return_value=judge_result)),
        patch(
            "core.memory.consolidation._promote_segment",
            AsyncMock(side_effect=RuntimeError("mem0 down")),
        ),
    ):
        from core.db.database import AsyncSessionLocal as ASL

        async with ASL() as s:
            result = await consolidation.consolidate(s, "u1", "c1")

    assert result == {"claimed": 1, "promoted": 1}
    async with AsyncSessionLocal() as s:
        ep = (await s.execute(
            select(MemoryEpisode).where(MemoryEpisode.user_id == "u1")
        )).scalars().first()
        km = (await s.execute(
            select(KnowledgeMastery).where(KnowledgeMastery.user_id == "u1")
        )).scalars().all()
    assert ep.status == "done"  # 不再回 pending
    assert len(km) == 1 and km[0].kp_id == "ohm"  # mastery 未被拖垮
    assert km[0].n_obs == 1  # 判定观测已折叠（mem0 失败不影响判定路径）
