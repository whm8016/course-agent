"""学情门控冷却：回合成功后落盘 + 不依赖 mastery 行（任务10）。

根因（旧实现两个洞）：
1. stitch_for_turn 在拼上下文时就 _mark_stitched（按 row_id UPDATE）——模型调用
   失败也烧掉 24h 冷却；
2. C2 复发证据来自 mastery_observation 观测日志，(user, course, topic) 可能还没有
   knowledge_mastery 行 → 无 row_id → 冷却不写、每轮重复拼。

修法：stitch_for_turn 只拼简报不写戳；pipeline_common 把 evidence_topic 塞进
ctx.metadata["_pending_stitch_cooldown"]；turn_runtime 在回合成功（agent_output
非空，CapabilityCompleteEvent 发布处）后调 commit_stitch_cooldown——按
(user, course, kp_id) upsert，无行建占位行。
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from core.context import UnifiedContext
from core.stream import StreamEvent, StreamEventType
from services.session.turn_runtime import TurnRuntimeManager


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


# ── commit_stitch_cooldown upsert ────────────────────────────────────────────


async def test_cooldown_upserts_when_no_mastery_row(db):
    """无 mastery 行（C2 观测日志有、折叠行没有）→ 建占位行写冷却，不再漏写。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from core.memory.proactive import commit_stitch_cooldown
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        await commit_stitch_cooldown(s, "u_cd", "c1", "kcl")

    async with AsyncSessionLocal() as s:
        row = (
            (
                await s.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == "u_cd",
                        KnowledgeMastery.course_id == "c1",
                        KnowledgeMastery.kp_id == "kcl",
                    )
                )
            )
            .scalars()
            .one()
        )
    assert row.last_stitched_at is not None


async def test_cooldown_updates_existing_row(db):
    """已有行 → 更新 last_stitched_at（不动 theta 等折叠状态）。"""
    from core.db.database import AsyncSessionLocal, KnowledgeMastery
    from core.memory.proactive import commit_stitch_cooldown
    from sqlalchemy import select

    async with AsyncSessionLocal() as s:
        s.add(KnowledgeMastery(
            user_id="u_cd2", course_id="c1", kp_id="kcl", label="KCL",
            theta=1.5, n_obs=4, last_stitched_at=100.0,
        ))
        await s.commit()

    async with AsyncSessionLocal() as s:
        await commit_stitch_cooldown(s, "u_cd2", "c1", "kcl")

    async with AsyncSessionLocal() as s:
        row = (
            (
                await s.execute(
                    select(KnowledgeMastery).where(
                        KnowledgeMastery.user_id == "u_cd2",
                    )
                )
            )
            .scalars()
            .one()
        )
    assert row.last_stitched_at > 100.0  # 已更新
    assert row.theta == 1.5 and row.n_obs == 4  # 折叠状态不动


# ── turn_runtime 成功后才落盘 ────────────────────────────────────────────────


class _Orch:
    def __init__(self, events):
        self._events = events

    async def handle(self, ctx):
        for ev in self._events:
            yield ev


async def _run_turn_with(events, ctx_metadata=None):
    """跑一个 turn 捕获 commit_stitch_cooldown 是否被调。"""
    called: list[tuple] = []

    async def _fake_commit(db, user_id, course_id, topic_id):
        called.append((user_id, course_id, topic_id))

    ctx = UnifiedContext(
        course_id="c1", user_id="u1", user_message="问", mode="chat",
        session_id="s", conversation_history=[],
        metadata=ctx_metadata or {},
    )
    with patch(
        "core.orchestrator.get_orchestrator", return_value=_Orch(events)
    ), patch(
        "core.memory.proactive.commit_stitch_cooldown", _fake_commit
    ), patch(
        "services.session.turn_runtime.get_settings",
        return_value=type(
            "S", (),
            {
                "research": type("R", (), {"clarify_enabled": False, "clarify_wait_timeout_s": 0, "clarify_max_questions": 3})(),
                "context_budget": type("CB", (), {"coordinator_enabled": False})(),
            },
        )(),
    ):
        mgr = TurnRuntimeManager()
        turn_id = await mgr.start_turn(ctx)
        async for _ev in mgr.subscribe_turn(turn_id):
            pass
    return called


async def test_successful_turn_commits_cooldown():
    """回合成功（有 ANSWER）→ 待写冷却落盘。"""
    events = [StreamEvent(type=StreamEventType.ANSWER, payload={"content": "KVL 是……"})]
    called = await _run_turn_with(
        events, ctx_metadata={"_pending_stitch_cooldown": {"kcl": True}}
    )
    assert called == [("u1", "c1", "kcl")]


async def test_failed_turn_does_not_burn_cooldown():
    """回合失败（无 ANSWER/result）→ 冷却不落盘（模型失败不烧 24h）。"""
    events = [StreamEvent(type=StreamEventType.ERROR, payload={"message": "LLM down"})]
    called = await _run_turn_with(
        events, ctx_metadata={"_pending_stitch_cooldown": {"kcl": True}}
    )
    assert called == []


async def test_no_pending_metadata_no_commit():
    """无待写冷却 → 不触碰 DB。"""
    events = [StreamEvent(type=StreamEventType.ANSWER, payload={"content": "ok"})]
    called = await _run_turn_with(events, ctx_metadata={})
    assert called == []
