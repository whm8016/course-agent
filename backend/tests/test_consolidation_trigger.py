"""L3 巩固触发路径（lifecycle_subscribers._on_capability_complete）——此前零覆盖。

「攒一攒」整改（删问句信号 + 阈值 0.7→1.2）的行为锚点：
- 累计未过阈值 → 不 enqueue、不清零
- 跨过阈值 → enqueue 一次 + 清零重新攒
- ARQ 池不可用 → 不清零（池恢复后续攒；旧实现清零白丢，只剩 cron 兜底）
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
async def db():
    from core.db.database import close_db, init_db

    await init_db()
    yield
    await close_db()


def _event(turn_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        user_id="u_trig", course_id="c_trig", session_id="s_trig", turn_id=turn_id,
        mode="chat", user_message="什么是欧姆定律", agent_output="欧姆定律是……" * 30,
        metadata={"tools_used": ("rag",)},
    )


async def test_below_threshold_no_enqueue_no_reset(db):
    """单轮 importance（≈0.26，rag+长度）累计 0.3 < 1.2：不触发、不清零。"""
    from core.memory.lifecycle_subscribers import _on_capability_complete

    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    with (
        patch("core.memory.consolidation.add_importance", AsyncMock(return_value=0.3)),
        patch("core.memory.consolidation.reset_importance", AsyncMock()) as ri,
        patch("core.memory.lifecycle_subscribers.get_arq_pool", AsyncMock(return_value=pool)) as gp,
    ):
        await _on_capability_complete(_event("t_below"))

    gp.assert_not_awaited()  # 未过阈值不会去查池
    pool.enqueue_job.assert_not_called()
    ri.assert_not_called()


async def test_crossing_threshold_enqueues_once_and_resets(db):
    """累计 ≥1.2：enqueue consolidate_memory 一次 + 清零（下一轮重新攒）。"""
    from core.memory.lifecycle_subscribers import _on_capability_complete

    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    with (
        patch("core.memory.consolidation.add_importance", AsyncMock(return_value=1.25)),
        patch("core.memory.consolidation.reset_importance", AsyncMock()) as ri,
        patch("core.memory.lifecycle_subscribers.get_arq_pool", AsyncMock(return_value=pool)),
    ):
        await _on_capability_complete(_event("t_cross"))

    pool.enqueue_job.assert_awaited_once_with(
        "consolidate_memory", user_id="u_trig", course_id="c_trig"
    )
    ri.assert_awaited_once_with("u_trig", "c_trig")


async def test_pool_unavailable_keeps_accumulator(db):
    """回归：ARQ 池不可用时不再清零累计分（旧实现 enqueue 跳过但 reset 照跑，分数白丢）。"""
    from core.memory.lifecycle_subscribers import _on_capability_complete

    with (
        patch("core.memory.consolidation.add_importance", AsyncMock(return_value=1.25)),
        patch("core.memory.consolidation.reset_importance", AsyncMock()) as ri,
        patch("core.memory.lifecycle_subscribers.get_arq_pool", AsyncMock(return_value=None)),
    ):
        await _on_capability_complete(_event("t_nopool"))

    ri.assert_not_called()  # 分数留着，池恢复后下一轮续攒
