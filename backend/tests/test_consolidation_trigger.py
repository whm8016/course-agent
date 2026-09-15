"""L3 巩固触发路径（lifecycle_subscribers._on_capability_complete）——此前零覆盖。

「攒一攒」整改（删问句信号 + 阈值 0.7→1.2）的行为锚点：
- 累计未过阈值 → 不 enqueue、不清零
- 跨过阈值 → enqueue 一次 + 清零重新攒
- ARQ 池不可用 → 不清零（池恢复后续攒；旧实现清零白丢，只剩 cron 兜底）
"""
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory import episodic
from events.event_bus import CapabilityCompleteEvent


def _pool() -> MagicMock:
    pool = MagicMock()
    pool.enqueue_job = AsyncMock()
    return pool


async def _run_trigger(accumulated: float, pool):
    """跑一轮 _on_capability_complete（真实 record_episode + 真实启发式打分），
    add_importance 固定返回 accumulated、ARQ 池注入 pool。返回 (get_arq_pool, reset) 两 mock。"""
    from core.memory.lifecycle_subscribers import _on_capability_complete

    event = CapabilityCompleteEvent(
        user_id="u_trig", course_id="c_trig", session_id="s_trig", turn_id="t1",
        mode="chat", user_message="什么是欧姆定律",
        agent_output="欧姆定律是……" * 30, metadata={"tools_used": ("rag",)},
    )
    with (
        patch("core.memory.consolidation.add_importance", AsyncMock(return_value=accumulated)),
        patch("core.memory.consolidation.reset_importance", AsyncMock()) as ri,
        patch("core.memory.lifecycle_subscribers.get_arq_pool", AsyncMock(return_value=pool)) as gp,
    ):
        await _on_capability_complete(event)
    return gp, ri


async def test_below_threshold_no_enqueue_no_reset(db):
    """单轮 importance（≈0.26，rag+长度）累计 0.3 < 1.2：不触发、不清零。"""
    pool = _pool()
    gp, ri = await _run_trigger(0.3, pool)

    gp.assert_not_awaited()  # 未过阈值不会去查池
    pool.enqueue_job.assert_not_called()
    ri.assert_not_called()


async def test_crossing_threshold_enqueues_once_and_resets(db):
    """累计 ≥1.2：enqueue consolidate_memory 一次 + 清零（下一轮重新攒）。"""
    pool = _pool()
    _, ri = await _run_trigger(1.25, pool)

    pool.enqueue_job.assert_awaited_once_with(
        "consolidate_memory", user_id="u_trig", course_id="c_trig"
    )
    ri.assert_awaited_once_with("u_trig", "c_trig")


async def test_pool_unavailable_keeps_accumulator(db):
    """回归：ARQ 池不可用时不再清零累计分（旧实现 enqueue 跳过但 reset 照跑，分数白丢）。"""
    _, ri = await _run_trigger(1.25, pool=None)

    ri.assert_not_called()  # 分数留着，池恢复后下一轮续攒


def test_default_threshold_above_single_turn_ceiling():
    """「攒一攒」承重不变量：默认阈值 > 单轮满分（由真实 estimate_importance 算出）。

    阈值是 env 可调字段——若被调到 ≤ 单轮满分，纠错+rag+长回复单轮即 promote，
    本次修的「每轮进 mem0」原样复活；trigger 测试读的是默认值，env 侧漂移只有此锚。"""
    from settings import get_settings

    ceiling = episodic.estimate_importance(
        user_message="我这里算错了，需要纠正" + "长" * 3000,
        agent_output="长" * 3000,
        mode="quiz",
        tools_used=("rag", "solve"),
    )
    assert get_settings().mem0.consolidation_importance_threshold > ceiling
