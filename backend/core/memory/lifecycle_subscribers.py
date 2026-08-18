"""CAPABILITY_COMPLETE 的记忆/学情订阅者（自 main.py 归位的业务层）。

此前住在入口文件里（main.py L73-153）：记忆巩固触发策略（importance 累计阈值 →
enqueue consolidate）与 L2 压缩时机属于 core/memory 的领域逻辑，调记忆阈值的人
不该翻 FastAPI 入口。main 的 lifespan 只调 ``subscribe_memory_handlers(bus)`` 一行。
"""
from __future__ import annotations

import logging

from core.arq_pool import get_arq_pool

logger = logging.getLogger(__name__)


# ---- EventBus：turn 完成后的记忆更新订阅者 ----
async def _on_capability_complete(event) -> None:
    """CAPABILITY_COMPLETE：写 episodic 原始层（L3 outbox，取代 Redis buffer）+ L2 摘要压缩。"""
    try:
        from core.memory.episodic import record_episode

        await record_episode(
            user_id=event.user_id,
            course_id=event.course_id,
            session_id=event.session_id,
            turn_id=event.turn_id,
            user_msg=event.user_message,
            assistant_msg=event.agent_output,
            mode=event.mode,
            tools_used=event.metadata.get("tools_used") or (),
        )
    except Exception:
        logger.warning("EventBus: record_episode failed", exc_info=True)

    # L3 巩固触发（Phase 3）：importance 累计超阈值 → enqueue consolidate_memory。
    # 对齐 Generative Agents 的 reflection 累计触发；5min cron safety net 兜底（worker.py）。
    try:
        from core.memory import consolidation, episodic
        from settings import get_settings

        score = episodic.estimate_importance(
            user_message=event.user_message,
            agent_output=event.agent_output,
            mode=event.mode,
            tools_used=event.metadata.get("tools_used") or (),
        )
        if score > 0:
            accumulated = await consolidation.add_importance(event.user_id, event.course_id, score)
            threshold = get_settings().mem0.consolidation_importance_threshold
            if accumulated >= threshold:
                pool = await get_arq_pool()
                if pool is not None:
                    await pool.enqueue_job(
                        "consolidate_memory", user_id=event.user_id, course_id=event.course_id
                    )
                await consolidation.reset_importance(event.user_id, event.course_id)
    except Exception:
        logger.warning("EventBus: consolidation trigger failed", exc_info=True)

    # L2: 触发 session summary 增量压缩（异步，不阻塞主链路）
    session_id = getattr(event, "session_id", "") or event.metadata.get("session_id", "") or ""
    if session_id:
        import asyncio
        asyncio.create_task(_maybe_compress_summary(session_id))


async def _maybe_compress_summary(session_id: str) -> None:
    """异步执行 L2 摘要压缩（不阻塞主链路）。"""
    try:
        from core.memory.session_summary import get_summary_manager
        from core.db.database import AsyncSessionLocal

        summary_mgr = get_summary_manager()
        async with AsyncSessionLocal() as db:
            compressed = await summary_mgr.maybe_compress(db, session_id)
            if compressed:
                logger.info("[L2] session summary compressed session=%s", session_id)
    except Exception:
        logger.warning("[L2] session summary compress failed session=%s", session_id, exc_info=True)


async def _on_complete_record_learning_event(event) -> None:
    """CAPABILITY_COMPLETE 订阅者：落一条 verb=asked 学情事件（L0）。

    供 p2-c FAQ 语义聚类 / p2-b rollup 聚合。作为独立订阅者并发派发（见上）。
    record_learning_event 内部吞异常，best-effort。
    """
    from core.analytics.events import record_learning_event
    session_id = getattr(event, "session_id", "") or event.metadata.get("session_id") or ""
    await record_learning_event(
        user_id=event.user_id,
        course_id=event.course_id,
        verb="asked",
        object_text=(event.user_message or "")[:1000],
        session_id=session_id,
        metadata={"mode": event.mode, "tools_used": list(event.metadata.get("tools_used") or ())},
    )


def subscribe_memory_handlers(bus) -> None:
    """注册 CAPABILITY_COMPLETE 记忆/学情订阅者（main lifespan 调用）。"""
    from events.event_bus import EventType

    bus.subscribe(EventType.CAPABILITY_COMPLETE, _on_capability_complete)
    bus.subscribe(EventType.CAPABILITY_COMPLETE, _on_complete_record_learning_event)
