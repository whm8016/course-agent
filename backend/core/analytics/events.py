"""学情事件层（L0）写入：把对话/答题/反馈信号落 learning_events 表。

best-effort：事件记录失败只记日志，绝不阻塞主链路（对话/答题的成功不依赖事件落盘）。
读模型层（rollup / course_faq）由 ARQ cron 从本表增量聚合（见 学情分析四模块设计 §第二期）。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import AsyncSessionLocal, LearningEvent

logger = logging.getLogger(__name__)


async def record_learning_event(
    *,
    user_id: str,
    course_id: str,
    verb: str,
    object_id: str = "",
    object_text: str = "",
    session_id: str = "",
    metadata: dict[str, Any] | None = None,
    db: AsyncSession | None = None,
) -> None:
    """追加一条学情事件（verb ∈ asked/answered/feedback）。best-effort，失败仅记日志。

    事件只追加不修改（append-only 事实）；course_id 为空（非课程对话）也记，
    rollup/FAQ cron 按 course_id 非空过滤即可。

    ``db`` 传入时仅 add（由调用方 session 负责提交）——供 REST 端点复用请求 session，
    避免 SQLite StaticPool 单连接下另开 AsyncSessionLocal 死锁（M-42 同源坑）；
    不传则自开 session 提交（asked 事件经 EventBus 订阅者派发时，请求 session 已关闭）。
    """

    def _add(s: AsyncSession) -> None:
        s.add(LearningEvent(
            actor_user_id=user_id,
            course_id=course_id or "",
            verb=verb,
            object_id=object_id,
            object_text=object_text,
            session_id=session_id,
            metadata_=metadata,
        ))

    try:
        if db is not None:
            _add(db)
        else:
            async with AsyncSessionLocal() as s:
                _add(s)
                await s.commit()
    except Exception:
        logger.warning(
            "record_learning_event failed verb=%s user=%s course=%s",
            verb, user_id, course_id, exc_info=True,
        )
