"""高频问题读模型查询（admin/teacher 共用）。

p2-c：course_faq 语义聚类表为真相源（cron 聚类，见 faq_cluster.py）；冷启动
（cron 未跑/无 asked 事件）回退近 30 天重复提问 SQL（同 P1-a 课程过滤口径）。
P1-c 的 Redis faq_top 精确匹配已退役--"这题怎么算"/"这个怎么算"原文不等但语义同簇。
"""
from __future__ import annotations

import time

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.db.database import CourseFaq, Message, Session


async def frequent_questions_merged(
    db: AsyncSession, course_id: str, top_n: int
) -> list[dict]:
    """单课程 Top-N 高频问题：course_faq 优先，冷启动回退 SQL 重复提问。"""
    faq_rows = (await db.execute(
        select(CourseFaq.question, CourseFaq.count, CourseFaq.last_asked_at)
        .where(CourseFaq.course_id == course_id)
        .order_by(CourseFaq.count.desc())
        .limit(top_n)
    )).all()
    if faq_rows:
        return [
            {"question": r.question, "count": r.count, "last_asked": r.last_asked_at}
            for r in faq_rows
        ]

    # 冷启动兜底：course_faq 未聚簇 -> 近 30 天重复提问（出现 ≥2 次）SQL
    thirty_days_ago = time.time() - 86400 * 30
    # PG 严格模式：SELECT 与 GROUP BY 须为同一表达式对象以复用 bindparam。
    # 用 substr 而非 left：后者 PG 专有，SQLite 测试库没有。
    _content_prefix = func.substr(Message.content, 1, 80)
    sql_rows = (await db.execute(
        select(
            _content_prefix.label("q"),
            func.count().label("cnt"),
            func.max(Message.created_at).label("last_ts"),
        )
        .join(Session, Message.session_id == Session.id)
        .where(
            Session.course_id == course_id,
            Message.role == "user",
            Message.created_at >= thirty_days_ago,
            func.length(Message.content) > 4,
        )
        .group_by(_content_prefix)
        .having(func.count() >= 2)
        .order_by(func.count().desc())
        .limit(top_n)
    )).all()
    return [
        {"question": r.q, "count": r.cnt, "last_asked": r.last_ts}
        for r in sql_rows
    ]
