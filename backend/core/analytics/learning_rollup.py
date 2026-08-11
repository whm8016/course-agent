"""学情读模型（L1）聚合：course_daily_rollup + student_course_rollup。

对标 core.analytics.token_usage.rollup_daily：删后重算（幂等），epoch 边界在 Python
算（PG/SQLite 双方言无关），best-effort。展示层（教师学情统计/仪表盘）切读这两张表，
不再每次现算 8 路串行查询（学情分析四模块设计 §第二期 p2-rollup）。

- ``rollup_course_daily(days)``：从 learning_events 按 (课程, 日) 聚合活跃度。
- ``rollup_student_course(course_ids)``：从 Session/Message/NotebookEntry 按 (学生, 课程)
  重算累计；mastery_avg/risk 留给 Phase 4 BKT（NULL 占位，读侧遇 NULL 回退旧公式）。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import Integer, case, delete, func, select

from core.db.database import (
    AsyncSessionLocal,
    CourseDailyRollup,
    Enrollment,
    LearningEvent,
    Message,
    NotebookEntry,
    Session as SessionModel,
    StudentCourseRollup,
)

logger = logging.getLogger(__name__)
# _day_range / recent_days 用 UTC 自然日边界，PG/SQLite 双方言无关。
_DAY_SECONDS = 86400.0
# course_daily_rollup 滚动重算窗口：覆盖展示层 7 天趋势，且首次 cron 即回填历史天。
DAILY_ROLLUP_WINDOW_DAYS = 7


def _day_range(day: str) -> tuple[float, float]:
    """"YYYYMMDD" → 该 UTC 自然日的 ``[start_epoch, end_epoch)``（扫事件明细用）。"""
    start = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    return start, start + _DAY_SECONDS


def recent_days(n: int = DAILY_ROLLUP_WINDOW_DAYS) -> list[str]:
    """最近 n 天（UTC "YYYYMMDD"，今日在前）。

    cron 滚动重算 course_daily_rollup：覆盖展示层 7 天趋势、且首次运行即回填历史天。
    过去天 learning_events 不可变，删后重插幂等无副作用；n 取 7（= 趋势窗口），成本可忽略。
    """
    now = datetime.now(timezone.utc)
    return [(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(n)]


async def rollup_course_daily(days: list[str]) -> int:
    """重算指定天的 course_daily_rollup（每天先删后插，幂等）。返回插入行数；失败返回 0。

    按课程聚合该日 learning_events：去重学生数、asked/answered 计数。
    case + distinct 均标准 SQL，SQLite/PG 通用。
    """
    inserted = 0
    try:
        async with AsyncSessionLocal() as db:
            now = time.time()
            for day in days:
                start, end = _day_range(day)
                await db.execute(delete(CourseDailyRollup).where(CourseDailyRollup.day == day))
                rows = (await db.execute(
                    select(
                        LearningEvent.course_id,
                        func.count(func.distinct(LearningEvent.actor_user_id)).label("active_students"),
                        func.sum(case((LearningEvent.verb == "asked", 1), else_=0)).label("questions"),
                        func.sum(case((LearningEvent.verb == "answered", 1), else_=0)).label("answers"),
                    )
                    .where(
                        LearningEvent.created_at >= start,
                        LearningEvent.created_at < end,
                        LearningEvent.course_id != "",
                    )
                    .group_by(LearningEvent.course_id)
                )).all()
                for row in rows:
                    db.add(CourseDailyRollup(
                        course_id=row.course_id,
                        day=day,
                        active_students=row.active_students or 0,
                        questions=row.questions or 0,
                        answers=row.answers or 0,
                        updated_at=now,
                    ))
                    inserted += 1
            await db.commit()
    except Exception:
        logger.warning("rollup_course_daily failed days=%s", days, exc_info=True)
        return 0
    return inserted


async def recent_active_course_ids(since_ts: float) -> list[str]:
    """``since_ts`` 之后有会话活动的课程 id（圈定 student_course 重算范围，避免全量）。"""
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(
                select(SessionModel.course_id)
                .where(SessionModel.updated_at >= since_ts, SessionModel.course_id != "")
                .group_by(SessionModel.course_id)
            )).all()
            return [r[0] for r in rows]
    except Exception:
        logger.warning("recent_active_course_ids failed", exc_info=True)
        return []


async def rollup_student_course(course_ids: list[str]) -> int:
    """重算指定课程的 student_course_rollup（每课程先删后插，幂等）。返回插入行数。

    从 Session/Message/NotebookEntry 重算累计；quiz 按 P1-a 口径 JOIN Session 取 course
    （避免跨课程泄漏）。mastery_avg/risk 不在此填（Phase 4 BKT，保留 NULL）。
    """
    if not course_ids:
        return 0
    inserted = 0
    try:
        async with AsyncSessionLocal() as db:
            now = time.time()
            for cid in course_ids:
                await db.execute(
                    delete(StudentCourseRollup).where(StudentCourseRollup.course_id == cid)
                )
                enrolled = (await db.execute(
                    select(Enrollment.student_id).where(Enrollment.course_id == cid)
                )).scalars().all()
                if not enrolled:
                    continue

                # 会话数 + 末次活跃
                sess = {
                    r.user_id: (r.cnt, r.last) for r in (await db.execute(
                        select(
                            SessionModel.user_id,
                            func.count(SessionModel.id).label("cnt"),
                            func.max(SessionModel.updated_at).label("last"),
                        )
                        .where(SessionModel.course_id == cid, SessionModel.user_id.in_(enrolled))
                        .group_by(SessionModel.user_id)
                    )).all()
                }
                # user 消息数
                msg = {
                    r.user_id: r.cnt for r in (await db.execute(
                        select(SessionModel.user_id, func.count(Message.id).label("cnt"))
                        .join(Message, Message.session_id == SessionModel.id)
                        .where(
                            SessionModel.course_id == cid,
                            Message.role == "user",
                            SessionModel.user_id.in_(enrolled),
                        )
                        .group_by(SessionModel.user_id)
                    )).all()
                }
                # quiz（JOIN Session 按课程过滤，同 P1-a）。读 OLTP NotebookEntry 是因为
                # verb='answered' 事件尚未接通（p2-b 只 wired asked）；Phase 4 接通后，quiz
                # 计数应改为从 learning_events(verb=answered) 派生——届时两 rollup 同源、
                # 事件自带 course_id 免 JOIN。此处勿当永久口径。
                quiz = {
                    r.user_id: (r.total, r.correct) for r in (await db.execute(
                        select(
                            NotebookEntry.user_id,
                            func.count(NotebookEntry.id).label("total"),
                            func.sum(func.cast(NotebookEntry.is_correct, Integer)).label("correct"),
                        )
                        .where(
                            NotebookEntry.course_id == cid,
                            NotebookEntry.user_id.in_(enrolled),
                        )
                        .group_by(NotebookEntry.user_id)
                    )).all()
                }
                for uid in enrolled:
                    s_sess = sess.get(uid, (0, None))
                    q = quiz.get(uid, (0, 0))
                    db.add(StudentCourseRollup(
                        user_id=uid,
                        course_id=cid,
                        sessions=s_sess[0],
                        messages=msg.get(uid, 0),
                        quiz_total=q[0],
                        quiz_correct=q[1] or 0,
                        last_active_at=s_sess[1],
                        updated_at=now,
                    ))
                    inserted += 1
            await db.commit()
    except Exception:
        logger.warning("rollup_student_course failed courses=%s", course_ids, exc_info=True)
        return 0
    return inserted
