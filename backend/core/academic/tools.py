"""只读学业查询工具：query_timetable / query_grades / query_mistakes。

设计要点（对照学业工具与 RAG 防线 plan §2.4 权限设计）：
  - 身份只用 registry 注入的 ``user_id``（``core/agent/registry.py:100``），schema **绝不**
    暴露 user_id / student_id 参数——若模型幻觉出该参数，``tool_dispatch.py:56`` 的
    ``**call_kwargs`` 会与注入的 user_id 撞成 TypeError，被 registry 兜底成「工具执行失败」
    （安全但报错不友好，故 schema 干脆不出现身份参数）。
  - 三个工具全部只读 SELECT；写入只走教师 REST API + JWT 角色校验（读写分离，OWASP LLM06）。
  - 用参数化查询（SQLAlchemy ORM）而非 text-to-SQL，把越权在结构上变成不可能：
    WHERE 条件里的身份来自注入值，不来自模型输入。
  - 返回值只回决策相关字段 + LIMIT，避免 context 膨胀。

挂载范围：web 对话（/api/chat）前端 ``tools`` 数组透传（api/chat.py）；与 cron_tool 同款，
本模块自带 schema + executor，由 ``register_builtins`` 装配（core/agent/registry.py）。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import aliased

from core.agent.tool_protocol import ToolResult

logger = logging.getLogger(__name__)

# 单条记录渲染时的文本截断上限，避免错题长题面把 context 撑爆。
_ITEM_TEXT_LIMIT = 200


def _truncate(text: str, limit: int = _ITEM_TEXT_LIMIT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _weekday_cn(weekday: int | None) -> str:
    if isinstance(weekday, bool) or weekday is None:  # bool 是 int 子类，先挡掉
        return "—"
    names = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}
    return names.get(int(weekday), "—")


# ── query_timetable：个人课表（跨已选课程）───────────────────────────────────

async def execute_query_timetable(
    *, course_id: str = "", user_id: str = "", **kwargs: Any
) -> ToolResult:
    """查询学生本人的课表（JOIN Enrollment 限定「我选的课」，跨课程）。

    课表本质是跨课程的个人视图（「下周三几点上课」要覆盖所有已选课），故不按当前
    course_id 收窄，而是按选课关系查询。可选 weekday 过滤（1=周一…7=周日）。
    """
    if not user_id:
        return ToolResult(content="当前会话无法确认学生身份，无法查询课表。", success=False)

    # weekday 可选：1-7 整数。非法值（0/8/负数）当作不过滤，避免空结果误导。
    weekday = kwargs.get("weekday")
    if weekday is not None:
        try:
            weekday = int(weekday)
        except (TypeError, ValueError):
            weekday = None
        if weekday not in range(1, 8):
            weekday = None

    from core.db.database import AsyncSessionLocal, CourseSchedule, Enrollment, KnowledgeBase

    try:
        async with AsyncSessionLocal() as db:
            # JOIN Enrollment 限定「我选的课」，带出课程名（KnowledgeBase.name）便于阅读。
            kb = aliased(KnowledgeBase)
            stmt = (
                select(CourseSchedule, kb.name)
                .join(Enrollment, Enrollment.course_id == CourseSchedule.course_id)
                .outerjoin(kb, kb.course_id == CourseSchedule.course_id)
                .where(Enrollment.student_id == user_id)
            )
            if weekday is not None:
                stmt = stmt.where(CourseSchedule.weekday == weekday)
            stmt = stmt.order_by(CourseSchedule.weekday.asc(), CourseSchedule.start_time.asc())
            rows = (await db.execute(stmt)).all()
    except Exception as exc:
        logger.exception("query_timetable failed user=%s", user_id)
        return ToolResult(content=f"（课表查询失败：{exc}）", success=False)

    if not rows:
        when = f"（星期{_weekday_cn(weekday)}）" if weekday is not None else ""
        return ToolResult(content=f"没有查询到你{when}的课表记录。", success=False)

    lines = [f"共查询到 {len(rows)} 条课表记录："]
    for sched, course_name in rows:
        name = course_name or sched.course_id
        weeks = f" 第{sched.weeks}周" if sched.weeks else ""
        teacher = f" {sched.teacher_name}" if sched.teacher_name else ""
        lines.append(
            f"- {_weekday_cn(sched.weekday)} {sched.start_time}–{sched.end_time} "
            f"{name} @ {sched.location or '未定'}{teacher}{weeks}"
        )
    return ToolResult(content="\n".join(lines))


# ── query_grades：本人成绩（当前课程优先，可选条目关键词）──────────────────

async def execute_query_grades(
    *, course_id: str = "", user_id: str = "", **kwargs: Any
) -> ToolResult:
    """查询学生本人的成绩记录（强制 WHERE student_id==注入的 user_id）。

    成绩天然按课程问询（「期中考了多少分」在课程上下文里问），故按注入的 course_id
    收窄；课程上下文缺失（自由问答 general）时返回全部课程成绩。可选 item_keyword。
    """
    if not user_id:
        return ToolResult(content="当前会话无法确认学生身份，无法查询成绩。", success=False)

    item_keyword = str(kwargs.get("item_keyword") or "").strip()
    try:
        limit = int(kwargs.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 50))

    from core.db.database import AsyncSessionLocal, Grade

    try:
        async with AsyncSessionLocal() as db:
            stmt = select(Grade).where(Grade.student_id == user_id)
            # 真实课程上下文才收窄；general / 空 → 跨课程全查（学生问「我所有成绩」也能答）。
            if course_id and course_id != "general":
                stmt = stmt.where(Grade.course_id == course_id)
            if item_keyword:
                stmt = stmt.where(Grade.item_name.ilike(f"%{item_keyword}%"))
            stmt = stmt.order_by(Grade.graded_at.desc().nullslast(), Grade.created_at.desc()).limit(limit)
            grades = (await db.execute(stmt)).scalars().all()
    except Exception as exc:
        logger.exception("query_grades failed user=%s", user_id)
        return ToolResult(content=f"（成绩查询失败：{exc}）", success=False)

    if not grades:
        scope = f"（关键词「{item_keyword}」）" if item_keyword else ""
        return ToolResult(content=f"没有查询到你的成绩记录{scope}。", success=False)

    lines = [f"共查询到 {len(grades)} 条成绩记录："]
    for g in grades:
        comment = f" 评语：{_truncate(g.comment, 60)}" if g.comment else ""
        lines.append(f"- {g.item_name} {g.score:g}/{g.full_score:g}{comment}")
    return ToolResult(content="\n".join(lines))


# ── query_mistakes：本人错题（NotebookEntry 中 is_correct=False）────────────

async def execute_query_mistakes(
    *, course_id: str = "", user_id: str = "", **kwargs: Any
) -> ToolResult:
    """查询学生本人做错的题目（读 NotebookEntry，只取 is_correct=False）。

    与 query_grades 的语义区分（plan §2.2 按数据形态描述）：成绩是「得分记录」，
    错题本是「做错的题目内容（你的错答 / 正确答案 / 解析）」。可选题目关键词。
    """
    if not user_id:
        return ToolResult(content="当前会话无法确认学生身份，无法查询错题。", success=False)

    keyword = str(kwargs.get("keyword") or "").strip()
    try:
        limit = int(kwargs.get("limit") or 10)
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 30))

    from core.db.database import AsyncSessionLocal, NotebookEntry

    try:
        async with AsyncSessionLocal() as db:
            stmt = select(NotebookEntry).where(
                NotebookEntry.user_id == user_id,
                NotebookEntry.is_correct == False,  # noqa: E712 — ORM 比较需用 ==
            )
            if keyword:
                stmt = stmt.where(NotebookEntry.question.ilike(f"%{keyword}%"))
            stmt = stmt.order_by(NotebookEntry.updated_at.desc()).limit(limit)
            entries = (await db.execute(stmt)).scalars().all()
    except Exception as exc:
        logger.exception("query_mistakes failed user=%s", user_id)
        return ToolResult(content=f"（错题查询失败：{exc}）", success=False)

    if not entries:
        scope = f"（关键词「{keyword}」）" if keyword else ""
        return ToolResult(content=f"没有查询到你的错题记录{scope}。", success=False)

    lines = [f"共查询到 {len(entries)} 条错题："]
    for e in entries:
        qtype = f"【{e.question_type}】" if e.question_type else ""
        ua = f" 你的答案：{_truncate(e.user_answer, 40)}" if e.user_answer else ""
        ca = f" 正确答案：{_truncate(e.correct_answer, 40)}" if e.correct_answer else ""
        lines.append(f"- {qtype}{_truncate(e.question)}{ua}{ca}")
    return ToolResult(content="\n".join(lines))


# ── OpenAI function schemas（不含任何身份参数）──────────────────────────────

QUERY_TIMETABLE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_timetable",
        "description": (
            "查询学生本人的课表——所选修全部课程的上课时间、地点、任课教师与周次。"
            "用于「我几点上课 / 下周三有什么课 / 这周在哪上课」这类查询。只能查自己的课表。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "weekday": {
                    "type": "integer",
                    "description": "按星期几过滤：1=周一、2=周二 … 7=周日。不传则返回整张课表。",
                },
            },
            "required": [],
        },
    },
}


QUERY_GRADES_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_grades",
        "description": (
            "查询学生本人的成绩记录——各次作业 / 考试的得分、满分与评语。"
            "用于「我期中考了多少分 / 这门课成绩怎样 / 哪些没及格」这类查询。只能查自己的成绩。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_keyword": {
                    "type": "string",
                    "description": "按条目名称关键词过滤，如「期中」「作业1」。不传则返回全部。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回的条目数，默认 20。",
                },
            },
            "required": [],
        },
    },
}


QUERY_MISTAKES_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_mistakes",
        "description": (
            "查询学生本人做错的题目（错题本）——之前答错的题面、你的错误答案、正确答案与解析。"
            "用于「我哪些题做错了 / 最近错题 / 错在哪」这类查询。只能查自己的错题。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "按题目内容关键词过滤。不传则返回最近的错题。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回的条目数，默认 10。",
                },
            },
            "required": [],
        },
    },
}


__all__ = [
    "execute_query_timetable",
    "execute_query_grades",
    "execute_query_mistakes",
    "QUERY_TIMETABLE_SCHEMA",
    "QUERY_GRADES_SCHEMA",
    "QUERY_MISTAKES_SCHEMA",
]
