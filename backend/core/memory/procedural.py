"""L3 procedural 层：把稳定的掌握度/错误模式沉淀成 personal SKILL.md 草稿。

巩固 job 在掌握度数据积累到一定程度时，调 LLM 把学生的薄弱模式/学习偏好总结成一份
SKILL.md 草稿，写入 personal 层（data/skills_user/<user_id>/）。

关键约束（plan Phase 5）：
- **不自动 always**：草稿以 always:false 写入，需人工确认后才可开 always（避免 LLM 自由发挥
  把不可靠总结塞进每轮对话）。course 层本就禁 always（M-48），personal 层允许但草稿不开。
- **审核位**：frontmatter 标 ``auto_generated: true``（SkillService 的 _rewrite_frontmatter
  保留未知键），description 标「待人工确认」，教师/学生经 skill CRUD 复核后决定是否启用。
- **幂等不覆盖**：同名草稿已存在则跳过（等人工处理），不反复生成。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DRAFT_NAME_PREFIX = "profile-"
# 累计观测数超此才生成草稿——数据太少时总结不可靠，避免噪声沉淀成"技能"
_DRAFT_TRIGGER_OBSERVATIONS = 20


def _draft_name(course_id: str) -> str:
    """course_id → 合法 skill slug（^[a-z0-9][a-z0-9-]{0,63}$）。"""
    slug = re.sub(r"[^a-z0-9]+", "-", (course_id or "global").lower()).strip("-")[:40] or "global"
    return f"{_DRAFT_NAME_PREFIX}{slug}"


def build_draft_prompt(mastery_rows: list, course_id: str, risk_by_label: dict[str, float] | None = None) -> str:
    """纯函数：把掌握度薄弱点拼成 LLM 草稿生成 prompt。

    risk_by_label 非空时按派生口径（risk = 1 - sigmoid(theta - b_topic)，与门控
    读侧同源）覆盖行上 risk——difficulty_b 变更后其他学生行上的旧 risk 陈旧，
    派生口径永远吃到当前 b。
    """
    def _risk(r) -> float:
        if risk_by_label is not None:
            return risk_by_label.get(r.label, r.risk or 0)
        return r.risk or 0

    weak = sorted(
        [r for r in mastery_rows if _risk(r) >= 0.5],
        key=lambda r: -_risk(r),
    )[:10]
    lines = [f"- {r.label}：风险{_risk(r):.2f}（{r.observation_count}次观测）" for r in weak]
    weak_text = "\n".join(lines) or "（暂无明显薄弱点）"
    return (
        f"你是学习分析助手。根据下面该学生在课程「{course_id}」的掌握度数据，总结其稳定的学习模式，"
        "生成一份 SKILL.md 草稿正文，供教师/学生本人确认后启用。\n\n"
        f"掌握度薄弱点（按风险排序）：\n{weak_text}\n\n"
        "要求：\n"
        "- 输出中文 Markdown 正文（不要 frontmatter，系统会自动加），≤300 字\n"
        "- 总结反复出现的薄弱主题、可能的学习偏好（如「偏好先看结论」「符号方向易错」）\n"
        "- 给出 2-3 条可操作的教学建议\n"
        "- 基于数据，不编造"
    )


async def generate_skill_draft(
    course_id: str, user_id: str, mastery_rows: list,
    risk_by_label: dict[str, float] | None = None,
) -> str | None:
    """调 LLM 生成草稿正文（Markdown，无 frontmatter）。失败返回 None。"""
    from core.llm.llm import client as async_openai_client
    from settings import get_settings

    prompt = build_draft_prompt(mastery_rows, course_id, risk_by_label)
    try:
        model = get_settings().llm.text_model
        resp = await async_openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
            stream=False,
        )
        # 用量埋点（当前该链路休眠，保留接入保证启用即有账）
        from core.analytics.token_usage import track_llm_response

        await track_llm_response(
            resp, model=model, mode="procedural_draft",
            user_id=user_id, course_id=course_id,
        )
        return ((resp.choices[0].message.content or "").strip()) or None
    except Exception as exc:
        logger.warning(
            "[procedural] draft generation failed user=%s course=%s: %s", user_id, course_id, exc
        )
        return None


def write_skill_draft(course_id: str, user_id: str, body: str) -> str | None:
    """把草稿正文写成 personal SKILL.md（不自动 always；标 auto_generated 待审核）。

    已存在则跳过（不覆盖，等人工确认）。返回 slug 或 None。
    """
    from core.skills.skill_service import SkillExistsError, get_skill_service

    name = _draft_name(course_id)
    description = "【自动生成·待人工确认】该生学习模式与教学建议（复核后可开 always）"
    # 带 auto_generated frontmatter：_normalize_content → _rewrite_frontmatter 保留未知键，
    # 故 auto_generated 会留存于最终文件，作审核位。always 留空（不自动注入每轮）。
    content = (
        f"---\nname: {name}\ndescription: {description}\nauto_generated: true\n---\n\n{body}\n"
    )
    svc = get_skill_service(course_id, user_id)
    try:
        svc.create(name=name, description=description, content=content, always=False)
        logger.info(
            "[procedural] draft written user=%s course=%s name=%s", user_id, course_id, name
        )
        return name
    except SkillExistsError:
        logger.debug("[procedural] draft exists, skip user=%s course=%s", user_id, course_id)
        return None


async def maybe_generate_procedural(db, user_id: str, course_id: str) -> str | None:
    """门槛：掌握度累计观测数足够才生成；已存在草稿则由 write 阶段跳过。返回 slug 或 None。"""
    from sqlalchemy import func, select

    from core.db.database import KnowledgeMastery

    total = (
        await db.execute(
            select(func.coalesce(func.sum(KnowledgeMastery.observation_count), 0)).where(
                KnowledgeMastery.user_id == user_id,
                KnowledgeMastery.course_id == course_id,
            )
        )
    ).scalar() or 0
    if total < _DRAFT_TRIGGER_OBSERVATIONS:
        return None

    rows = (
        (
            await db.execute(
                select(KnowledgeMastery).where(
                    KnowledgeMastery.user_id == user_id,
                    KnowledgeMastery.course_id == course_id,
                )
            )
        )
        .scalars()
        .all()
    )
    # 薄弱判定按派生口径（theta - b_topic 实时折 risk），与门控读侧同源——行上的
    # risk 列在 difficulty_b 变更后对其他学生是陈旧值（P1 陈旧行处理选派生不选重折）
    from core.db.database import CourseTopic

    b_map = {
        r.topic_id: float(r.difficulty_b or 0.0) for r in (await db.execute(
            select(CourseTopic.topic_id, CourseTopic.difficulty_b)
            .where(CourseTopic.course_id == course_id)
        )).all()
    }
    import math as _math

    # risk = 1 - sigmoid(theta - b_topic)（掌握度取反）。旧版写成 1 - sigmoid(-theta + b)
    # = sigmoid(theta - b) = 掌握度本身——风险算成了掌握度，薄弱点排序全反（掌握越高
    # 越"薄弱"）。与门控读侧（proactive._read_mastery）、mastery_state 同口径。
    def _risk_derived(theta: float, b: float) -> float:
        # 1 - sigmoid(theta - b) = 1/(1+e^{theta-b})：theta 越高（掌握越好）risk 越低
        return 1.0 / (1.0 + _math.exp(theta - b))

    risk_by_label = {
        r.label: _risk_derived(r.theta or 0.0, b_map.get(r.kp_id, 0.0))
        for r in rows
        if r.label
    }
    body = await generate_skill_draft(course_id, user_id, rows, risk_by_label)
    if not body:
        return None
    return write_skill_draft(course_id, user_id, body)


__all__ = [
    "build_draft_prompt",
    "generate_skill_draft",
    "write_skill_draft",
    "maybe_generate_procedural",
]
