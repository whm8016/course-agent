"""判分观测接入：答题结果（NotebookEntry.is_correct）→ mastery_observation。

plan 阶段F（§4b）：题面经 match_topics 映射 topic_id（零 LLM），写 source=quiz 的
观测（outcome 0/1、weight=W_QUIZ）。判分观测是全班难度估计权重最高的来源
（w=1.0），也是上线后校准的金标数据。难度由 cron_refit_difficulty 日频全班重折
（写路径不在线更新 b，见 plan「班级难度估计修复」）。

幂等：同 (user, question_id) 重复提交不重复写观测（upsert 场景常见——重答/改注）。
失败非致命：观测丢了可由 notebook 重放，不阻塞答题主流程。
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def record_quiz_observation(
    db, user_id: str, course_id: str, question_id: str,
    question_text: str, level: str | bool,
    user_answer: str = "", correct_answer: str = "",
    weight: float | None = None, feedback: str = "",
) -> bool:
    """题面 match_topics → 首命中主题写一条 quiz 观测。返回是否写入。

    无 course_id / 无主题命中 / 重复提交 → 不写（返回 False）。映射失败是常态
    （题面与主题图不对齐），warning 不 error。

    level（A3 等级化）：good/fair/poor（fair=0.5 部分正确通道）；传 bool 是
    choice 本地判的兼容写法（True→good / False→poor），旧调用方零改动。

    weight（可选）：观测权重，默认 W_QUIZ(1.0)；LLM 判分来源传 W_JUDGE(0.6)——
    软证据。feedback（可选）：判分反馈——非 good 时 evidence_quote 优先用它
    （LLM/coding 判分的错因比「学生答 vs 正确答案」拼接更聚焦，门控简报直接可读）。

    user_answer/correct_answer（可选尾参）：feedback 为空且非 good 时存
    「学生答 vs 正确答案」的具体差异（错因闭环的落点，门控简报读它拼「曾见
    错法」）。good 或无学生作答时回退题面截断 question_text[:200]。
    """
    from core.db.database import MasteryObservation
    from core.memory.topic_match import match_topics
    from core.memory.mastery_state import W_QUIZ
    from sqlalchemy import select

    if isinstance(level, bool):  # choice 本地判兼容：bool → 等级
        level = "good" if level else "poor"
    level = str(level).strip().lower()
    if level not in ("good", "fair", "poor"):
        logger.warning("[quiz_observe] 非法 level=%r，跳过观测", level)
        return False
    if weight is None:
        weight = W_QUIZ

    if not course_id or not (question_text or "").strip():
        return False
    # 幂等：同一 (user, course, question) 已有 quiz 观测则跳过（重答改注不重复计）。
    # 补 course_id——跨课程同题号会被误判重复（观测是 course 隔离的）
    dup = (
        await db.execute(
            select(MasteryObservation.id)
            .where(
                MasteryObservation.user_id == user_id,
                MasteryObservation.course_id == course_id,
                MasteryObservation.source == "quiz",
                MasteryObservation.question_id == question_id,
            )
            .limit(1)
        )
    ).first()
    if dup:
        return False

    try:
        hits = await match_topics(question_text, course_id, db)
    except Exception as exc:
        logger.warning("[quiz_observe] match_topics failed q=%s: %s", question_id, exc)
        return False
    if not hits:
        return False
    topic_id = hits[0][0]
    # evidence_quote 优先级（A3）：判分 feedback > 学生答 vs 正确答案 > 题面截断。
    # feedback 是 LLM/coding 判分的一句话错因，非 good 时最聚焦；为空时回退既有
    # 「学生答」拼接（只有 correct_answer 没有 user_answer 时写出来是空半截，
    # 不如回退题面）。
    if level != "good" and feedback.strip():
        evidence = feedback[:200]
    elif level != "good" and user_answer.strip():
        evidence = f"学生答：{user_answer[:100]}"
        if correct_answer.strip():
            evidence += f"；正确答案：{correct_answer[:100]}"
    else:
        evidence = question_text[:200]
    # 观测追加 + 折叠写回（含 b_topic 题目侧更新）复用 consolidation 同一段逻辑
    from core.memory.consolidation import _write_mastery_observations
    from core.memory.judge import Judgement

    await _write_mastery_observations(
        db, user_id, course_id,
        [Judgement(topic_id, level, 1.0, evidence)],
        [],  # 无 episode 关联
        source_override="quiz", weight_override=weight,
        question_id=question_id,
    )
    return True
