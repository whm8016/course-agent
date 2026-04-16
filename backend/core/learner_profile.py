from __future__ import annotations

import json
import time

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import User

PROFILE_DEFAULT = {
    "level": "unknown",
    "style": "step_by_step",
    "goal": "",
    "preferred_mode": "chat",
    "updated_at": 0.0,
}


def _merge_profile(base: dict, patch: dict) -> dict:
    merged = {**PROFILE_DEFAULT, **(base or {})}
    for key in ("level", "style", "goal", "preferred_mode"):
        value = patch.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    merged["updated_at"] = time.time()
    return merged


def _infer_profile_patch(message: str, mode: str) -> dict:
    text = (message or "").lower()
    patch: dict[str, str] = {"preferred_mode": mode}
    if any(k in text for k in ("基础", "入门", "小白", "从零")):
        patch["level"] = "beginner"
    elif any(k in text for k in ("进阶", "深入", "证明", "推导")):
        patch["level"] = "advanced"

    if any(k in text for k in ("简洁", "要点", "结论")):
        patch["style"] = "concise"
    elif any(k in text for k in ("详细", "一步一步", "展开讲")):
        patch["style"] = "step_by_step"

    if "考试" in text:
        patch["goal"] = "exam"
    elif "作业" in text:
        patch["goal"] = "homework"
    elif "项目" in text:
        patch["goal"] = "project"
    return patch


def build_memory_context(user: dict | None) -> str:
    if not user:
        return ""
    summary = str(user.get("summary_memory") or "").strip()
    profile_raw = user.get("profile_memory") or {}
    profile = profile_raw if isinstance(profile_raw, dict) else {}
    if not summary and not profile:
        return ""

    lines: list[str] = ["【学生画像与学习记忆】"]
    level = str(profile.get("level") or "").strip()
    style = str(profile.get("style") or "").strip()
    goal = str(profile.get("goal") or "").strip()
    preferred_mode = str(profile.get("preferred_mode") or "").strip()

    if level:
        lines.append(f"- 当前水平：{level}")
    if style:
        lines.append(f"- 偏好讲解风格：{style}")
    if goal:
        lines.append(f"- 主要目标：{goal}")
    if preferred_mode:
        lines.append(f"- 偏好模式：{preferred_mode}")
    if summary:
        lines.append("- 近期学习轨迹（简要）：")
        lines.append(summary)
    lines.append("请根据以上信息个性化回答，语言保持清晰、友好。")
    return "\n".join(lines)


async def update_learner_memory(
    db: AsyncSession,
    user_id: str,
    *,
    course_id: str,
    mode: str,
    user_message: str,
    assistant_answer: str,
):
    result = await db.execute(
        select(User.summary_memory, User.profile_memory)
        .where(User.id == user_id)
        .with_for_update()
    )
    row = result.first()
    if not row:
        return

    old_summary = (row.summary_memory or "").strip()
    summary_line = f"- [{course_id}/{mode}] 问：{user_message[:80]}；答：{assistant_answer[:120]}"
    new_summary = "\n".join([old_summary, summary_line]).strip() if old_summary else summary_line
    summary_lines = [line for line in new_summary.splitlines() if line.strip()]
    summary_capped = "\n".join(summary_lines[-25:])

    try:
        old_profile = json.loads(row.profile_memory or "{}")
        if not isinstance(old_profile, dict):
            old_profile = {}
    except json.JSONDecodeError:
        old_profile = {}
    patch = _infer_profile_patch(user_message, mode)
    merged_profile = _merge_profile(old_profile, patch)

    await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(
            summary_memory=summary_capped,
            profile_memory=json.dumps(merged_profile, ensure_ascii=False),
        )
    )
