"""定时学习总结：每日 22:00 + 每周五 22:10 自动汇总活跃用户的学习情况。

由 ARQ cron 触发，遍历当日有对话记录的用户：
  - 日总结：提取当天对话精华 → 更新 summary_memory + 图谱
  - 周总结：聚合本周日总结 → 生成周报写入 summary_memory

参考 MathClaw ScheduledSummaryManager 的设计。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import TEXT_MODEL
from core.db.database import AsyncSessionLocal, User, Session, Message
from core.llm.llm import client as async_openai_client
from core.memory.learner_profile import _save_fields, _load_fields, _read_counter, _write_counter
from core.memory.graph_memory import load_graphs

logger = logging.getLogger(__name__)

_DAILY_SYSTEM_PROMPT = """你是学习总结助手。根据学生今天的对话记录，生成一份简洁的每日学习总结。

格式（Markdown）：
## 今日学习总结
- **学习内容**：今天学了什么
- **掌握情况**：哪些做得好
- **薄弱点**：哪些需要加强
- **明日建议**：下一步该重点复习什么

规则：
- 简洁有力，每项不超过 2-3 句
- 如果今天对话很少或内容无关学习，可以简短总结"""

_WEEKLY_SYSTEM_PROMPT = """你是学习规划助手。根据学生本周的学习情况，生成一份周报和下周建议。

格式（Markdown）：
## 本周学习周报
- **本周概览**：本周主要学了什么
- **进步亮点**：哪些方面有进步
- **持续薄弱**：哪些知识点本周反复出错
- **下周重点**：下周建议关注什么

规则：
- 基于提供的学习摘要和图谱数据做判断
- 简洁实用，帮助学生明确方向"""


async def run_daily_summary(ctx) -> None:
    """ARQ cron job：每日学习总结。"""
    logger.info("Daily summary job started")
    today_start = time.time() - 86400

    async with AsyncSessionLocal() as db:
        active_users = await _get_active_users(db, since=today_start)
        logger.info("Daily summary: %d active users", len(active_users))

        for user_id in active_users:
            try:
                await _generate_daily_for_user(db, user_id, since=today_start)
            except Exception:
                logger.exception("Daily summary failed for user=%s", user_id)
        await db.commit()

    logger.info("Daily summary job completed")


async def run_weekly_summary(ctx) -> None:
    """ARQ cron job：每周学习总结。"""
    logger.info("Weekly summary job started")
    week_start = time.time() - 7 * 86400

    async with AsyncSessionLocal() as db:
        active_users = await _get_active_users(db, since=week_start)
        logger.info("Weekly summary: %d active users", len(active_users))

        for user_id in active_users:
            try:
                await _generate_weekly_for_user(db, user_id)
            except Exception:
                logger.exception("Weekly summary failed for user=%s", user_id)
        await db.commit()

    logger.info("Weekly summary job completed")


async def _get_active_users(db: AsyncSession, since: float) -> list[str]:
    """获取 since 之后有发消息的用户列表。"""
    result = await db.execute(
        select(Session.user_id)
        .where(Session.updated_at >= since)
        .where(Session.user_id != "")
        .group_by(Session.user_id)
    )
    return [row[0] for row in result.all()]


async def _get_recent_messages(db: AsyncSession, user_id: str, since: float, limit: int = 30) -> str:
    """获取用户近期对话的文本摘要。"""
    result = await db.execute(
        select(Message.role, Message.content)
        .join(Session, Message.session_id == Session.id)
        .where(Session.user_id == user_id)
        .where(Message.created_at >= since)
        .where(Message.role.in_(["user", "assistant"]))
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = list(reversed(result.all()))
    if not rows:
        return ""
    lines = []
    for role, content in rows:
        prefix = "学生" if role == "user" else "助教"
        lines.append(f"{prefix}: {(content or '').strip()[:200]}")
    return "\n".join(lines)


async def _generate_daily_for_user(db: AsyncSession, user_id: str, since: float) -> None:
    """为单个用户生成日总结。"""
    transcript = await _get_recent_messages(db, user_id, since)
    if not transcript or len(transcript) < 50:
        return

    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": _DAILY_SYSTEM_PROMPT},
                {"role": "user", "content": f"今日对话记录：\n{transcript[:3000]}"},
            ],
            temperature=0.3,
            max_tokens=600,
            stream=False,
        )
        summary = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Daily summary LLM failed user=%s: %s", user_id, e)
        return

    if not summary:
        return

    loaded = await _load_fields(db, user_id)
    if loaded is None:
        return
    existing_summary, profile_raw = loaded
    date_str = datetime.now().strftime("%Y-%m-%d")
    new_summary = f"{existing_summary}\n\n---\n### {date_str}\n{summary}".strip()
    if len(new_summary) > 8000:
        new_summary = new_summary[-8000:]
    await _save_fields(db, user_id, summary=new_summary)
    logger.info("Daily summary saved for user=%s", user_id)


async def _generate_weekly_for_user(db: AsyncSession, user_id: str) -> None:
    """为单个用户生成周总结。"""
    loaded = await _load_fields(db, user_id)
    if loaded is None:
        return
    current_summary, _ = loaded
    kg, eg = await load_graphs(db, user_id)

    high_risk = sorted(
        [n for n in (kg.get("nodes") or []) if n.get("status") == "active"],
        key=lambda n: -(n.get("risk") or 0),
    )[:5]
    frequent_errors = sorted(
        [n for n in (eg.get("nodes") or [])],
        key=lambda n: -(n.get("error_count") or 0),
    )[:5]

    context = f"本周学习摘要：\n{current_summary[-2000:]}\n\n"
    if high_risk:
        context += "高风险知识点：" + ", ".join(n["label"] for n in high_risk) + "\n"
    if frequent_errors:
        context += "高频错误：" + ", ".join(n["label"] for n in frequent_errors) + "\n"

    try:
        resp = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": _WEEKLY_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            temperature=0.3,
            max_tokens=800,
            stream=False,
        )
        weekly = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Weekly summary LLM failed user=%s: %s", user_id, e)
        return

    if not weekly:
        return

    date_str = datetime.now().strftime("%Y-%m-%d")
    new_summary = f"{current_summary}\n\n===\n### 周报 {date_str}\n{weekly}".strip()
    if len(new_summary) > 10000:
        new_summary = new_summary[-10000:]
    await _save_fields(db, user_id, summary=new_summary)
    logger.info("Weekly summary saved for user=%s", user_id)
