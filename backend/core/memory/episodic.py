"""L3 episodic 记忆层：原始对话 turn 持久化 + 巩固 outbox。

取代旧的 Redis buffer（flush_manager）：每轮 turn 完成后 INSERT 一条 episode，
永不删除（保留 provenance，提取逻辑改进后可重算历史）。status 字段表达巩固进度，
(session_id, turn_id) 唯一索引保证幂等。巩固 job（Phase 3）从 pending 批量消费。

四层记忆中的「episodic」层：
- episodic（本模块）：原始 turn，永不删除，兼 outbox
- semantic（mem0）：由后台从 segment 升格的事实条目
- mastery（knowledge_mastery）：知识点掌握度，追加观测 + 读时衰减
- procedural（SKILL.md）：稳定的教学策略模式
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

# importance 启发式信号（零 LLM）
_QUESTION_SIGNALS = ("?", "？", "什么", "为什么", "怎么", "如何", "哪里", "是否", "对吗", "对不对")
_ERROR_SIGNALS = ("不对", "错了", "不理解", "不懂", "搞错", "纠正", "混淆", "记错", "弄错")
_WEIGHTY_TOOLS = {"rag", "solve", "web_search", "retrieve"}


def estimate_importance(
    *, user_message: str, agent_output: str, mode: str = "", tools_used=()
) -> float:
    """零 LLM 启发式估计一轮对话的教学重要性（0.0-1.0）。

    信号叠加（封顶 1.0）：
      - 消息长度（实质内容越多越重要，最多 0.3）
      - 疑问词（学生在提问 = 学习信号，+0.25）
      - 纠错/困惑词（错题、误解 = 掌握度关键证据，+0.25）
      - 触发重量级工具 rag/solve/web_search（实质检索求解，+0.2）
      - 深度模式 quiz/deep_research/deep_solve 比闲聊更重要（+0.1）

    供巩固 job 排序与触发阈值使用，绝不进热路径 LLM 调用。
    """
    score = 0.0
    u = user_message or ""
    a = agent_output or ""

    score += min((len(u) + len(a)) / 2000.0, 1.0) * 0.3
    if any(s in u for s in _QUESTION_SIGNALS):
        score += 0.25
    if any(s in u for s in _ERROR_SIGNALS):
        score += 0.25
    if tools_used and any(t in _WEIGHTY_TOOLS for t in tools_used):
        score += 0.2
    if mode in {"quiz", "deep_research", "deep_solve"}:
        score += 0.1

    return round(min(score, 1.0), 3)


async def record_episode(
    *,
    user_id: str,
    course_id: str,
    session_id: str,
    turn_id: str,
    user_msg: str,
    assistant_msg: str,
    mode: str = "",
    tools_used=(),
) -> bool:
    """记录一轮原始对话到 episodic 表（L3 outbox）。

    幂等：(session_id, turn_id) 唯一约束 → 同一 turn 重放只写一次（IntegrityError 视为已存在）。
    保留所有实质内容（含纯噪声「好的」——靠 importance 降权，巩固时跳过）；仅完全空消息跳过。

    Returns:
        True=新写入；False=跳过（无 user_id / 完全空 / 重复 turn）。
    """
    if not user_id:
        return False
    if not (user_msg or "").strip() and not (assistant_msg or "").strip():
        return False
    if not turn_id:
        turn_id = uuid.uuid4().hex[:16]

    importance = estimate_importance(
        user_message=user_msg, agent_output=assistant_msg, mode=mode, tools_used=tools_used
    )

    from core.db.database import AsyncSessionLocal, MemoryEpisode

    try:
        async with AsyncSessionLocal() as db:
            db.add(
                MemoryEpisode(
                    user_id=user_id,
                    course_id=course_id or "",
                    session_id=session_id or "",
                    turn_id=turn_id,
                    mode=mode or "",
                    user_msg=user_msg or "",
                    assistant_msg=assistant_msg or "",
                    importance=importance,
                    status="pending",
                )
            )
            await db.commit()
        return True
    except IntegrityError:
        # (session_id, turn_id) 已存在——重放/重试幂等忽略
        logger.debug(
            "[episodic] duplicate episode skipped session=%s turn=%s", session_id, turn_id
        )
        return False


__all__ = ["estimate_importance", "record_episode"]
