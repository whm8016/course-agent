"""深度研究阶段级 checkpoint 读写（plan 阶段 2B）。

best-effort：所有操作 catch 异常只记日志——checkpoint 是「worker 重启可恢复」的增强，绝不能让
它的读写失败把正在跑的研究搞挂（研究主流程不依赖 checkpoint 落库）。语义对齐 LangGraph
checkpointer：resume 时被中断阶段整段重放（接受该阶段内部 LLM/检索成本重付一次）。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from core.db.database import AsyncSessionLocal, ResearchCheckpoint

logger = logging.getLogger(__name__)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


async def save_phase(
    research_id: str,
    *,
    user_id: str | None = None,
    course_id: str | None = None,
    topic: str | None = None,
    phase: str | None = None,
    state: dict[str, Any] | None = None,
    status: str | None = None,
    pending_question: dict[str, Any] | None = None,
) -> None:
    """upsert 一条 checkpoint（仅覆盖显式传入的字段）。失败 best-effort 记日志不抛。"""
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                ckpt = await db.get(ResearchCheckpoint, research_id)
                if ckpt is None:
                    ckpt = ResearchCheckpoint(research_id=research_id)
                    db.add(ckpt)
                if user_id is not None:
                    ckpt.user_id = user_id
                if course_id is not None:
                    ckpt.course_id = course_id
                if topic is not None:
                    ckpt.topic = topic
                if phase is not None:
                    ckpt.phase = phase
                if state is not None:
                    ckpt.state_json = _dumps(state)
                if status is not None:
                    ckpt.status = status
                if pending_question is not None:
                    ckpt.pending_question_json = _dumps(pending_question)
                ckpt.updated_at = time.time()
    except Exception:
        logger.warning("research checkpoint save_phase 失败 research_id=%s", research_id, exc_info=True)


async def load(research_id: str) -> ResearchCheckpoint | None:
    """读一条 checkpoint；不存在 / 异常返回 None。"""
    try:
        async with AsyncSessionLocal() as db:
            return await db.get(ResearchCheckpoint, research_id)
    except Exception:
        logger.warning("research checkpoint load 失败 research_id=%s", research_id, exc_info=True)
        return None


async def clear(research_id: str) -> None:
    """研究终态清理（best-effort）。保留也行（供事后排查），这里 done/error 后删。"""
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                ckpt = await db.get(ResearchCheckpoint, research_id)
                if ckpt is not None:
                    await db.delete(ckpt)
    except Exception:
        logger.warning("research checkpoint clear 失败 research_id=%s", research_id, exc_info=True)
