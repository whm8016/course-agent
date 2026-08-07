"""get_course_prompt 行为单测。

回归：原先教师未设置 system_prompt 时会兜底"你是一个通用学习助手…"，与
chat.yaml / solve.yaml 的通用 loop.system（已覆盖助手身份）两段身份描述打架。
现在未设置 → 返回空串，被 assemble_system_prompt / assemble_common_context 的
空段过滤丢弃，agent loop 只用通用 prompt，保持干净。

直连 in-memory sqlite（不走 httpx client），照 test_kb_builds 的 init_db 范式。
"""
from __future__ import annotations

import os

from core.db.database import AsyncSessionLocal, KnowledgeBase, close_db, init_db
from core.llm.prompts import get_course_prompt


def _cid() -> str:
    return f"cp_{os.urandom(3).hex()}"


async def test_get_course_prompt_empty_when_unset():
    """教师未设置 system_prompt → 返回空串（不兜底旧默认人设句）。"""
    cid = _cid()
    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(KnowledgeBase(course_id=cid, name="t"))
        assert await get_course_prompt(cid) == ""
    finally:
        await close_db()


async def test_get_course_prompt_returns_teacher_set():
    """教师设置了 system_prompt → 原样返回（含两端空白裁剪）。"""
    cid = _cid()
    await init_db()
    try:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                db.add(
                    KnowledgeBase(
                        course_id=cid, name="t", system_prompt="  你是高数助教，请逐步推导  "
                    )
                )
        assert await get_course_prompt(cid) == "你是高数助教，请逐步推导"
    finally:
        await close_db()
