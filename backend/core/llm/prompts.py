from __future__ import annotations

from core.db.cache import cache_delete, cache_get, cache_set

SUMMARY_PROMPT = """你是一个学习总结专家。根据对话历史，为学生生成一份结构清晰的学习小结。

要求：
1. 提炼对话中涉及的核心知识点
2. 用简洁的条目列出要点
3. 标注学生可能还需要加强的地方
4. 给出下一步学习建议
5. 使用 Markdown 格式，结构清晰"""

_FALLBACK_PROMPT = "你是一个通用学习助手。请尽力回答学生与课程学习相关的问题。如果问题与课程学习完全无关，请礼貌拒绝。"

_PROMPT_CACHE_KEY = "course:prompt:{}"
_PROMPT_CACHE_TTL = 600  # 10 分钟


async def get_course_prompt(course_id: str) -> str:
    """从 Redis 缓存或数据库获取课程 system_prompt。"""
    key = _PROMPT_CACHE_KEY.format(course_id)
    cached = await cache_get(key)
    if cached is not None:
        return cached

    from sqlalchemy import select
    from core.db.database import AsyncSessionLocal, KnowledgeBase

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(KnowledgeBase.system_prompt).where(KnowledgeBase.course_id == course_id)
        )
        row = result.first()

    prompt = (row[0] or "").strip() if row else ""
    if not prompt:
        prompt = _FALLBACK_PROMPT

    await cache_set(key, prompt, ttl=_PROMPT_CACHE_TTL)
    return prompt


async def invalidate_course_prompt_cache(course_id: str) -> None:
    """管理员更新 system_prompt 后调用，使缓存立即失效。"""
    await cache_delete(_PROMPT_CACHE_KEY.format(course_id))
