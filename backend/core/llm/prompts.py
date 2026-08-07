from __future__ import annotations

from core.db.cache import cache_delete, cache_get, cache_set

_PROMPT_CACHE_KEY = "course:prompt:{}"
_PROMPT_CACHE_TTL = 600  # 10 分钟


async def get_course_prompt(course_id: str) -> str:
    """从 Redis 缓存或数据库获取课程 system_prompt。

    教师未设置时返回空串——不兜底默认人设。agent loop 的通用行为规范（chat.yaml /
    solve.yaml 的 loop.system）已覆盖助手身份，再叠一句课程级默认人设会造成两段身份
    描述打架。空串会被 assemble_system_prompt / assemble_common_context 的空段过滤丢弃，
    故未设置 system_prompt 的课程，loop 只用通用 prompt，保持干净。
    """
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

    await cache_set(key, prompt, ttl=_PROMPT_CACHE_TTL)
    return prompt


async def invalidate_course_prompt_cache(course_id: str) -> None:
    """管理员更新 system_prompt 后调用，使缓存立即失效。"""
    await cache_delete(_PROMPT_CACHE_KEY.format(course_id))
