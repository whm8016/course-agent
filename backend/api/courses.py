from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.cache import cache_delete, cache_get, cache_set
from core.database import KnowledgeBase, get_db

router = APIRouter()

_COURSES_CACHE_KEY = "courses:list:v2"
_COURSES_CACHE_TTL = 100


def _kb_to_course(kb: KnowledgeBase) -> dict:
    return {
        "id": kb.course_id,
        "name": kb.name or kb.course_id,
        "icon": kb.icon or "📘",
        "description": kb.description or "",
        "kb_status": kb.status,
        "rag_enabled": kb.status == "ready",
        "source": "db",
    }


@router.get("/courses")
async def list_courses(db: AsyncSession = Depends(get_db)):
    cached = await cache_get(_COURSES_CACHE_KEY)
    if cached is not None:
        return cached

    result = await db.execute(
    select(KnowledgeBase)
    .where(KnowledgeBase.is_visible == True)
    .order_by(KnowledgeBase.sort_order.asc(), KnowledgeBase.created_at.asc())
)
    courses = [_kb_to_course(kb) for kb in result.scalars().all()]
    payload = {"courses": courses}
    await cache_set(_COURSES_CACHE_KEY, payload, ttl=_COURSES_CACHE_TTL)
    return payload


async def invalidate_courses_cache() -> None:
    """供 admin 模块在创建/删除/索引完成时调用，确保前端立即看到变化。"""
    await cache_delete(_COURSES_CACHE_KEY)
    await cache_delete("courses:list")
