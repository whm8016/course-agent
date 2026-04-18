from fastapi import APIRouter

from core.cache import cache_get, cache_set
from core.prompts import get_course_list

router = APIRouter()

_COURSES_CACHE_KEY = "courses:list"
_COURSES_CACHE_TTL = 300


@router.get("/courses")
async def list_courses():
    cached = await cache_get(_COURSES_CACHE_KEY)
    if cached is not None:
        return cached
    result = {"courses": get_course_list()}
    await cache_set(_COURSES_CACHE_KEY, result, ttl=_COURSES_CACHE_TTL)
    return result
