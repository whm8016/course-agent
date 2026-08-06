
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.codes import normalize_code
from core.db.cache import cache_delete, cache_delete_pattern, cache_get, cache_set, course_access_get, course_access_set
from core.db.database import Enrollment, KnowledgeBase, aggregate_build_status, get_db

router = APIRouter()

_COURSES_CACHE_TTL = 100


def _kb_to_course(kb: KnowledgeBase, include_join_code: bool = False) -> dict:
    builds = kb.builds
    # 已就绪的后端（status==ready 且真摄入 chunks）：学生端按此决定 RAG 可用 + 选择器可见性。
    # chunks_total>0 防空库（两后端索引完成都会写 chunks_total）。
    ready_backends = sorted(
        {b.backend for b in builds if b.status == "ready" and (b.chunks_total or 0) > 0}
    )
    d: dict = {
        "id": kb.course_id,
        "name": kb.name or kb.course_id,
        "icon": kb.icon or "📘",
        "description": kb.description or "",
        # kb_status 由 kb_builds 聚合（indexing/error/paused/ready/pending）；任一后端就绪即 rag_enabled。
        "kb_status": aggregate_build_status(builds),
        "rag_enabled": bool(ready_backends),
        # 学生端问答选择器：仅当 lightrag 已就绪才显示 mix/naive/local；auto 永远可用
        "index_backends": ready_backends,
        "source": "db",
    }
    if include_join_code:
        d["join_code"] = kb.join_code
    return d


def _user_courses_cache_key(user_id: str) -> str:
    return f"courses:list:{user_id}"


@router.get("/courses")
async def list_courses(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cache_key = _user_courses_cache_key(user["id"])
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    role = user.get("role", "student")
    show_join_code = role in ("teacher", "admin")

    if role == "admin":
        result = await db.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.is_visible == True)
            .order_by(KnowledgeBase.sort_order.asc(), KnowledgeBase.created_at.asc())
        )
    elif role == "teacher":
        result = await db.execute(
            select(KnowledgeBase)
            .where(
                KnowledgeBase.owner_id == user["id"],
                KnowledgeBase.is_visible == True,
            )
            .order_by(KnowledgeBase.sort_order.asc(), KnowledgeBase.created_at.asc())
        )
    else:
        result = await db.execute(
            select(KnowledgeBase)
            .join(Enrollment, Enrollment.course_id == KnowledgeBase.course_id)
            .where(
                Enrollment.student_id == user["id"],
                KnowledgeBase.is_visible == True,
            )
            .order_by(KnowledgeBase.sort_order.asc(), KnowledgeBase.created_at.asc())
        )

    courses = [_kb_to_course(kb, include_join_code=show_join_code) for kb in result.scalars().all()]
    payload = {"courses": courses}
    await cache_set(cache_key, payload, ttl=_COURSES_CACHE_TTL)
    return payload


async def check_course_access(db: AsyncSession, course_id: str, user: dict) -> None:
    """Raise 403 if the user has no access to this course.

    管理员直接放行；教师/学生先查 Redis 缓存（TTL 5 min），未命中才走 DB。
    """
    # 自由问答 / 未选课：虚拟课程，任何登录用户都可访问（不挂知识库、不查 Enrollment）
    if not course_id or course_id == "general":
        return

    role = user.get("role", "student")
    user_id: str = user["id"]

    if role == "admin":
        return

    # 从缓存快速判断
    cached = await course_access_get(user_id, course_id)
    if cached is True:
        return
    if cached is False:
        raise HTTPException(status_code=403, detail="未选此课程，无法访问")

    # 缓存未命中，走 DB
    if role == "teacher":
        result = await db.execute(
            select(KnowledgeBase.id).where(
                KnowledgeBase.course_id == course_id,
                KnowledgeBase.owner_id == user_id,
            )
        )
        allowed = result.first() is not None
        await course_access_set(user_id, course_id, allowed)
        if not allowed:
            raise HTTPException(status_code=403, detail="无权访问此课程")
        return

    result = await db.execute(
        select(Enrollment.id).where(
            Enrollment.student_id == user_id,
            Enrollment.course_id == course_id,
        )
    )
    allowed = result.first() is not None
    await course_access_set(user_id, course_id, allowed)
    if not allowed:
        raise HTTPException(status_code=403, detail="未选此课程，无法访问")


async def invalidate_courses_cache() -> None:
    """供 admin / teacher 模块在创建/删除/索引完成时调用。

    课程列表按用户缓存（key 为 courses:list:{user_id}），课程的可见性/状态变更会影响
    所有学生，因此用 SCAN 匹配前缀批量失效全部用户的缓存，而非删单个不存在的固定 key。
    """
    await cache_delete_pattern("courses:list:*")


class JoinCourseBody(BaseModel):
    join_code: str


@router.post("/courses/join", status_code=200)
async def join_course_by_code(
    body: JoinCourseBody,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """学生凭课程码自助入课。"""
    code = normalize_code(body.join_code)
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.join_code == code)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="课程码无效")

    existing = await db.execute(
        select(Enrollment.id).where(
            Enrollment.student_id == user["id"],
            Enrollment.course_id == kb.course_id,
        )
    )
    if existing.first():
        return {"course_id": kb.course_id, "name": kb.name, "already_enrolled": True}

    enrollment = Enrollment(student_id=user["id"], course_id=kb.course_id)
    db.add(enrollment)
    await db.commit()
    # 让学生课程列表缓存失效，并写入权限缓存（True，避免下次查 DB）
    await cache_delete(_user_courses_cache_key(user["id"]))
    await course_access_set(user["id"], kb.course_id, True)
    return {"course_id": kb.course_id, "name": kb.name, "already_enrolled": False}
