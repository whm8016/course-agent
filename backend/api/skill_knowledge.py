"""Skill 知识包（SKILL.md playbook）管理 REST API。

式 skill 知识包（教师创建教学 playbook，模型经 read_skill 按需读取）。
与 /api/skills（output_cards 对话后补充框）是两个不同的概念，路由分开。

- GET    /api/skill-knowledge           -> 列出所有 skill（builtin + user）
- GET    /api/skill-knowledge/{name}    -> skill 详情（正文 + meta）
- POST   /api/skill-knowledge           -> 创建 user skill
- PUT    /api/skill-knowledge/{name}    -> 更新（description/content/always/rename_to）
- DELETE /api/skill-knowledge/{name}    -> 删除 user skill
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.db.database import KnowledgeBase, get_db
from core.skills.skill_service import (
    InvalidSkillAlwaysError,
    InvalidSkillNameError,
    SkillExistsError,
    SkillNotFoundError,
    SkillReadOnlyError,
    get_skill_service,
)

router = APIRouter(prefix="/skill-knowledge")


class CreateSkillRequest(BaseModel):
    name: str
    description: str
    content: str = ""
    always: bool = False
    course_id: str = ""


class UpdateSkillRequest(BaseModel):
    description: str | None = None
    content: str | None = None
    always: bool | None = None
    rename_to: str | None = None
    course_id: str = ""


def _owner_of(user: dict) -> str:
    return str(user.get("id") or "")


def _skill_user_id(user: dict) -> str:
    """决定 skill 操作的 user_id：学生 → personal 层（私有），教师/admin → course 层（共享）。"""
    return _owner_of(user) if user.get("role") == "student" else ""


async def _assert_course_skill_write_access(
    db: AsyncSession, course_id: str, user: dict
) -> None:
    """M-47：写 course 层 skill（course_id 非空）前校验当前用户对该课程有管理权。

    只有该课程的 owner（教师）或 admin 才能把 skill 写进课程共享目录；其它人（学生、
    别的课程教师）传他人 course_id 一律 403。这堵住"学生 A 把恶意 skill 塞进老师课程
    目录、污染该课程所有学生对话"的越权写入。

    course_id 为空（写个人/全局层）时直接放行——那是用户自己的私有 skill。
    """
    cid = (course_id or "").strip()
    if not cid or cid == "_global":
        return
    role = user.get("role", "student")
    if role == "admin":
        return
    if role != "teacher":
        # 学生无权写任何课程层（学生只能写 personal）
        raise HTTPException(status_code=403, detail="无权向课程写入共享 skill")
    result = await db.execute(
        select(KnowledgeBase.id).where(
            KnowledgeBase.course_id == cid,
            KnowledgeBase.owner_id == user["id"],
        )
    )
    if result.first() is None:
        raise HTTPException(status_code=403, detail="无权管理此课程的 skill")


@router.get("")
async def list_skills(course_id: str = "", user: dict = Depends(get_current_user)):
    svc = get_skill_service(course_id, _skill_user_id(user))
    return {"skills": [asdict(e) for e in svc.summary_entries()]}


@router.get("/{name}")
async def get_skill(name: str, course_id: str = "", user: dict = Depends(get_current_user)):
    svc = get_skill_service(course_id, _skill_user_id(user))
    try:
        return svc.get_detail(name)
    except SkillNotFoundError:
        raise HTTPException(status_code=404, detail="Skill not found")
    except InvalidSkillNameError:
        raise HTTPException(status_code=400, detail="非法 skill 名")


@router.post("")
async def create_skill(
    payload: CreateSkillRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not payload.name.strip() or not payload.description.strip():
        raise HTTPException(status_code=400, detail="name 和 description 不能为空")
    # M-47：写 course 层前校验课程管理权，防越权写他人课程目录。
    await _assert_course_skill_write_access(db, payload.course_id, user)
    svc = get_skill_service(payload.course_id, _skill_user_id(user))
    try:
        info = svc.create(payload.name, payload.description, payload.content, always=payload.always)
    except SkillExistsError:
        raise HTTPException(status_code=409, detail="同名 skill 已存在")
    except InvalidSkillNameError:
        raise HTTPException(
            status_code=400, detail="非法 skill 名（需匹配 ^[a-z0-9][a-z0-9-]{0,63}$）"
        )
    except InvalidSkillAlwaysError as exc:
        # M-48：course 层禁止 always:true（会污染所有学生每轮 prompt）
        raise HTTPException(status_code=400, detail=str(exc))
    return {"skill": info.to_dict(), "created": True}


@router.put("/{name}")
async def update_skill(
    name: str,
    payload: UpdateSkillRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # M-47：写 course 层前校验课程管理权。
    await _assert_course_skill_write_access(db, payload.course_id, user)
    svc = get_skill_service(payload.course_id, _skill_user_id(user))
    try:
        info = svc.update(
            name,
            description=payload.description,
            content=payload.content,
            always=payload.always,
            rename_to=payload.rename_to,
        )
    except SkillNotFoundError:
        raise HTTPException(status_code=404, detail="Skill not found")
    except SkillReadOnlyError:
        raise HTTPException(status_code=403, detail="builtin skill 只读，不可修改")
    except SkillExistsError:
        raise HTTPException(status_code=409, detail="目标名称已存在")
    except InvalidSkillNameError:
        raise HTTPException(status_code=400, detail="非法 skill 名")
    except InvalidSkillAlwaysError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"skill": info.to_dict()}


@router.delete("/{name}")
async def delete_skill(
    name: str,
    course_id: str = "",
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # M-47：删 course 层前同样校验课程管理权（删除也是写操作）。
    await _assert_course_skill_write_access(db, course_id, user)
    svc = get_skill_service(course_id, _skill_user_id(user))
    try:
        svc.delete(name)
    except SkillNotFoundError:
        raise HTTPException(status_code=404, detail="Skill not found")
    except SkillReadOnlyError:
        raise HTTPException(status_code=403, detail="builtin skill 只读，不可删除")
    except InvalidSkillNameError:
        raise HTTPException(status_code=400, detail="非法 skill 名")
    return {"deleted": True}
