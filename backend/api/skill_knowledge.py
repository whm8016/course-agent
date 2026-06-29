"""Skill 知识包（SKILL.md playbook）管理 REST API。

DeepTutor 式 skill 知识包（教师创建教学 playbook，模型经 read_skill 按需读取）。
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

from api.auth import get_current_user
from core.skills.skill_service import (
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
async def create_skill(payload: CreateSkillRequest, user: dict = Depends(get_current_user)):
    if not payload.name.strip() or not payload.description.strip():
        raise HTTPException(status_code=400, detail="name 和 description 不能为空")
    svc = get_skill_service(payload.course_id, _skill_user_id(user))
    try:
        info = svc.create(payload.name, payload.description, payload.content, always=payload.always)
    except SkillExistsError:
        raise HTTPException(status_code=409, detail="同名 skill 已存在")
    except InvalidSkillNameError:
        raise HTTPException(
            status_code=400, detail="非法 skill 名（需匹配 ^[a-z0-9][a-z0-9-]{0,63}$）"
        )
    return {"skill": info.to_dict(), "created": True}


@router.put("/{name}")
async def update_skill(
    name: str, payload: UpdateSkillRequest, user: dict = Depends(get_current_user)
):
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
    return {"skill": info.to_dict()}


@router.delete("/{name}")
async def delete_skill(name: str, course_id: str = "", user: dict = Depends(get_current_user)):
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
