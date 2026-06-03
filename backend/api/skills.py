"""Custom Output Skills REST API。

- GET    /api/skills          -> 列出所有 Skills
- POST   /api/skills          -> 创建新 Skill
- PATCH  /api/skills/{id}     -> 切换启用/禁用
- DELETE /api/skills/{id}     -> 删除 Skill
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import get_current_user
from core.skills.output_skills import get_skill_store

router = APIRouter(prefix="/skills")


class CreateSkillRequest(BaseModel):
    title: str
    description: str = ""
    instruction: str
    course_id: str = ""


class ToggleSkillRequest(BaseModel):
    enabled: bool


@router.get("")
async def list_skills(user: dict = Depends(get_current_user)):
    store = get_skill_store()
    skills = store.list_all()
    return {"skills": [s.to_dict() for s in skills]}


@router.post("")
async def create_skill(
    payload: CreateSkillRequest,
    user: dict = Depends(get_current_user),
):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="仅教师/管理员可创建 Skill")
    if not payload.title.strip() or not payload.instruction.strip():
        raise HTTPException(status_code=400, detail="title 和 instruction 不能为空")

    store = get_skill_store()
    try:
        skill = store.create(
            title=payload.title,
            description=payload.description,
            instruction=payload.instruction,
            course_id=payload.course_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"skill": skill.to_dict(), "created": True}


@router.patch("/{skill_id}")
async def toggle_skill(
    skill_id: str,
    payload: ToggleSkillRequest,
    user: dict = Depends(get_current_user),
):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="仅教师/管理员可修改 Skill")

    store = get_skill_store()
    skill = store.toggle(skill_id, payload.enabled)
    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"skill": skill.to_dict()}


@router.delete("/{skill_id}")
async def delete_skill(
    skill_id: str,
    user: dict = Depends(get_current_user),
):
    if user.get("role") not in ("teacher", "admin"):
        raise HTTPException(status_code=403, detail="仅教师/管理员可删除 Skill")

    store = get_skill_store()
    if not store.delete(skill_id):
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"deleted": True}
