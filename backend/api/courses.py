from fastapi import APIRouter

from core.prompts import get_course_list

router = APIRouter()


@router.get("/courses")
async def list_courses():
    return {"courses": get_course_list()}
