"""Teacher API: course CRUD, KB upload/index, student enrollment."""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_teacher
from api.admin import _kb_to_dict, _run_indexing, _ALLOWED_EXT, _MAX_BYTES
from api.courses import invalidate_courses_cache
from config import KB_STORE_DIR, MAX_KB_UPLOAD_MB
from core.db.database import AsyncSessionLocal, Enrollment, KBFile, KnowledgeBase, User, get_db
from core.rag.ingestion import IndexingControl

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/teacher", tags=["teacher"])


def _kb_raw_dir(course_id: str) -> Path:
    return Path(KB_STORE_DIR) / course_id / "raw"


async def _get_owned_kb(
    db: AsyncSession, course_id: str, user: dict
) -> KnowledgeBase:
    """Fetch KB and verify the teacher owns it (admin bypasses)."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail=f"课程 '{course_id}' 不存在")
    if user.get("role") != "admin" and kb.owner_id != user["id"]:
        raise HTTPException(status_code=403, detail="无权管理此课程")
    return kb


# ── 课程 CRUD ─────────────────────────────────────────────────────────────────

@router.get("/courses")
async def my_courses(
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    if teacher.get("role") == "admin":
        result = await db.execute(
            select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
        )
    else:
        result = await db.execute(
            select(KnowledgeBase)
            .where(KnowledgeBase.owner_id == teacher["id"])
            .order_by(KnowledgeBase.updated_at.desc())
        )
    return [_kb_to_dict(kb) for kb in result.scalars().all()]


class CreateCourseBody(BaseModel):
    course_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(..., min_length=1, max_length=256)
    description: str = ""
    icon: str = "📘"
    system_prompt: str = ""
    is_visible: bool = True


@router.post("/courses", status_code=201)
async def create_course(
    body: CreateCourseBody,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.course_id == body.course_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"课程 '{body.course_id}' 已存在")

    kb = KnowledgeBase(
        course_id=body.course_id,
        name=body.name,
        description=body.description,
        icon=body.icon,
        system_prompt=body.system_prompt,
        is_visible=body.is_visible,
        owner_id=teacher["id"],
    )
    db.add(kb)
    await db.flush()
    _kb_raw_dir(body.course_id).mkdir(parents=True, exist_ok=True)
    logger.info("教师创建课程 course_id=%s owner=%s", body.course_id, teacher["id"])
    await invalidate_courses_cache()
    return _kb_to_dict(kb)


class UpdateCourseBody(BaseModel):
    name: str | None = None
    description: str | None = None
    icon: str | None = None
    system_prompt: str | None = None
    is_visible: bool | None = None


@router.put("/courses/{course_id}")
async def update_course(
    course_id: str,
    body: UpdateCourseBody,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    for field in ("name", "description", "icon", "system_prompt", "is_visible"):
        val = getattr(body, field)
        if val is not None:
            setattr(kb, field, val)
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()
    return _kb_to_dict(kb)


# ── 知识库文件上传 ─────────────────────────────────────────────────────────────

@router.post("/courses/{course_id}/upload")
async def upload_files(
    course_id: str,
    files: list[UploadFile] = File(...),
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    raw_dir = _kb_raw_dir(course_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in _ALLOWED_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型 '{ext}'，允许：{', '.join(_ALLOWED_EXT)}",
            )
        content = await file.read()
        if len(content) > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件 '{file.filename}' 超过 {MAX_KB_UPLOAD_MB} MB",
            )
        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        (raw_dir / safe_name).write_bytes(content)
        db.add(KBFile(
            kb_id=kb.id,
            original_name=file.filename or safe_name,
            file_path=str(raw_dir / safe_name),
            file_size=len(content),
        ))
        saved.append(file.filename or safe_name)

    await db.flush()
    count_result = await db.execute(
        select(func.count()).select_from(KBFile).where(KBFile.kb_id == kb.id)
    )
    kb.file_count = count_result.scalar_one()
    kb.updated_at = time.time()
    if kb.status == "ready":
        kb.status = "pending"
    return {"uploaded": saved, "total_files": kb.file_count}


# ── 触发索引 ──────────────────────────────────────────────────────────────────

@router.post("/courses/{course_id}/index")
async def index_course(
    course_id: str,
    background_tasks: BackgroundTasks,
    force: bool = False,
    resume: bool = False,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    if kb.status == "indexing" and not force:
        raise HTTPException(status_code=409, detail="正在索引中")

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="请先上传文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件不存在，请重新上传")

    resume_from = 0
    if resume and kb.status in ("error", "paused") and kb.chunks_done > 0:
        resume_from = kb.chunks_done

    background_tasks.add_task(_run_indexing, kb.id, course_id, file_paths, resume_from)
    return {
        "message": "索引任务已启动" if resume_from == 0 else f"续传（从第 {resume_from} 块）",
        "course_id": course_id,
        "file_count": len(file_paths),
    }


# ── 学生管理（选课） ──────────────────────────────────────────────────────────

@router.get("/courses/{course_id}/students")
async def list_students(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_kb(db, course_id, teacher)
    result = await db.execute(
        select(User.id, User.username, User.display_name, Enrollment.created_at)
        .join(Enrollment, Enrollment.student_id == User.id)
        .where(Enrollment.course_id == course_id)
        .order_by(Enrollment.created_at.desc())
    )
    return [
        {
            "id": row.id,
            "username": row.username,
            "display_name": row.display_name,
            "enrolled_at": row.created_at,
        }
        for row in result.all()
    ]


class EnrollBody(BaseModel):
    username: str = Field(..., min_length=1)


@router.post("/courses/{course_id}/students", status_code=201)
async def enroll_student(
    course_id: str,
    body: EnrollBody,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_kb(db, course_id, teacher)

    result = await db.execute(select(User).where(User.username == body.username))
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail=f"用户 '{body.username}' 不存在")
    if student.role not in ("student",):
        raise HTTPException(status_code=400, detail="只能添加学生角色的用户")

    exists = await db.execute(
        select(Enrollment).where(
            Enrollment.student_id == student.id,
            Enrollment.course_id == course_id,
        )
    )
    if exists.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="该学生已选此课程")

    db.add(Enrollment(student_id=student.id, course_id=course_id))
    await db.flush()
    logger.info("教师添加学生 student=%s course=%s", student.id, course_id)
    return {"message": f"已添加 {body.username}", "student_id": student.id}


@router.delete("/courses/{course_id}/students/{student_id}")
async def remove_student(
    course_id: str,
    student_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    await _get_owned_kb(db, course_id, teacher)
    result = await db.execute(
        select(Enrollment).where(
            Enrollment.student_id == student_id,
            Enrollment.course_id == course_id,
        )
    )
    enrollment = result.scalar_one_or_none()
    if not enrollment:
        raise HTTPException(status_code=404, detail="选课记录不存在")
    await db.delete(enrollment)
    return {"message": "已移除"}


# ── 课程码管理 ─────────────────────────────────────────────────────────────────

def _gen_join_code() -> str:
    """生成 8 位大写字母数字课程码。"""
    return uuid.uuid4().hex[:8].upper()


@router.post("/courses/{course_id}/join-code")
async def refresh_join_code(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """生成或重置课程码，返回新码。"""
    kb = await _get_owned_kb(db, course_id, teacher)
    # 确保唯一（极低概率重复，循环至多 3 次）
    for _ in range(3):
        code = _gen_join_code()
        clash = await db.execute(
            select(KnowledgeBase.id).where(
                KnowledgeBase.join_code == code,
                KnowledgeBase.course_id != course_id,
            )
        )
        if not clash.first():
            break
    kb.join_code = code
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()
    return {"course_id": course_id, "join_code": code}


# ── 知识库详情（含文件列表） ──────────────────────────────────────────────────

@router.get("/courses/{course_id}")
async def get_course_detail(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    files_result = await db.execute(
        select(KBFile).where(KBFile.kb_id == kb.id).order_by(KBFile.created_at.desc())
    )
    files = files_result.scalars().all()
    data = _kb_to_dict(kb)
    data["files"] = [
        {"id": f.id, "original_name": f.original_name, "file_size": f.file_size,
         "status": f.status, "error_msg": f.error_msg, "created_at": f.created_at}
        for f in files
    ]
    return data


@router.patch("/courses/{course_id}")
async def patch_course(
    course_id: str,
    body: UpdateCourseBody,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    for field in ("name", "description", "icon", "system_prompt", "is_visible"):
        val = getattr(body, field)
        if val is not None:
            setattr(kb, field, val)
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()
    return _kb_to_dict(kb)


@router.delete("/courses/{course_id}")
async def delete_course(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    await db.delete(kb)
    kb_dir = Path(KB_STORE_DIR) / course_id
    if kb_dir.exists():
        shutil.rmtree(kb_dir, ignore_errors=True)
    logger.info("教师删除课程 course_id=%s", course_id)
    await invalidate_courses_cache()
    return {"message": f"课程 '{course_id}' 已删除"}


@router.delete("/courses/{course_id}/files/{file_id}")
async def delete_course_file(
    course_id: str,
    file_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    file_result = await db.execute(
        select(KBFile).where(KBFile.id == file_id, KBFile.kb_id == kb.id)
    )
    f = file_result.scalar_one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="文件不存在")
    fp = Path(f.file_path)
    if fp.exists():
        fp.unlink(missing_ok=True)
    await db.delete(f)
    await db.flush()
    count_result = await db.execute(
        select(func.count()).select_from(KBFile).where(KBFile.kb_id == kb.id)
    )
    kb.file_count = count_result.scalar_one()
    kb.updated_at = time.time()
    return {"message": "文件已删除", "remaining_files": kb.file_count}


@router.post("/courses/{course_id}/index/pause")
async def pause_course_index(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    if kb.status != "indexing":
        raise HTTPException(status_code=409, detail="当前没有正在进行的索引任务")
    ctrl = IndexingControl(kb.id)
    try:
        await ctrl.request_pause()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法下发暂停信号：{e}")
    kb.progress_msg = "暂停请求已发送，等待当前批次完成…"
    kb.updated_at = time.time()
    return {"message": "暂停请求已发送", "course_id": course_id}


@router.post("/courses/{course_id}/index/stop")
async def stop_course_index(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    kb = await _get_owned_kb(db, course_id, teacher)
    if kb.status == "indexing":
        ctrl = IndexingControl(kb.id)
        try:
            await ctrl.request_stop()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"无法下发终止信号：{e}")
        kb.progress_msg = "终止请求已发送，等待当前批次完成…"
        kb.updated_at = time.time()
        return {"message": "终止请求已发送", "course_id": course_id}
    if kb.status == "paused":
        kb.status = "pending"
        kb.progress = 0
        kb.progress_msg = "已终止"
        kb.chunks_done = 0
        kb.chunks_total = 0
        kb.token_estimate = 0
        kb.error_msg = ""
        kb.updated_at = time.time()
        await invalidate_courses_cache()
        return {"message": "已终止并清除进度", "course_id": course_id}
    raise HTTPException(status_code=409, detail="当前状态不可终止")


# ── LlamaIndex 向量索引构建 ───────────────────────────────────────────────────

@router.post("/courses/{course_id}/llamaindex/build")
async def build_llamaindex_index(
    course_id: str,
    background_tasks: BackgroundTasks,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """触发 LlamaIndex 向量索引构建（后台任务）。"""
    from api.llama_rag import _run_llamaindex_build  # 避免循环导入

    kb = await _get_owned_kb(db, course_id, teacher)

    files_result = await db.execute(select(KBFile).where(KBFile.kb_id == kb.id))
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    if kb.status == "indexing":
        raise HTTPException(status_code=409, detail="知识库正在索引中，请稍候完成后再试")

    kb.status = "indexing"
    kb.progress = 0
    kb.error_msg = ""
    kb.progress_msg = "LlamaIndex 向量索引构建中\u2026"
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()

    background_tasks.add_task(_run_llamaindex_build, kb.id, course_id, file_paths)

    return {"accepted": True, "message": "LlamaIndex 构建已提交后台", "course_id": course_id}
