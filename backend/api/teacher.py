"""Teacher API: course CRUD, KB upload/index, student enrollment, analytics."""
from __future__ import annotations

import datetime
import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import Integer, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_teacher
from api.admin import _kb_to_dict, _ALLOWED_EXT, _MAX_BYTES, _safe_upload_name
from api.courses import invalidate_courses_cache
from api.kb_indexing import trigger_kb_indexing, trigger_llamaindex_build
from settings import get_settings
KB_STORE_DIR = get_settings().paths.kb_store_dir
MAX_KB_UPLOAD_MB = get_settings().max_kb_upload_mb
from core.db.cache import course_access_invalidate, faq_top
from core.db.limiter import limiter
from core.codes import ensure_unique_join_code
from core.db.database import (
    Enrollment, KBFile, KnowledgeBase, Message,
    NotebookEntry, Session, User, get_db,
)
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
    # 建课即自动生成课程码（替代旧的"教师手动点生成"）
    kb.join_code = await ensure_unique_join_code(db)
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
        # M-51：清洗客户端文件名（防 ../ 穿越 / 控制字符），再拼 uuid 前缀落盘
        clean_name = _safe_upload_name(file.filename)
        safe_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"
        (raw_dir / safe_name).write_bytes(content)
        db.add(KBFile(
            kb_id=kb.id,
            original_name=clean_name,
            file_path=str(raw_dir / safe_name),
            file_size=len(content),
        ))
        saved.append(clean_name)

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
@limiter.limit("6/minute")
async def index_course(
    course_id: str,
    request: Request,
    force: bool = False,
    resume: bool = False,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """触发知识库索引（ARQ 后台任务）。公共逻辑见 api.kb_indexing.trigger_kb_indexing。"""
    kb = await _get_owned_kb(db, course_id, teacher)
    return await trigger_kb_indexing(db, kb, course_id, force, resume)


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
    # M-43：学生此前可能因未选课被缓存为"拒绝"（access:{sid}:{cid}=0，TTL 5min）。
    # 教师此刻把学生加进来后必须立即失效该 deny 缓存，否则学生在 5 分钟窗口内仍被拒，
    # 表现为"老师刚加我我却进不去"。
    await course_access_invalidate(student.id, course_id)
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
    # H-6：删除 enrollment 后必须失效该学生的课程访问缓存（access:{sid}:{cid}）。
    # 否则权限缓存（TTL 5 分钟）仍记着"允许"，被踢的学生在窗口期内仍能继续访问课程。
    await course_access_invalidate(student_id, course_id)
    return {"message": "已移除"}


# ── 课程码管理 ─────────────────────────────────────────────────────────────────

@router.post("/courses/{course_id}/join-code")
async def refresh_join_code(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """生成或重置课程码，返回新码。"""
    kb = await _get_owned_kb(db, course_id, teacher)
    kb.join_code = await ensure_unique_join_code(db, exclude_course_id=course_id)
    kb.updated_at = time.time()
    await db.flush()
    await invalidate_courses_cache()
    return {"course_id": course_id, "join_code": kb.join_code}


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
        await ctrl.request_pause()  # 通知 worker；事件循环被阻塞读不到也无妨
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法下发暂停信号：{e}")
    # 立即落库为 paused：超大文档的 ainsert 会长时间阻塞 worker 事件循环，导致
    # checkpoint 永远读不到信号、前端卡在"请求已发送"。这里直接置 paused 让前端
    # 立即解脱；worker 终态写入有"不覆盖非 indexing 状态"保护，跑完不会回写。
    done = kb.chunks_done or 0
    total = kb.chunks_total or 0
    kb.status = "paused"
    kb.progress_msg = f"已暂停（已完成 {done}{f'/{total}' if total else ''} 个文本块）"
    kb.updated_at = time.time()
    await invalidate_courses_cache()
    # 不清 Redis 控制信号：留给 worker 的 checkpoint 读到后自行 cancel 停止，
    # worker 终止后会在 finally 里 control.clear()。这里清了反而让 worker 读不到、继续跑。
    return {"message": "已暂停", "course_id": course_id}


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
            await ctrl.request_stop()  # 通知 worker；事件循环被阻塞读不到也无妨
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"无法下发终止信号：{e}")
        # 立即落库为 pending 并清零进度：见 pause 的说明。worker 跑完不会回写。
        kb.status = "pending"
        kb.progress = 0
        kb.progress_msg = "已终止"
        kb.chunks_done = 0
        kb.chunks_total = 0
        kb.token_estimate = 0
        kb.error_msg = ""
        kb.updated_at = time.time()
        await invalidate_courses_cache()
        # 不清 Redis 控制信号：留给 worker 的 checkpoint 读到后自行 cancel 停止，
        # worker 终止后会在 finally 里 control.clear()。这里清了反而让 worker 读不到、继续跑。
        return {"message": "已终止", "course_id": course_id}
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
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """触发 LlamaIndex 向量索引构建（后台任务）。公共逻辑见 api.kb_indexing.trigger_llamaindex_build。"""
    kb = await _get_owned_kb(db, course_id, teacher)
    return await trigger_llamaindex_build(db, kb, course_id)


# ══════════════════════════════════════════════════════════════════════════════
# Analytics endpoints
# ══════════════════════════════════════════════════════════════════════════════

def _today_start() -> float:
    """Return timestamp of 00:00 today (local timezone)."""
    today = datetime.date.today()
    return datetime.datetime(today.year, today.month, today.day).timestamp()


@router.get("/courses/{course_id}/analytics/overview")
async def analytics_overview(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Activity overview for a course: totals + 7-day trend."""
    await _get_owned_kb(db, course_id, teacher)

    today_ts = _today_start()
    week_ago_ts = today_ts - 86400 * 7

    total_students = (await db.execute(
        select(func.count()).select_from(Enrollment).where(Enrollment.course_id == course_id)
    )).scalar_one()

    total_sessions = (await db.execute(
        select(func.count()).select_from(Session).where(Session.course_id == course_id)
    )).scalar_one()

    total_messages = (await db.execute(
        select(func.count()).select_from(Message)
        .join(Session, Message.session_id == Session.id)
        .where(Session.course_id == course_id, Message.role == "user")
    )).scalar_one()

    today_questions = (await db.execute(
        select(func.count()).select_from(Message)
        .join(Session, Message.session_id == Session.id)
        .where(Session.course_id == course_id, Message.role == "user",
               Message.created_at >= today_ts)
    )).scalar_one()

    today_active = (await db.execute(
        select(func.count(func.distinct(Session.user_id)))
        .where(Session.course_id == course_id, Session.updated_at >= today_ts)
    )).scalar_one()

    # 7-day daily trend
    recent_msgs = (await db.execute(
        select(Message.created_at)
        .join(Session, Message.session_id == Session.id)
        .where(Session.course_id == course_id, Message.role == "user",
               Message.created_at >= week_ago_ts)
    )).scalars().all()

    buckets: dict[str, int] = {}
    for i in range(7):
        d = datetime.date.today() - datetime.timedelta(days=6 - i)
        buckets[d.isoformat()] = 0
    for ts in recent_msgs:
        d = datetime.date.fromtimestamp(ts).isoformat()
        if d in buckets:
            buckets[d] += 1

    return {
        "total_students": total_students,
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "today_questions": today_questions,
        "today_active_students": today_active,
        "daily_trend": [{"date": d, "count": c} for d, c in buckets.items()],
    }


@router.get("/courses/{course_id}/analytics/frequent-questions")
async def analytics_frequent_questions(
    course_id: str,
    top_n: int = Query(20, ge=1, le=100),
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Top-N most asked questions (Redis FAQ + SQL fallback)."""
    await _get_owned_kb(db, course_id, teacher)

    redis_items = await faq_top(course_id, top_n)

    # SQL complement: recent user messages grouped by content prefix
    thirty_days_ago = time.time() - 86400 * 30
    # PG 严格模式要求 SELECT 与 GROUP BY 用同一表达式；func.left(...,80) 写两遍会被
    # asyncpg 编译成两个独立 bindparam（$1 vs $6），PG 无法证明相等 → GroupingError。
    # 提取为同一表达式对象，让 SQLAlchemy 复用同一个 bindparam。
    _content_prefix = func.left(Message.content, 80)
    sql_rows = (await db.execute(
        select(
            _content_prefix.label("q"),
            func.count().label("cnt"),
            func.max(Message.created_at).label("last_ts"),
        )
        .join(Session, Message.session_id == Session.id)
        .where(
            Session.course_id == course_id,
            Message.role == "user",
            Message.created_at >= thirty_days_ago,
            func.length(Message.content) > 4,
        )
        .group_by(_content_prefix)
        .having(func.count() >= 2)
        .order_by(func.count().desc())
        .limit(top_n)
    )).all()

    # Merge: Redis items take priority, SQL fills gaps
    seen = {item["question"].strip().lower()[:80] for item in redis_items}
    merged = [
        {"question": item["question"], "count": item["count"], "last_asked": None}
        for item in redis_items
    ]
    for row in sql_rows:
        key = row.q.strip().lower()[:80]
        if key not in seen:
            merged.append({"question": row.q, "count": row.cnt, "last_asked": row.last_ts})
            seen.add(key)
        if len(merged) >= top_n:
            break

    merged.sort(key=lambda x: -x["count"])
    return {"questions": merged[:top_n]}


@router.get("/courses/{course_id}/analytics/student-chats")
async def analytics_student_chats(
    course_id: str,
    student_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Student session list with message counts."""
    await _get_owned_kb(db, course_id, teacher)

    base = (
        select(
            Session.id.label("session_id"),
            Session.title,
            Session.user_id,
            Session.created_at.label("session_created"),
            Session.updated_at.label("session_updated"),
            func.count(Message.id).label("message_count"),
            func.max(Message.created_at).label("last_message_at"),
        )
        .outerjoin(Message, Message.session_id == Session.id)
        .where(Session.course_id == course_id)
    )
    if student_id:
        base = base.where(Session.user_id == student_id)

    base = (
        base.group_by(Session.id, Session.title, Session.user_id,
                       Session.created_at, Session.updated_at)
        .order_by(Session.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    rows = (await db.execute(base)).all()

    user_ids = list({r.user_id for r in rows if r.user_id})
    user_map: dict[str, dict] = {}
    if user_ids:
        users = (await db.execute(
            select(User.id, User.username, User.display_name).where(User.id.in_(user_ids))
        )).all()
        user_map = {u.id: {"id": u.id, "username": u.username, "display_name": u.display_name} for u in users}

    return {
        "sessions": [
            {
                "session_id": r.session_id,
                "title": r.title,
                "student": user_map.get(r.user_id, {"id": r.user_id, "username": "", "display_name": ""}),
                "message_count": r.message_count,
                "last_message_at": r.last_message_at,
                "created_at": r.session_created,
            }
            for r in rows
        ],
        "page": page,
        "page_size": page_size,
    }


@router.get("/courses/{course_id}/analytics/sessions/{session_id}/messages")
async def analytics_session_messages(
    course_id: str,
    session_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Full message history of a single student session."""
    await _get_owned_kb(db, course_id, teacher)

    session = (await db.execute(
        select(Session).where(Session.id == session_id, Session.course_id == course_id)
    )).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在或不属于此课程")

    msgs = (await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )).scalars().all()

    return {
        "session_id": session_id,
        "title": session.title,
        "user_id": session.user_id,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "msg_type": m.msg_type,
                "created_at": m.created_at,
            }
            for m in msgs
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Student-learning stats (per-course aggregate + per-student detail)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_graph_counts(knowledge_graph: dict | None, error_graph: dict | None):
    """Extract node-level stats from JSON graph columns."""
    kg = knowledge_graph or {}
    eg = error_graph or {}
    nodes = kg.get("nodes", []) if isinstance(kg, dict) else []
    error_nodes = eg.get("nodes", []) if isinstance(eg, dict) else []
    knowledge_node_count = len(nodes)
    error_node_count = len(error_nodes)
    high_risk_count = sum(1 for n in nodes if (n.get("risk") or 0) > 0.6)
    return knowledge_node_count, error_node_count, high_risk_count


@router.get("/courses/{course_id}/analytics/student-stats")
async def analytics_student_stats(
    course_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Per-course student learning stats aggregate."""
    kb = await _get_owned_kb(db, course_id, teacher)

    today_ts = _today_start()
    week_ago_ts = today_ts - 86400 * 7

    # 1. All enrolled students with their graph data
    enroll_rows = (await db.execute(
        select(Enrollment.student_id, User.username, User.display_name,
               User.knowledge_graph, User.error_graph)
        .join(User, User.id == Enrollment.student_id)
        .where(Enrollment.course_id == course_id)
    )).all()

    if not enroll_rows:
        return {
            "course_name": kb.name,
            "total_students": 0,
            "student_summaries": [],
            "accuracy_distribution": {},
            "high_risk_students": [],
            "daily_active_trend": [],
        }

    student_ids = [r.student_id for r in enroll_rows]

    # 2. Session counts per student
    session_rows = (await db.execute(
        select(Session.user_id, func.count(Session.id).label("cnt"))
        .where(Session.course_id == course_id, Session.user_id.in_(student_ids))
        .group_by(Session.user_id)
    )).all()
    session_map: dict[str, int] = {r.user_id: r.cnt for r in session_rows}

    # 3. Message counts per student (user-role only)
    msg_rows = (await db.execute(
        select(Session.user_id, func.count(Message.id).label("cnt"))
        .join(Session, Message.session_id == Session.id)
        .where(Session.course_id == course_id, Message.role == "user",
               Session.user_id.in_(student_ids))
        .group_by(Session.user_id)
    )).all()
    msg_map: dict[str, int] = {r.user_id: r.cnt for r in msg_rows}

    # 4. Last active per student
    last_active_rows = (await db.execute(
        select(Session.user_id, func.max(Session.updated_at).label("last"))
        .where(Session.course_id == course_id, Session.user_id.in_(student_ids))
        .group_by(Session.user_id)
    )).all()
    last_active_map: dict[str, float] = {r.user_id: r.last for r in last_active_rows}

    # 5. Quiz stats per student (NotebookEntry via session → course)
    quiz_rows = (await db.execute(
        select(NotebookEntry.user_id,
               func.count(NotebookEntry.id).label("total"),
               func.sum(func.cast(NotebookEntry.is_correct, Integer)).label("correct"))
        .where(NotebookEntry.user_id.in_(student_ids))
        .group_by(NotebookEntry.user_id)
    )).all()
    quiz_map: dict[str, dict] = {r.user_id: {"total": r.total, "correct": r.correct or 0} for r in quiz_rows}

    # 6. Build per-student summaries
    student_summaries = []
    for r in enroll_rows:
        kn_cnt, err_cnt, hr_cnt = _parse_graph_counts(r.knowledge_graph, r.error_graph)
        q = quiz_map.get(r.student_id, {"total": 0, "correct": 0})
        acc = q["correct"] / q["total"] if q["total"] > 0 else 0.5
        risk_score = 0.5 * (hr_cnt / max(kn_cnt, 1)) + 0.5 * (1 - acc)
        student_summaries.append({
            "student_id": r.student_id,
            "username": r.username,
            "display_name": r.display_name,
            "total_sessions": session_map.get(r.student_id, 0),
            "total_messages": msg_map.get(r.student_id, 0),
            "total_questions": q["total"],
            "correct_count": q["correct"],
            "accuracy_rate": round(acc, 3),
            "last_active_at": last_active_map.get(r.student_id),
            "knowledge_node_count": kn_cnt,
            "high_risk_count": hr_cnt,
            "error_node_count": err_cnt,
            "risk_score": round(risk_score, 3),
        })

    # 7. Accuracy distribution (5 buckets)
    buckets = {"0_20": 0, "20_40": 0, "40_60": 0, "60_80": 0, "80_100": 0}
    for s in student_summaries:
        pct = s["accuracy_rate"] * 100
        if pct < 20:
            buckets["0_20"] += 1
        elif pct < 40:
            buckets["20_40"] += 1
        elif pct < 60:
            buckets["40_60"] += 1
        elif pct < 80:
            buckets["60_80"] += 1
        else:
            buckets["80_100"] += 1

    # 8. High-risk students (risk_score > 0.6)
    high_risk_students = []
    for s in student_summaries:
        if s["risk_score"] > 0.6:
            reasons = []
            if s["high_risk_count"] > 0:
                reasons.append(f"高风险知识点 {s['high_risk_count']} 个")
            if s["accuracy_rate"] < 0.5:
                reasons.append(f"答题正确率 {int(s['accuracy_rate'] * 100)}%")
            if s["total_sessions"] == 0:
                reasons.append("无学习记录")
            high_risk_students.append({
                "student_id": s["student_id"],
                "display_name": s["display_name"] or s["username"],
                "risk_score": s["risk_score"],
                "reasons": reasons or ["综合风险较高"],
            })

    # 9. 7-day active student trend
    recent_sessions = (await db.execute(
        select(Session.user_id, Session.updated_at)
        .where(Session.course_id == course_id, Session.updated_at >= week_ago_ts)
    )).all()
    day_buckets: dict[str, set] = {}
    for i in range(7):
        d = (datetime.date.today() - datetime.timedelta(days=6 - i)).isoformat()
        day_buckets[d] = set()
    for uid, ts in recent_sessions:
        d = datetime.date.fromtimestamp(ts).isoformat()
        if d in day_buckets:
            day_buckets[d].add(uid)
    daily_active_trend = [{"date": d, "active_count": len(uids)} for d, uids in day_buckets.items()]

    return {
        "course_name": kb.name,
        "total_students": len(student_summaries),
        "student_summaries": student_summaries,
        "accuracy_distribution": buckets,
        "high_risk_students": high_risk_students,
        "daily_active_trend": daily_active_trend,
    }


@router.get("/courses/{course_id}/analytics/student/{student_id}/detail")
async def analytics_student_detail(
    course_id: str,
    student_id: str,
    teacher: dict = Depends(get_current_teacher),
    db: AsyncSession = Depends(get_db),
):
    """Detailed learning data for a single student in a course."""
    await _get_owned_kb(db, course_id, teacher)

    # Verify student is enrolled
    enrolled = (await db.execute(
        select(Enrollment).where(
            Enrollment.course_id == course_id, Enrollment.student_id == student_id
        )
    )).scalar_one_or_none()
    if not enrolled:
        raise HTTPException(status_code=404, detail="该学生未选此课程")

    # Student basic info + graphs
    user = (await db.execute(
        select(User).where(User.id == student_id)
    )).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="学生不存在")

    # Session & message counts for this course
    session_count = (await db.execute(
        select(func.count()).select_from(Session)
        .where(Session.course_id == course_id, Session.user_id == student_id)
    )).scalar_one()

    message_count = (await db.execute(
        select(func.count()).select_from(Message)
        .join(Session, Message.session_id == Session.id)
        .where(Session.course_id == course_id, Session.user_id == student_id, Message.role == "user")
    )).scalar_one()

    # Quiz stats for this course (via session)
    course_session_ids = (await db.execute(
        select(Session.id).where(Session.course_id == course_id, Session.user_id == student_id)
    )).scalars().all()

    questions_total = 0
    questions_correct = 0
    recent_questions: list[dict] = []
    if course_session_ids:
        quiz_stats = (await db.execute(
            select(func.count(NotebookEntry.id).label("total"),
                   func.sum(func.cast(NotebookEntry.is_correct, Integer)).label("correct"))
            .where(NotebookEntry.session_id.in_(course_session_ids))
        )).one()
        questions_total = quiz_stats.total or 0
        questions_correct = quiz_stats.correct or 0

        recent = (await db.execute(
            select(NotebookEntry)
            .where(NotebookEntry.session_id.in_(course_session_ids))
            .order_by(NotebookEntry.created_at.desc())
            .limit(10)
        )).scalars().all()
        recent_questions = [
            {
                "question": q.question[:200],
                "is_correct": q.is_correct,
                "difficulty": q.difficulty,
                "created_at": q.created_at,
            }
            for q in recent
        ]

    accuracy_rate = questions_correct / questions_total if questions_total > 0 else None

    # Graph data
    kg = user.knowledge_graph if isinstance(user.knowledge_graph, dict) else {"nodes": [], "edges": []}
    eg = user.error_graph if isinstance(user.error_graph, dict) else {"nodes": [], "edges": []}

    return {
        "student": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
        },
        "sessions": session_count,
        "messages": message_count,
        "questions_total": questions_total,
        "questions_correct": questions_correct,
        "accuracy_rate": round(accuracy_rate, 3) if accuracy_rate is not None else None,
        "knowledge_graph": kg,
        "error_graph": eg,
        "recent_questions": recent_questions,
    }
