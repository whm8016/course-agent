"""Admin API：知识库管理 & 用户管理（仅管理员可访问）。"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_admin
from core.db.cache import invalidate_courses_cache
from api.kb_indexing import (
    get_topic_graph_payload,
    trigger_kb_indexing,
    trigger_topic_graph_build,
)
from settings import get_settings
FAQ_CACHE_THRESHOLD = get_settings().question.faq_cache_threshold
KB_STORE_DIR = get_settings().paths.kb_store_dir
from core.rag import get_indexer, is_lightrag_available
from core.rag.es_client import delete_course_es_docs
from core.rag.lightrag import purge_course_workspace
from core.db.limiter import limiter
from core.llm.prompts import invalidate_course_prompt_cache
from core.codes import JoinCodeExhaustedError, ensure_unique_join_code, generate_code
from core.db.database import (
    ApplicationStatus,
    BotNotification,
    KBFile,
    KnowledgeBase,
    TeacherApplication,
    TeacherInvite,
    User,
    get_db,
)
from core.analytics.faq import frequent_questions_merged

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# 注：暂停/终止控制信号通过 Redis 跨 worker 传递，
# 不再依赖进程内字典。详见 core.ingestion.IndexingControl。


# ── 辅助函数：已下沉 api/kb_shared.py（admin/teacher 去重），旧名 re-export ─────
from api.kb_shared import (  # noqa: E402,F401
    ALLOWED_EXT as _ALLOWED_EXT,
    MAX_BYTES as _MAX_BYTES,
    kb_raw_dir as _kb_raw_dir,
    kb_to_dict as _kb_to_dict,
    safe_upload_name as _safe_upload_name,
)
from api.kb_shared import delete_kb_file, pause_index_build, save_uploaded_files, stop_index_build


async def _get_kb_or_404(db: AsyncSession, course_id: str) -> KnowledgeBase:
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail=f"知识库 '{course_id}' 不存在")
    return kb


# ── 后台索引任务 ──────────────────────────────────────────────────────────────
# run_indexing/run_indexing_llamaindex_pg 已归位 core/rag/indexing_runner.py
# （纯业务无 HTTP 概念；此前住在这里导致 worker 反向 import api 层）。
# 保留旧名 re-export：测试（test_kb_builds）与历史调用方仍从 api.admin import。
from core.rag.indexing_runner import (  # noqa: E402,F401
    run_indexing as _run_indexing,
    run_indexing_llamaindex_pg as _run_indexing_llamaindex_pg,
)

# ── 系统信息 ──────────────────────────────────────────────────────────────────

@router.get("/info")
async def admin_info(_: dict = Depends(get_current_admin)):
    """返回管理后台基本信息。"""
    rag_ok, rag_reason = is_lightrag_available()
    return {
        "lightrag_available": rag_ok,
        "lightrag_reason": rag_reason if not rag_ok else "",
        "kb_store_dir": KB_STORE_DIR,
    }


# ── 知识库 CRUD ───────────────────────────────────────────────────────────────

@router.get("/kb")
async def list_kbs(
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出所有知识库。"""
    result = await db.execute(
        select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
    )
    return [_kb_to_dict(kb) for kb in result.scalars().all()]


class CreateKBBody(BaseModel):
    course_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_\-]+$")
    name: str = Field(..., min_length=1, max_length=256)
    description: str = ""
    icon: str = "📘"
    system_prompt: str = ""
    sort_order: int = 0
    is_visible: bool = True
    # 索引后端：lightrag（默认，知识图谱，慢但支持多跳）| llamaindex_pg（pgvector 快速向量，分钟级索引）
    index_backend: str = "lightrag"

@router.post("/kb", status_code=201)
async def create_kb(
    body: CreateKBBody,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """创建新知识库（不上传文件）。"""
    existing = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.course_id == body.course_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"课程 '{body.course_id}' 的知识库已存在")

    kb = KnowledgeBase(
        course_id=body.course_id,
        name=body.name,
        description=body.description,
        icon=body.icon,
        system_prompt=body.system_prompt,
        sort_order=body.sort_order,
        is_visible=body.is_visible,
        index_backend=body.index_backend,
    )
    db.add(kb)
    await db.flush()
    # 与教师建课一致：建库即自动生成课程码（管理员库历史漏生成，前端凭此渲染二维码）
    try:
        kb.join_code = await ensure_unique_join_code(db)
    except JoinCodeExhaustedError as e:
        raise HTTPException(status_code=500, detail=str(e))
    await db.flush()

    _kb_raw_dir(body.course_id).mkdir(parents=True, exist_ok=True)
    logger.info("创建知识库 course_id=%s", body.course_id)
    await invalidate_courses_cache()
    # 新建 KB 尚无 builds；显式加载避免 _kb_to_dict 访问 kb.builds 触发 async lazy-load（MissingGreenlet）
    await db.refresh(kb, ["builds"])
    return _kb_to_dict(kb)


@router.get("/rag/engines")
async def list_rag_engines(_: dict = Depends(get_current_admin)):
    """索引后端 + 解析引擎能力探测（前端建库选择用）。

    托管引擎看 API key 是否配置；自托管引擎用 importlib 探测（对标 DeepTutor
    services/rag/factory.py 的两层 readiness）。
    """
    from core.rag.parsing.registry import is_engine_available  # noqa: PLC0415
    from core.rag.registry import is_backend_available  # noqa: PLC0415

    index_backends = [
        {
            "id": "lightrag",
            "name": "LightRAG（知识图谱）",
            "description": "逐 chunk LLM 实体抽取，慢但支持多跳关系推理",
            "requires_api_key": False,
        },
        {
            "id": "llamaindex_pg",
            "name": "pgvector（快速向量）",
            "description": "embedding 批调用分钟级建索引，dense+sparse 融合检索",
            "requires_api_key": False,
        },
    ]
    for b in index_backends:
        ok, reason = is_backend_available(b["id"])
        b["configured"] = ok
        if not ok:
            b["reason"] = reason

    parse_cfg = get_settings().parsing
    parse_engines = [
        {
            "id": "mineru_api",
            "name": "MinerU API（托管，去 torch）",
            "description": "云端托管，不装 torch，公式/表格强项",
            "requires_api_key": True,
        },
        {
            "id": "docling",
            "name": "Docling（自托管）",
            "description": "本地版面/表格/OCR，需装 parse-docling extra",
            "requires_api_key": False,
        },
    ]
    for e in parse_engines:
        if e["id"] == "mineru_api":
            e["configured"] = bool(parse_cfg.mineru_api_key.get_secret_value())
        else:
            e["configured"] = is_engine_available(e["id"])

    return {"index_backends": index_backends, "parse_engines": parse_engines}


class UpdateKBBody(BaseModel):
    name: str | None = None
    description: str | None = None
    icon: str | None = None
    system_prompt: str | None = None
    sort_order: int | None = None
    is_visible: bool | None = None


@router.patch("/kb/{course_id}")
async def update_kb(
    course_id: str,
    body: UpdateKBBody,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """更新知识库元信息（名称、描述、图标、system_prompt、排序）。"""
    kb = await _get_kb_or_404(db, course_id)

    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    if body.icon is not None:
        kb.icon = body.icon
    if body.is_visible is not None:
        kb.is_visible = body.is_visible
    if body.system_prompt is not None:
        kb.system_prompt = body.system_prompt
    if body.sort_order is not None:
        kb.sort_order = body.sort_order
    kb.updated_at = time.time()

    await db.flush()
    logger.info("更新知识库 course_id=%s", course_id)
    await invalidate_courses_cache()
    await invalidate_course_prompt_cache(course_id)
    return _kb_to_dict(kb)


@router.get("/kb/{course_id}")
async def get_kb(
    course_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """获取知识库详情（含文件列表）。"""
    kb = await _get_kb_or_404(db, course_id)

    files_result = await db.execute(
        select(KBFile)
        .where(KBFile.kb_id == kb.id)
        .order_by(KBFile.created_at.desc())
    )
    files = files_result.scalars().all()

    data = _kb_to_dict(kb)
    data["files"] = [
        {
            "id": f.id,
            "original_name": f.original_name,
            "file_size": f.file_size,
            "status": f.status,
            "error_msg": f.error_msg,
            "created_at": f.created_at,
        }
        for f in files
    ]
    return data


@router.delete("/kb/{course_id}")
async def delete_kb(
    course_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除知识库（含磁盘文件 + 三处检索副本）。"""
    kb = await _get_kb_or_404(db, course_id)
    await db.delete(kb)

    kb_dir = Path(KB_STORE_DIR) / course_id
    if kb_dir.exists():
        shutil.rmtree(kb_dir, ignore_errors=True)

    # 三处检索副本也按 course_id 清：LightRAG 工作区（PG 行 + graphml + 实例池）、
    # pgvector 行、ES 文档。都是 best-effort——失败只告警不阻断删库（DB 行已删，
    # 前端不再展示该库；残留副本由重索引/存储 GC 兜底），避免清副本异常让删库 500。
    try:
        await purge_course_workspace(course_id)
    except Exception:
        logger.exception("删库后清 LightRAG 工作区失败 course=%s", course_id)
    try:
        await get_indexer("llamaindex_pg").delete(course_id)
    except Exception:
        logger.exception("删库后清 pgvector 行失败 course=%s", course_id)
    try:
        await delete_course_es_docs(course_id)
    except Exception:
        logger.exception("删库后清 ES 文档失败 course=%s", course_id)

    logger.info("删除知识库 course_id=%s", course_id)
    await invalidate_courses_cache()
    return {"message": f"知识库 '{course_id}' 已删除"}


# ── 文件上传 ──────────────────────────────────────────────────────────────────

@router.post("/kb/{course_id}/upload")
async def upload_files(
    course_id: str,
    files: list[UploadFile] = File(...),
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """上传文件到知识库（支持 PDF/DOCX/PPTX/TXT/MD）。公共逻辑见 api.kb_shared。"""
    kb = await _get_kb_or_404(db, course_id)
    return await save_uploaded_files(db, kb, course_id, files)


@router.delete("/kb/{course_id}/files/{file_id}")
async def delete_file(
    course_id: str,
    file_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除知识库中的单个文件。公共逻辑见 api.kb_shared。"""
    kb = await _get_kb_or_404(db, course_id)
    return await delete_kb_file(db, kb, file_id)


# ── 触发索引 ──────────────────────────────────────────────────────────────────

@router.post("/kb/{course_id}/index")
@limiter.limit("6/minute")
async def index_kb(
    course_id: str,
    request: Request,
    backend: str | None = None,
    force: bool = False,
    resume: bool = False,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """触发知识库索引（后台任务：LlamaIndex 解析 → LightRAG / pgvector 摄入）。

    - backend：lightrag | llamaindex_pg；未传则回退 kb.index_backend（兼容旧前端）。
    - force=true：强制重新索引（即使正在进行中）
    - resume=true：从上次中断位置续传（仅限 error 状态）

    公共逻辑（状态校验 / DB 预置 / ARQ 入队）见 api.kb_indexing.trigger_kb_indexing。
    """
    kb = await _get_kb_or_404(db, course_id)
    backend = backend or kb.index_backend or "lightrag"
    return await trigger_kb_indexing(db, kb, course_id, backend, force, resume)


@router.post("/kb/{course_id}/topic-graph")
@limiter.limit("6/minute")
async def build_topic_graph_kb(
    course_id: str,
    request: Request,
    force: bool = False,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """触发课程主题图构建（ARQ 后台任务）。公共逻辑见 api.kb_indexing.trigger_topic_graph_build。"""
    kb = await _get_kb_or_404(db, course_id)
    return await trigger_topic_graph_build(db, kb, course_id, force)


@router.get("/kb/{course_id}/topic-graph")
async def get_topic_graph_kb(
    course_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """返回课程主题图（主题数、边数、主题/边列表），供核对与审计。"""
    await _get_kb_or_404(db, course_id)
    return await get_topic_graph_payload(db, course_id)


@router.post("/kb/{course_id}/index/pause")
async def pause_index(
    course_id: str,
    backend: str = "lightrag",
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """请求暂停正在进行的索引（在下一个 batch 边界生效，进度可续传）。
    公共逻辑见 api.kb_shared.pause_index_build；admin 侧 503 文案带（Redis 异常）标注。
    """
    kb = await _get_kb_or_404(db, course_id)
    return await pause_index_build(db, kb, course_id, backend, err_hint="（Redis 异常）")


@router.post("/kb/{course_id}/index/stop")
async def stop_index(
    course_id: str,
    backend: str = "lightrag",
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """请求终止索引。公共逻辑见 api.kb_shared.stop_index_build。"""
    kb = await _get_kb_or_404(db, course_id)
    return await stop_index_build(db, kb, course_id, backend, err_hint="（Redis 异常）")


# ── 用户管理 ──────────────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出所有用户。"""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "display_name": u.display_name,
            "role": u.role,
            "is_admin": (u.role == "admin"),
            "created_at": u.created_at,
        }
        for u in users
    ]


class ChangeRoleBody(BaseModel):
    role: str = Field(..., pattern=r"^(student|teacher|admin)$")


@router.put("/users/{user_id}/role")
async def change_user_role(
    user_id: str,
    body: ChangeRoleBody,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """管理员直接修改用户角色。"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    user.role = body.role
    user.is_admin = (body.role == "admin")
    await db.flush()
    logger.info("管理员修改用户角色 user=%s role=%s", user_id, body.role)
    return {"id": user.id, "username": user.username, "role": user.role}


class CreateInviteBody(BaseModel):
    count: int = Field(1, ge=1, le=50)
    expires_hours: float | None = Field(None, ge=1)


@router.post("/invite-codes", status_code=201)
async def create_invite_codes(
    body: CreateInviteBody,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """批量生成教师邀请码。"""
    codes: list[str] = []
    now = time.time()
    expires_at = now + body.expires_hours * 3600 if body.expires_hours else None

    for _ in range(body.count):
        code = generate_code(8)
        invite = TeacherInvite(
            code=code,
            created_by=admin["id"],
            expires_at=expires_at,
        )
        db.add(invite)
        codes.append(code)

    await db.flush()
    logger.info("管理员生成 %d 个邀请码", body.count)
    return {"codes": codes, "expires_at": expires_at}


@router.get("/invite-codes")
async def list_invite_codes(
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出所有邀请码。"""
    result = await db.execute(
        select(TeacherInvite).order_by(TeacherInvite.created_at.desc())
    )
    invites = result.scalars().all()
    return [
        {
            "id": inv.id,
            "code": inv.code,
            "used_by": inv.used_by,
            "expires_at": inv.expires_at,
            "created_at": inv.created_at,
        }
        for inv in invites
    ]


# ── 教师申请审批 ───────────────────────────────────────────────────────────────

class ReviewApplicationBody(BaseModel):
    note: str = Field("", max_length=500)


@router.get("/teacher-applications")
async def list_teacher_applications(
    status: str | None = Query(None, pattern=r"^(pending|approved|rejected)$"),
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """列出教师申请（可按状态过滤），join User 取用户名/显示名。"""
    stmt = select(TeacherApplication, User.username, User.display_name).join(
        User, User.id == TeacherApplication.user_id
    )
    if status:
        stmt = stmt.where(TeacherApplication.status == status)
    stmt = stmt.order_by(TeacherApplication.created_at.desc())
    rows = (await db.execute(stmt)).all()
    return [
        {
            "id": app.id,
            "user_id": app.user_id,
            "username": username,
            "display_name": display_name,
            "reason": app.reason,
            "status": app.status,
            "reviewed_by": app.reviewed_by,
            "reviewed_at": app.reviewed_at,
            "review_note": app.review_note,
            "created_at": app.created_at,
        }
        for app, username, display_name in rows
    ]


async def _load_pending_application(
    db: AsyncSession, app_id: str
) -> TeacherApplication:
    """加载申请并校验状态机：仅 pending 可审批（显式守卫，终态不可逆）。"""
    result = await db.execute(
        select(TeacherApplication).where(TeacherApplication.id == app_id)
    )
    app = result.scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="申请不存在")
    if app.status != ApplicationStatus.PENDING.value:
        raise HTTPException(
            status_code=409,
            detail=f"申请当前状态为 {app.status}，无法审批（仅 pending 可操作）",
        )
    return app


@router.post("/teacher-applications/{app_id}/approve")
async def approve_teacher_application(
    app_id: str,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """通过申请：升 users.role=teacher + 写站内通知。

    单 session 内 app.status + user.role + BotNotification 三写同 commit，
    天然原子（get_db 依赖统一 commit；异常自动 rollback）。
    """
    app = await _load_pending_application(db, app_id)
    user_result = await db.execute(select(User).where(User.id == app.user_id))
    user = user_result.scalar_one()
    app.status = ApplicationStatus.APPROVED.value
    app.reviewed_by = admin["id"]
    app.reviewed_at = time.time()
    user.role = "teacher"
    user.is_admin = False
    db.add(
        BotNotification(
            user_id=app.user_id,
            bot_id="",  # 前端 bot_id 为空时显示"通知"
            content="✅ 您的教师申请已通过，现在可以使用教师功能。",
        )
    )
    await db.flush()
    logger.info("审批通过 app=%s user=%s → teacher", app_id, app.user_id)
    return {
        "id": app.id,
        "status": app.status,
        "user_id": app.user_id,
        "role": "teacher",
    }


@router.post("/teacher-applications/{app_id}/reject")
async def reject_teacher_application(
    app_id: str,
    body: ReviewApplicationBody,
    admin: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """拒绝申请：保持 student + 写拒绝通知（可附理由），允许后续重新申请。"""
    app = await _load_pending_application(db, app_id)
    app.status = ApplicationStatus.REJECTED.value
    app.reviewed_by = admin["id"]
    app.reviewed_at = time.time()
    app.review_note = body.note.strip()
    note_suffix = f"（理由：{body.note.strip()}）" if body.note.strip() else ""
    db.add(
        BotNotification(
            user_id=app.user_id,
            bot_id="",
            content=f"❌ 您的教师申请未通过{note_suffix}，可修改理由后重新申请。",
        )
    )
    await db.flush()
    logger.info("审批拒绝 app=%s user=%s", app_id, app.user_id)
    return {"id": app.id, "status": app.status}


# ── 高频问题看板 ───────────────────────────────────────────────────────────────

@router.get("/faq")
async def get_faq(
    course_id: str | None = None,
    top_n: int = 20,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """返回各课程 Top-N 高频问题列表（按次数降序）。
    - course_id 为空时，查询所有已知课程并合并返回。
    - Redis 命中优先，SQL 兜底（近 30 天重复提问 ≥2 次），与 teacher 端同口径。
    """
    if top_n < 1 or top_n > 100:
        top_n = 20

    if course_id:
        items = await frequent_questions_merged(db, course_id, top_n)
        return {"course_id": course_id, "threshold": FAQ_CACHE_THRESHOLD, "questions": items}

    # 遍历所有课程
    kb_result = await db.execute(select(KnowledgeBase.course_id, KnowledgeBase.name))
    courses = kb_result.all()
    all_items: list[dict] = []
    for cid, cname in courses:
        for item in await frequent_questions_merged(db, cid, top_n):
            all_items.append({"course_id": cid, "course_name": cname, **item})
    # 按 count 降序排列后取 top_n
    all_items.sort(key=lambda x: x["count"], reverse=True)
    return {"course_id": None, "threshold": FAQ_CACHE_THRESHOLD, "questions": all_items[:top_n]}


@router.get("/usage/summary")
async def llm_usage_summary(
    start: str | None = Query(None, description="起始日 YYYYMMDD（含），缺省取近 30 天"),
    end: str | None = Query(None, description="结束日 YYYYMMDD（含），缺省取今日"),
    group_by: str = Query("course", description="分组维度，逗号分隔：day|user|course|model"),
    limit: int = Query(50, ge=1, le=500),
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """LLM 用量汇总（只读 llm_usage_daily 聚合表）：按维度分组求和、按 cost 降序。

    start/end 为 "YYYYMMDD" 日串（含闭区间）；group_by 逗号分隔取 day|user|course|model（默认
    course）。数据由 cron 每小时 :11 重算今日+昨日，统计滞后 ≤1 小时。Prometheus Counter 无法按
    user_id 展开（label 基数爆炸），故「按人/按课查账」走本表而非 /metrics。
    """
    from datetime import datetime, timedelta, timezone

    from core.analytics.token_usage import query_usage

    now = datetime.now(timezone.utc)
    end_day = (end or now.strftime("%Y%m%d"))[:8]
    start_day = (start or (now - timedelta(days=30)).strftime("%Y%m%d"))[:8]
    dims = [d.strip() for d in group_by.split(",") if d.strip()] or ["course"]
    return await query_usage(
        db, start=start_day, end=end_day, group_by=dims, limit=limit
    )


# ── 错误流水（可观测性重整）────────────────────────────────────────────────────

@router.get("/errors/recent")
async def errors_recent(
    limit: int = Query(50, ge=1, le=500),
    code: str | None = Query(None, description="按错误信封 code 过滤（llm_unavailable/…）"),
    course_id: str | None = Query(None, description="按课程过滤"),
    ref: str | None = Query(None, description="学生报障引用号精确匹配（主排障路径）"),
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """最近失败 turn 流水（turn_errors 表，倒序）。

    学生报障主路径：错误气泡带的 8 位编号 → ?ref= 精确命中该行 → turn_id 拿去 grep
    后端日志（ContextFilter 注入每行）。cancelled 不入流水（非错误）。
    """
    from core.db.database import TurnErrorRecord

    q = select(TurnErrorRecord).order_by(TurnErrorRecord.created_at.desc())
    if ref:
        q = q.where(TurnErrorRecord.ref == ref)
    if code:
        q = q.where(TurnErrorRecord.code == code)
    if course_id:
        q = q.where(TurnErrorRecord.course_id == course_id)
    rows = (await db.execute(q.limit(limit))).scalars().all()
    return {"items": [{
        "turn_id": r.turn_id,
        "ref": r.ref,
        "session_id": r.session_id,
        "user_id": r.user_id,
        "course_id": r.course_id,
        "mode": r.mode,
        "code": r.code,
        "message": r.message,
        "exc_type": r.exc_type,
        "exc_head": r.exc_head,
        "created_at": r.created_at,
    } for r in rows]}


# ── 磁盘派生数据 GC（offline，对标 Bazel 磁盘缓存 GC）──────────────────────────

@router.get("/storage/usage")
async def storage_usage_endpoint(
    _: dict = Depends(get_current_admin),
):
    """磁盘用量报告：逐项目录体积 + 整卷水位（只读，供运维观测/告警）。"""
    from core.storage.gc import storage_usage
    return storage_usage()


@router.post("/storage/gc")
async def storage_gc_endpoint(
    dry_run: bool = Query(True),  # 默认 dry-run：只报不删，放开前先核对报告
    _: dict = Depends(get_current_admin),
):
    """触发磁盘 GC。``dry_run=true``（默认）只报不删；放开需显式传 ``dry_run=false``。

    治理 parse_cache / lightrag_store(ingest_chunks) / uploads 三处只写不删的派生数据；
    kb_store/raw 只统计不清理。安全护栏（根路径校验 / mtime 宽限 / 跳过持锁课程）见
    core.storage.gc 模块 docstring。
    """
    from core.storage.gc import run_gc
    report = await run_gc(dry_run=dry_run)
    logger.info(
        "admin 触发 storage GC dry_run=%s freed_gib=%s disk_used_pct=%s",
        dry_run, report.get("total_freed_gib"), report.get("disk_used_pct"),
    )
    return report


@router.post("/context-window/probe")
async def reprobe_context_window(
    _: dict = Depends(get_current_admin),
):
    """手动重探当前 active profile 的模型上下文窗口（切 catalog profile 后调用）。

    强制重探（无视缓存 TTL）text+fast 模型，写回 ``data/context_window_cache.json``，并返回每模型
    的解析来源（``probe`` / ``table`` / ``heuristic`` / ``explicit``）与实际窗口值，让运维确认
    当前用的是哪一级。探测 best-effort：供应商 ``/models`` 未暴露窗口或不可达时 source 退回
    ``table``/``heuristic``（属正常降级，非错误）。

    注：热路径 ``resolve_effective_window`` 用 ``settings.llm.base_url``（进程启动时的 active
    profile）作缓存键。本端点读 catalog 当前 active profile 探测并报告--若启动后热切了 profile，
    探测结果要等下次重启才被热路径采用（见 context_window.resolve_effective_window_with_source）。
    """
    from core.llm.catalog import (
        active_profile_id_cached,
        get_profile_cached,
        profile_fast_model,
        profile_text_model,
    )
    from core.agentic.context_window import resolve_effective_window_with_source
    from core.agentic.window_probe import warmup_probe

    pid = await active_profile_id_cached()
    prof = await get_profile_cached(pid) or {}
    s = get_settings()
    base_url = (prof.get("base_url") or "").strip() or s.llm.base_url
    api_key = (prof.get("api_key") or "").strip() or s.llm.api_key.get_secret_value()
    models = [profile_text_model(prof), profile_fast_model(prof)]

    # force 重探 + 写缓存（best-effort，单模型失败不影响其余）
    await warmup_probe(models, base_url=base_url, api_key=api_key, force=True)

    # 解析每模型当前生效来源/值（用 profile base_url 对齐刚写入的探测缓存键）
    report: list[dict] = []
    seen: set[str] = set()
    for m in models:
        if not m or m in seen:
            continue
        seen.add(m)
        value, source = resolve_effective_window_with_source(m, base_url=base_url)
        report.append({"model": m, "source": source, "value": value})
    logger.info("admin 触发 context-window probe profile=%s models=%d", pid, len(report))
    return {"profile_id": pid, "base_url": base_url, "models": report}
