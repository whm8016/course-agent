"""Admin API：知识库管理 & 用户管理（仅管理员可访问）。"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_admin
from api.courses import invalidate_courses_cache
from api.kb_indexing import trigger_kb_indexing
from settings import get_settings
FAQ_CACHE_THRESHOLD = get_settings().question.faq_cache_threshold
KB_STORE_DIR = get_settings().paths.kb_store_dir
MAX_KB_UPLOAD_MB = get_settings().max_kb_upload_mb
LLAMA_INDEX_KB_ROOT = get_settings().paths.llama_index_kb_root
from core.rag import is_lightrag_available
from core.llm.prompts import invalidate_course_prompt_cache
from core.codes import ensure_unique_join_code, generate_code
from core.db.database import (
    AsyncSessionLocal,
    ApplicationStatus,
    BotNotification,
    KBFile,
    KnowledgeBase,
    TeacherApplication,
    TeacherInvite,
    User,
    get_db,
)
from core.db.cache import faq_top
from core.rag.ingestion import (
    IndexingAborted,
    IndexingControl,
    ingest_to_lightrag,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_ALLOWED_EXT = {".pdf", ".txt", ".md", ".docx", ".doc", ".pptx", ".ppt"}
_MAX_BYTES = MAX_KB_UPLOAD_MB * 1024 * 1024

# 注：暂停/终止控制信号通过 Redis 跨 worker 传递，
# 不再依赖进程内字典。详见 core.ingestion.IndexingControl。


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _kb_raw_dir(course_id: str) -> Path:
    return Path(KB_STORE_DIR) / course_id / "raw"


def _kb_to_dict(kb: KnowledgeBase) -> dict:
    return {
        "id": kb.id,
        "course_id": kb.course_id,
        "name": kb.name,
        "description": kb.description,
        "icon": kb.icon,
        "system_prompt": kb.system_prompt,
        "sort_order": kb.sort_order,
        "status": kb.status,
        "file_count": kb.file_count,
        "error_msg": kb.error_msg,
        "progress": kb.progress,
        "progress_msg": kb.progress_msg,
        "chunks_done": kb.chunks_done,
        "chunks_total": kb.chunks_total,
        "token_estimate": kb.token_estimate,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
        "is_visible": bool(kb.is_visible),
        "owner_id": kb.owner_id,
        "join_code": kb.join_code,
        "lightrag_built": bool(kb.file_count > 0),
        # LlamaIndex 是否已建：以索引产物 docstore.json 是否落盘为准（无独立 DB 列）。
        # 前端据此显示"LlamaIndex 索引已完成"绿勾、切换"首次构建/重新构建"按钮。
        "llamaindex_built": (
            Path(LLAMA_INDEX_KB_ROOT) / kb.course_id / "llamaindex_storage" / "docstore.json"
        ).exists(),
    }


async def _get_kb_or_404(db: AsyncSession, course_id: str) -> KnowledgeBase:
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.course_id == course_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail=f"知识库 '{course_id}' 不存在")
    return kb


# ── 后台索引任务 ──────────────────────────────────────────────────────────────

async def _run_indexing(
    kb_id: str,
    course_id: str,
    file_paths: list[str],
    resume_from_chunk: int = 0,
) -> None:
    """后台任务：LlamaIndex 解析 → LightRAG 摄入（附带进度回调，支持断点续传）。"""
    # 1. 重置/保留进度，更新状态为 indexing
    async with AsyncSessionLocal() as db:
        async with db.begin():
            result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            )
            kb = result.scalar_one_or_none()
            if not kb:
                return
            kb.status = "indexing"
            kb.error_msg = ""
            if resume_from_chunk == 0:
                kb.progress = 0
                kb.progress_msg = "准备中…"
                kb.chunks_done = 0
                kb.chunks_total = 0
                kb.token_estimate = 0
            else:
                kb.progress_msg = f"续传中（从第 {resume_from_chunk} 个文本块继续）…"
            kb.updated_at = time.time()

    # 状态从 pending/error/paused → indexing，让前端的「就绪/未就绪」徽章及时变更
    await invalidate_courses_cache()

    # 进度回调
    async def _on_progress(
        progress: int,
        msg: str,
        chunks_done: int,
        chunks_total: int,
        token_estimate: int,
    ) -> None:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
                )
                kb = result.scalar_one_or_none()
                if kb:
                    kb.progress = progress
                    kb.progress_msg = msg
                    kb.chunks_done = chunks_done
                    kb.chunks_total = chunks_total
                    kb.token_estimate = token_estimate
                    kb.updated_at = time.time()

    # 控制信号走 Redis，跨 worker 共享；先清掉上次残留
    control = IndexingControl(kb_id)
    await control.clear()

    # 全新索引（非续传）：清空 LightRAG 旧数据，杜绝 ainsert 整批判重
    # （"Duplicate document" → failed entries）。续传绝不能清，会抹掉进度。
    if resume_from_chunk == 0:
        from core.rag.lightrag import purge_course_workspace
        await purge_course_workspace(course_id)

    abort_action: str | None = None
    abort_chunks_done = 0
    # 预置"未正常结束"兜底：任何未被显式覆盖的退出路径都至少写成 error，绝不留在 indexing。
    final_status: str = "error"
    final_err: str = "索引任务未正常结束"
    try:
        summary = await ingest_to_lightrag(
            course_id,
            file_paths,
            on_progress=_on_progress,
            resume_from_chunk=resume_from_chunk,
            control=control,
        )
        logger.info("索引完成 kb_id=%s summary=%s", kb_id, summary)
        final_status = "ready"
        final_err = ""
    except IndexingAborted as e:
        abort_action = e.action
        abort_chunks_done = e.chunks_done
        if e.action == "pause":
            final_status = "paused"
            logger.info("索引已暂停 kb_id=%s chunks_done=%d", kb_id, e.chunks_done)
        else:
            final_status = "pending"
            logger.info("索引已终止 kb_id=%s", kb_id)
        final_err = ""
    except asyncio.CancelledError:
        # CancelledError 是 BaseException 子类，上面 except Exception 捕获不到。
        # ARQ worker 超时/OOM/重启会取消任务，若不在本分支兜底，异常直接冒泡 →
        # 下方终态回写不执行 → status 永久卡 indexing（前端一直"构建中"且无救）。
        logger.warning("索引任务被取消 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = "索引任务被中断（worker 超时/OOM/重启），可重试"
        # 捕获后取消计数仍在，后续 await（control.clear / 终态回写）会立即再抛
        # CancelledError。uncancel 清除计数，保证下方清理与回写能真正跑完。
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
    except Exception as e:
        logger.exception("索引失败 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = str(e)[:500]
    finally:
        # 不论结果如何，清掉信号，避免下次启动立即被旧信号中断
        await control.clear()

    # 2. 更新最终状态 —— 用 shield 保护回写协程，避免任务取消时 await 被二次取消；
    #    外层兜底 CancelledError/Exception，确保 _run_indexing 不会因回写失败而抛出，
    #    从而保证 indexing 一定会被改写成终态（ready/error/paused/pending），杜绝卡死。
    async def _apply_final() -> None:
        async with AsyncSessionLocal() as db:
            async with db.begin():
                result = await db.execute(
                    select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
                )
                kb = result.scalar_one_or_none()
                if not kb:
                    return
                # 期间若被 pause/stop 接口强制改态（≠indexing），说明用户已主动干预，
                # 不覆盖——否则被强制终止的任务跑完会回写 ready/paused，与用户意图冲突。
                if kb.status != "indexing":
                    logger.info(
                        "索引终态被外部干预 kb=%s 当前=%s，不覆盖为 %s",
                        kb_id, kb.status, final_status,
                    )
                    return
                kb.status = final_status
                kb.error_msg = final_err
                kb.updated_at = time.time()
                if abort_action == "pause":
                    kb.chunks_done = abort_chunks_done
                    kb.progress_msg = (
                        f"已暂停（已完成 {abort_chunks_done}"
                        f"{f'/{kb.chunks_total}' if kb.chunks_total else ''} 个文本块）"
                    )
                elif abort_action == "stop":
                    kb.progress = 0
                    kb.progress_msg = "已终止"
                    kb.chunks_done = 0
                    kb.chunks_total = 0
                    kb.token_estimate = 0

    try:
        await asyncio.shield(_apply_final())
    except asyncio.CancelledError:
        logger.warning("终态回写外层收到取消信号 kb_id=%s（shield 内已尽力完成）", kb_id)
    except Exception:
        logger.exception("索引终态回写失败 kb_id=%s course=%s", kb_id, course_id)

    # 索引结束（ready / error / paused / pending），重要：ready 时前端要切到 LightRAG 路径
    await invalidate_courses_cache()


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
    )
    db.add(kb)
    await db.flush()
    # 与教师建课一致：建库即自动生成课程码（管理员库历史漏生成，前端凭此渲染二维码）
    kb.join_code = await ensure_unique_join_code(db)
    await db.flush()

    _kb_raw_dir(body.course_id).mkdir(parents=True, exist_ok=True)
    logger.info("创建知识库 course_id=%s", body.course_id)
    await invalidate_courses_cache()
    return _kb_to_dict(kb)


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
    """删除知识库（含磁盘文件）。"""
    kb = await _get_kb_or_404(db, course_id)
    await db.delete(kb)

    kb_dir = Path(KB_STORE_DIR) / course_id
    if kb_dir.exists():
        shutil.rmtree(kb_dir, ignore_errors=True)

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
    """上传文件到知识库（支持 PDF/DOCX/PPTX/TXT/MD）。"""
    kb = await _get_kb_or_404(db, course_id)

    raw_dir = _kb_raw_dir(course_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_names: list[str] = []
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
                detail=f"文件 '{file.filename}' 超过大小限制 {MAX_KB_UPLOAD_MB} MB",
            )

        safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        file_path = raw_dir / safe_name
        file_path.write_bytes(content)

        kb_file = KBFile(
            kb_id=kb.id,
            original_name=file.filename or safe_name,
            file_path=str(file_path),
            file_size=len(content),
        )
        db.add(kb_file)
        saved_names.append(file.filename or safe_name)

    await db.flush()

    # 同步更新文件数
    count_result = await db.execute(
        select(func.count()).select_from(KBFile).where(KBFile.kb_id == kb.id)
    )
    kb.file_count = count_result.scalar_one()
    kb.updated_at = time.time()
    if kb.status == "ready":
        kb.status = "pending"  # 有新文件，需要重新索引

    logger.info("上传 %d 个文件到知识库 course_id=%s", len(saved_names), course_id)
    return {"uploaded": saved_names, "total_files": kb.file_count}


@router.delete("/kb/{course_id}/files/{file_id}")
async def delete_file(
    course_id: str,
    file_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """删除知识库中的单个文件。"""
    kb = await _get_kb_or_404(db, course_id)

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


# ── 触发索引 ──────────────────────────────────────────────────────────────────

@router.post("/kb/{course_id}/index")
async def index_kb(
    course_id: str,
    force: bool = False,
    resume: bool = False,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """触发知识库索引（后台任务：LlamaIndex 解析 → LightRAG 摄入）。

    - force=true：强制重新索引（即使正在进行中）
    - resume=true：从上次中断位置续传（仅限 error 状态）

    公共逻辑（状态校验 / DB 预置 / ARQ 入队）见 api.kb_indexing.trigger_kb_indexing。
    """
    kb = await _get_kb_or_404(db, course_id)
    return await trigger_kb_indexing(db, kb, course_id, force, resume)


@router.post("/kb/{course_id}/index/pause")
async def pause_index(
    course_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """请求暂停正在进行的索引（在下一个 batch 边界生效，进度可续传）。

    控制信号写入 Redis，跨 worker 通知；运行索引的那个 worker 在下一个
    batch 检查点会读到 "pause" 并主动中断。
    """
    kb = await _get_kb_or_404(db, course_id)
    if kb.status != "indexing":
        raise HTTPException(status_code=409, detail="当前没有正在进行的索引任务")

    ctrl = IndexingControl(kb.id)
    try:
        await ctrl.request_pause()  # 通知 worker；事件循环被阻塞读不到也无妨
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法下发暂停信号（Redis 异常）：{e}")

    # 立即落库为 paused：超大文档的 ainsert 会长时间阻塞 worker 事件循环，导致
    # checkpoint 永远读不到信号、前端卡在"请求已发送"。这里直接置 paused 让前端
    # 立即解脱；worker 终态写入有"不覆盖非 indexing 状态"保护，跑完不会回写。
    done = kb.chunks_done or 0
    total = kb.chunks_total or 0
    kb.status = "paused"
    kb.progress_msg = f"已暂停（已完成 {done}{f'/{total}' if total else ''} 个文本块）"
    kb.updated_at = time.time()
    logger.info("已暂停 course_id=%s", course_id)
    await invalidate_courses_cache()
    # 不清 Redis 控制信号：留给 worker 的 checkpoint 读到后自行 cancel 停止，
    # worker 终止后会在 finally 里 control.clear()。这里清了反而让 worker 读不到、继续跑。
    return {"message": "已暂停", "course_id": course_id}


@router.post("/kb/{course_id}/index/stop")
async def stop_index(
    course_id: str,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """请求终止索引。

    - indexing：在下一个 batch 边界中止，进度清零，状态置 pending。
    - paused：直接清零进度并置回 pending。
    """
    kb = await _get_kb_or_404(db, course_id)

    if kb.status == "indexing":
        ctrl = IndexingControl(kb.id)
        try:
            await ctrl.request_stop()  # 通知 worker；事件循环被阻塞读不到也无妨
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"无法下发终止信号（Redis 异常）：{e}")

        # 立即落库为 pending 并清零进度：见 pause 的说明。worker 跑完不会回写。
        kb.status = "pending"
        kb.progress = 0
        kb.progress_msg = "已终止"
        kb.chunks_done = 0
        kb.chunks_total = 0
        kb.token_estimate = 0
        kb.error_msg = ""
        kb.updated_at = time.time()
        logger.info("已终止 course_id=%s", course_id)
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
        logger.info("已清除暂停进度 course_id=%s", course_id)
        await invalidate_courses_cache()
        return {"message": "已终止并清除进度", "course_id": course_id}

    raise HTTPException(status_code=409, detail="当前状态不可终止")


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
    """
    if top_n < 1 or top_n > 100:
        top_n = 20

    if course_id:
        items = await faq_top(course_id, top_n)
        return {"course_id": course_id, "threshold": FAQ_CACHE_THRESHOLD, "questions": items}

    # 遍历所有课程
    kb_result = await db.execute(select(KnowledgeBase.course_id, KnowledgeBase.name))
    courses = kb_result.all()
    all_items: list[dict] = []
    for cid, cname in courses:
        items = await faq_top(cid, top_n)
        for item in items:
            all_items.append({"course_id": cid, "course_name": cname, **item})
    # 按 count 降序排列后取 top_n
    all_items.sort(key=lambda x: x["count"], reverse=True)
    return {"course_id": None, "threshold": FAQ_CACHE_THRESHOLD, "questions": all_items[:top_n]}
