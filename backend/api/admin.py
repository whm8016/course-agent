"""Admin API：知识库管理 & 用户管理（仅管理员可访问）。"""
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

from api.auth import get_current_admin
from api.courses import invalidate_courses_cache
from config import FAQ_CACHE_THRESHOLD, KB_STORE_DIR, MAX_KB_UPLOAD_MB
from core.prompts import invalidate_course_prompt_cache
from core.database import AsyncSessionLocal, KBFile, KnowledgeBase, User, get_db
from core.cache import faq_top
from core.ingestion import (
    IndexingAborted,
    IndexingControl,
    ingest_to_lightrag,
    llama_available,
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

    abort_action: str | None = None
    abort_chunks_done = 0
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
    except Exception as e:
        logger.exception("索引失败 kb_id=%s course=%s", kb_id, course_id)
        final_status = "error"
        final_err = str(e)[:500]
    finally:
        # 不论结果如何，清掉信号，避免下次启动立即被旧信号中断
        await control.clear()

    # 2. 更新最终状态
    async with AsyncSessionLocal() as db:
        async with db.begin():
            result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
            )
            kb = result.scalar_one_or_none()
            if kb:
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

    # 索引结束（ready / error / paused / pending），重要：ready 时前端要切到 LightRAG 路径
    await invalidate_courses_cache()


# ── 系统信息 ──────────────────────────────────────────────────────────────────

@router.get("/info")
async def admin_info(_: dict = Depends(get_current_admin)):
    """返回管理后台基本信息。"""
    from core.lightrag_engine import is_lightrag_available
    rag_ok, rag_reason = is_lightrag_available()
    return {
        "llama_index_available": llama_available(),
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
    background_tasks: BackgroundTasks,
    force: bool = False,
    resume: bool = False,
    _: dict = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    """触发知识库索引（后台任务：LlamaIndex 解析 → LightRAG 摄入）。

    - force=true：强制重新索引（即使正在进行中）
    - resume=true：从上次中断位置续传（仅限 error 状态）
    """
    kb = await _get_kb_or_404(db, course_id)

    if kb.status == "indexing" and not force:
        raise HTTPException(status_code=409, detail="正在索引中，请等待完成后再试")

    files_result = await db.execute(
        select(KBFile).where(KBFile.kb_id == kb.id)
    )
    files = files_result.scalars().all()
    if not files:
        raise HTTPException(status_code=400, detail="知识库中没有文件，请先上传文件")

    file_paths = [f.file_path for f in files if Path(f.file_path).exists()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="文件在磁盘上不存在，请重新上传")

    # 断点续传：在 error / paused 状态且有进度记录时生效
    resume_from = 0
    if resume and kb.status in ("error", "paused") and kb.chunks_done > 0:
        resume_from = kb.chunks_done
        logger.info("断点续传 course_id=%s 从 chunk %d 继续", course_id, resume_from)

    background_tasks.add_task(_run_indexing, kb.id, course_id, file_paths, resume_from)
    logger.info("启动索引任务 course_id=%s files=%d resume_from=%d", course_id, len(file_paths), resume_from)

    return {
        "message": "索引任务已启动" if resume_from == 0 else f"续传任务已启动（从第 {resume_from} 个文本块）",
        "course_id": course_id,
        "file_count": len(file_paths),
        "resume_from_chunk": resume_from,
    }


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
        await ctrl.request_pause()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法下发暂停信号（Redis 异常）：{e}")

    kb.progress_msg = "暂停请求已发送，等待当前批次完成…"
    kb.updated_at = time.time()
    logger.info("收到暂停请求 course_id=%s", course_id)
    return {"message": "暂停请求已发送", "course_id": course_id}


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
            await ctrl.request_stop()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"无法下发终止信号（Redis 异常）：{e}")

        kb.progress_msg = "终止请求已发送，等待当前批次完成…"
        kb.updated_at = time.time()
        logger.info("收到终止请求 course_id=%s", course_id)
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
            "is_admin": bool(u.is_admin),
            "created_at": u.created_at,
        }
        for u in users
    ]


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
