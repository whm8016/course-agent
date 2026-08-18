"""admin / teacher 共享的知识库文件与索引控制逻辑。

三组曾近乎逐行复制（upload ~45 行 / pause+stop ~70 行 / delete ~25 行，连中文注释
都一字不差）。照 api/kb_indexing.py 的共享先例抽出：权限校验（get_current_admin vs
get_current_teacher + _get_owned_kb）留在各自 router，共享函数只收「已校验的 kb」。

唯一行为差异参数化：pause/stop 下发信号失败时的 503 文案——admin 侧带「（Redis
异常）」标注、teacher 侧不带（历史文案；err_hint 传入，保逐字节兼容）。
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.kb_indexing import get_or_create_build
from core.db.cache import invalidate_courses_cache
from core.db.database import KBFile, KbBuild, KnowledgeBase, aggregate_build_status
from core.rag.ingestion import IndexingControl
from settings import get_settings

logger = logging.getLogger(__name__)

KB_STORE_DIR = get_settings().paths.kb_store_dir
MAX_KB_UPLOAD_MB = get_settings().max_kb_upload_mb

# M-27：去掉 .doc/.ppt——file_routing 仅支持 OOXML 的 .docx/.pptx，legacy 二进制
# 格式（.doc/.ppt）无解析 handler，上传后会在 classify_files 阶段归入 unsupported
# 被静默丢弃（用户以为上传成功，索引时却丢失）。在上传层直接拒绝，给出明确错误。
ALLOWED_EXT = {".pdf", ".txt", ".md", ".docx", ".pptx"}
MAX_BYTES = MAX_KB_UPLOAD_MB * 1024 * 1024


def kb_raw_dir(course_id: str) -> Path:
    return Path(KB_STORE_DIR) / course_id / "raw"


def safe_upload_name(raw_name: str | None) -> str:
    """M-51：清洗知识库上传的客户端文件名，防 path traversal。

    原先 ``f"{uuid}_{file.filename}"`` 直接拼接未清洗的 ``file.filename``，
    若客户端传 ``../evil.pdf`` 或含路径分隔符的名字，会写出 raw_dir 之外（path traversal）。
    这里直接拒 ``..``/``/``/``\\``/控制字符/空名——发现即 400 明确告知，不静默改写
    （与 api/upload.py 的 _safe_basename 同源策略，比 basename 兜底更严格、可审计）。
    """
    name = (raw_name or "").strip()
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or ".." in name
        or any(ord(c) < 32 for c in name)
    ):
        raise HTTPException(
            status_code=400,
            detail="无效的文件名",
        )
    return name


def build_to_dict(b: KbBuild) -> dict:
    """单个后端构建状态（前端按 backend 渲染两张卡）。"""
    return {
        "backend": b.backend,
        "label": "pgvector" if b.backend == "llamaindex_pg" else "LightRAG",
        "status": b.status,
        "progress": b.progress,
        "progress_msg": b.progress_msg,
        "chunks_done": b.chunks_done,
        "chunks_total": b.chunks_total,
        "token_estimate": b.token_estimate,
        "error_msg": b.error_msg,
        "updated_at": b.updated_at,
    }


def primary_build(builds: list[KbBuild], index_backend: str) -> KbBuild | None:
    """顶层（兼容旧字段）取一个代表 build：indexing 优先 → 主后端 → ready → 首个。"""
    for b in builds:
        if b.status == "indexing":
            return b
    primary = index_backend or "lightrag"
    for b in builds:
        if b.backend == primary:
            return b
    for b in builds:
        if b.status == "ready":
            return b
    return builds[0] if builds else None


def kb_to_dict(kb: KnowledgeBase) -> dict:
    # 状态/进度由 kb_builds 聚合（indexing 写本表，KB 行旧列不再被写、仅作历史保留）。
    # builds 经 lazy="selectin" 随 KB 一起取回，这里直接读。顶层字段取一个代表 build，
    # 供旧 UI / 列表徽标用；detail 面板按 builds 数组渲染两后端。
    builds = sorted(kb.builds, key=lambda b: b.backend)
    agg_status = aggregate_build_status(builds)
    p = primary_build(builds, kb.index_backend or "lightrag")
    return {
        "id": kb.id,
        "course_id": kb.course_id,
        "name": kb.name,
        "description": kb.description,
        "icon": kb.icon,
        "system_prompt": kb.system_prompt,
        "sort_order": kb.sort_order,
        "status": agg_status,
        "file_count": kb.file_count,
        "error_msg": p.error_msg if p else "",
        "progress": p.progress if p else 0,
        "progress_msg": p.progress_msg if p else "",
        "chunks_done": p.chunks_done if p else 0,
        "chunks_total": p.chunks_total if p else 0,
        "token_estimate": p.token_estimate if p else 0,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
        "is_visible": bool(kb.is_visible),
        "owner_id": kb.owner_id,
        "join_code": kb.join_code,
        "lightrag_built": bool(kb.file_count > 0),
        "index_backend": kb.index_backend or "lightrag",
        "builds": [build_to_dict(b) for b in builds],
    }


async def save_uploaded_files(
    db: AsyncSession, kb: KnowledgeBase, course_id: str, files: list[UploadFile]
) -> dict:
    """上传文件到知识库（支持 PDF/DOCX/PPTX/TXT/MD）：校验→落盘→KBFile 行→计数回写。

    有新文件时已就绪的后端索引失效置 pending（读取方按 kb_builds 聚合）。
    """
    raw_dir = kb_raw_dir(course_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_names: list[str] = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型 '{ext}'，允许：{', '.join(sorted(ALLOWED_EXT))}",
            )

        content = await file.read()
        if len(content) > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"文件 '{file.filename}' 超过大小限制 {MAX_KB_UPLOAD_MB} MB",
            )

        # M-51：清洗客户端文件名（防 ../ 穿越 / 控制字符），再拼 uuid 前缀落盘
        clean_name = safe_upload_name(file.filename)
        safe_name = f"{uuid.uuid4().hex[:8]}_{clean_name}"
        file_path = raw_dir / safe_name
        file_path.write_bytes(content)

        kb_file = KBFile(
            kb_id=kb.id,
            original_name=clean_name,
            file_path=str(file_path),
            file_size=len(content),
        )
        db.add(kb_file)
        saved_names.append(clean_name)

    await db.flush()

    # 同步更新文件数
    count_result = await db.execute(
        select(func.count()).select_from(KBFile).where(KBFile.kb_id == kb.id)
    )
    kb.file_count = count_result.scalar_one()
    kb.updated_at = time.time()
    # 有新文件：已就绪的后端索引失效，置 pending 等待重建（读取方按 kb_builds 聚合）
    for b in kb.builds:
        if b.status == "ready":
            b.status = "pending"

    logger.info("上传 %d 个文件到知识库 course_id=%s", len(saved_names), course_id)
    return {"uploaded": saved_names, "total_files": kb.file_count}


async def delete_kb_file(db: AsyncSession, kb: KnowledgeBase, file_id: str) -> dict:
    """删除知识库中的单个文件（磁盘 + KBFile 行 + 计数回写）。"""
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


async def pause_index_build(
    db: AsyncSession, kb: KnowledgeBase, course_id: str, backend: str,
    *, err_hint: str = "",
) -> dict:
    """请求暂停正在进行的索引（在下一个 batch 边界生效，进度可续传）。

    控制信号写入 Redis，跨 worker 通知；运行索引的那个 worker 在下一个
    batch 检查点会读到 "pause" 并主动中断。仅 lightrag 后端有 batch 检查点
    （IndexingControl）；backend 参数定位具体 kb_builds 行。
    """
    build = await get_or_create_build(db, kb.id, backend)
    if build.status != "indexing":
        raise HTTPException(status_code=409, detail="当前没有正在进行的索引任务")

    ctrl = IndexingControl(kb.id)
    try:
        await ctrl.request_pause()  # 通知 worker；事件循环被阻塞读不到也无妨
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法下发暂停信号{err_hint}：{e}")

    # 立即落库为 paused：超大文档的 ainsert 会长时间阻塞 worker 事件循环，导致
    # checkpoint 永远读不到信号、前端卡在"请求已发送"。这里直接置 paused 让前端
    # 立即解脱；worker 终态写入有"不覆盖非 indexing 状态"保护，跑完不会回写。
    done = build.chunks_done or 0
    total = build.chunks_total or 0
    build.status = "paused"
    build.progress_msg = f"已暂停（已完成 {done}{f'/{total}' if total else ''} 个文本块）"
    build.updated_at = time.time()
    logger.info("已暂停 course_id=%s backend=%s", course_id, backend)
    await invalidate_courses_cache()
    # 不清 Redis 控制信号：留给 worker 的 checkpoint 读到后自行 cancel 停止，
    # worker 终止后会在 finally 里 control.clear()。这里清了反而让 worker 读不到、继续跑。
    return {"message": "已暂停", "course_id": course_id, "backend": backend}


async def stop_index_build(
    db: AsyncSession, kb: KnowledgeBase, course_id: str, backend: str,
    *, err_hint: str = "",
) -> dict:
    """请求终止索引。

    - indexing：在下一个 batch 边界中止，进度清零，状态置 pending。
    - paused：直接清零进度并置回 pending。
    """
    build = await get_or_create_build(db, kb.id, backend)

    if build.status == "indexing":
        ctrl = IndexingControl(kb.id)
        try:
            await ctrl.request_stop()  # 通知 worker；事件循环被阻塞读不到也无妨
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"无法下发终止信号{err_hint}：{e}")

        # 立即落库为 pending 并清零进度：见 pause 的说明。worker 跑完不会回写。
        build.status = "pending"
        build.progress = 0
        build.progress_msg = "已终止"
        build.chunks_done = 0
        build.chunks_total = 0
        build.token_estimate = 0
        build.error_msg = ""
        build.updated_at = time.time()
        logger.info("已终止 course_id=%s backend=%s", course_id, backend)
        await invalidate_courses_cache()
        # 不清 Redis 控制信号：留给 worker 的 checkpoint 读到后自行 cancel 停止，
        # worker 终止后会在 finally 里 control.clear()。这里清了反而让 worker 读不到、继续跑。
        return {"message": "已终止", "course_id": course_id, "backend": backend}

    if build.status == "paused":
        build.status = "pending"
        build.progress = 0
        build.progress_msg = "已终止"
        build.chunks_done = 0
        build.chunks_total = 0
        build.token_estimate = 0
        build.error_msg = ""
        build.updated_at = time.time()
        logger.info("已清除暂停进度 course_id=%s backend=%s", course_id, backend)
        await invalidate_courses_cache()
        return {"message": "已终止并清除进度", "course_id": course_id, "backend": backend}

    raise HTTPException(status_code=409, detail="当前状态不可终止")
