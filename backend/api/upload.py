from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.auth import get_current_user
from core.attachment import Attachment, AttachmentType
from settings import get_settings
UPLOAD_DIR = get_settings().paths.upload_dir
MAX_UPLOAD_MB = get_settings().max_upload_mb

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

MAX_IMAGES_PER_TURN = 4  # 单轮最多图片附件数（防多图 base64 涨 token 致超时；chat/run 共用）

ALLOWED_MIME_TYPES = {
    # 图片
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    # 文档
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
                      ".pdf", ".txt", ".md", ".docx", ".doc"}


def _safe_ext(filename: str | None) -> str:
    if not filename:
        return ".png"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return ".png"
    return ext


def _safe_basename(filename: str) -> str:
    base = os.path.basename(filename)
    if not base or ".." in base or "/" in base or "\\" in base:
        raise HTTPException(status_code=400, detail="无效的文件名")
    return base


def _upload_owner(filename: str) -> str | None:
    """Filename format: {user_id}_{uuid}.ext"""
    base = _safe_basename(filename)
    if "_" not in base:
        return None
    return base.split("_", 1)[0]


def assert_upload_owner(filename: str, user: dict) -> None:
    """校验上传文件归属：仅 owner 或 admin 可访问，否则 403。

    供 chat 等需要引用已上传图片的入口做多租户归属校验，避免越权读取他人图片。
    """
    safe_name = _safe_basename(filename)
    owner = _upload_owner(safe_name)
    if owner is None or (owner != str(user["id"]) and not user.get("is_admin")):
        raise HTTPException(status_code=403, detail="无权访问此文件")


def resolve_upload_path(path_or_url: str | None) -> str | None:
    """Map client path (/api/uploads/... or /uploads/...) to server filesystem path."""
    if not path_or_url:
        return None
    p = path_or_url.strip()
    for prefix in ("/api/uploads/", "/uploads/"):
        if p.startswith(prefix):
            name = _safe_basename(p[len(prefix) :])
            return os.path.join(UPLOAD_DIR, name)
    if os.path.isabs(p) and os.path.isfile(p):
        return p
    candidate = os.path.join(UPLOAD_DIR, os.path.basename(p))
    if os.path.isfile(candidate):
        return candidate
    return p


def resolve_attachments(
    payload_attachments: list | None, image_path: str | None
) -> list[Attachment]:
    """统一附件解析：attachments 列表优先，回退旧 image_path 单图（作为 url）。

    chat / run 共用。回退统一填 ``url``（/api/uploads/xxx），由 materialize_attachments
    做归属校验 + 解析磁盘路径 + 读 base64——消除此前 chat 填 url、run 填 file_path
    （客户端 URL 形式的路径塞 file_path 不被 materialize 处理、无归属校验）的分歧与
    run 侧既有缺陷。from_image_path 仅适用于核心层真实磁盘绝对路径（llm/multimodal 兜底读盘）。
    """
    attachments: list[Attachment] = []
    for raw in (payload_attachments or []):
        try:
            attachments.append(
                raw if isinstance(raw, Attachment) else Attachment.model_validate(raw)
            )
        except Exception:
            logger.warning("resolve_attachments: 无效 attachment 已忽略: %r", raw)
    ip = (image_path or "").strip()
    if ip and not any(a.is_image() for a in attachments):
        attachments.append(Attachment(type=AttachmentType.IMAGE, url=ip))
    return attachments


def enforce_image_limit(attachments: list[Attachment]) -> None:
    """单轮图片数上限（防多图 base64 涨 token 致超时），超限抛 400。供 HTTP 入口（chat）用。"""
    image_count = sum(1 for a in attachments if a.is_image())
    if image_count > MAX_IMAGES_PER_TURN:
        raise HTTPException(
            status_code=400, detail=f"单次最多上传 {MAX_IMAGES_PER_TURN} 张图片"
        )


def materialize_attachments(attachments: list, user: dict) -> None:
    """物化本地 ``/api/uploads/`` 附件：归属校验 + 解析磁盘路径 + 读 base64。

    chat / run 等入口共用。外部 http(s) URL 不下载（安全）；仅本地 ``/api/uploads/``、
    ``/uploads/`` 前缀做归属校验并读盘填 base64（图片注入与文档文本提取的共同前提，
    下游 loop/vision 不再重复读盘）。原地修改 attachments。
    """
    import base64
    from pathlib import Path

    from core.llm.multimodal import _guess_mime_type

    for att in attachments:
        url = att.url or ""
        if not any(url.startswith(p) for p in ("/api/uploads/", "/uploads/")):
            continue
        assert_upload_owner(url.rsplit("/", 1)[-1], user)
        att.file_path = resolve_upload_path(url) or att.file_path
        if att.file_path and os.path.isfile(att.file_path):
            att.base64 = base64.b64encode(Path(att.file_path).read_bytes()).decode("ascii")
            if att.is_image():
                if not att.mime_type:
                    att.mime_type = _guess_mime_type(att.filename or att.file_path)
            elif not att.filename:
                att.filename = os.path.basename(att.file_path)


@router.post("/upload")
async def upload_image(file: UploadFile, user: dict = Depends(get_current_user)):
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}，支持图片与 PDF/Word/文本")

    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（最大 {MAX_UPLOAD_BYTES // (1024*1024)} MB）",
        )

    ext = _safe_ext(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不允许的文件扩展名")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{user['id']}_{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(content)

    logger.info("Upload by user=%s file=%s size=%d", user["id"], filename, len(content))
    return {"filename": filename, "path": f"/api/uploads/{filename}"}


@router.get("/uploads/{filename}")
async def get_upload(filename: str, user: dict = Depends(get_current_user)):
    assert_upload_owner(filename, user)
    safe_name = _safe_basename(filename)

    filepath = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(filepath)
