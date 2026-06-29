from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.auth import get_current_user
from config import UPLOAD_DIR, MAX_UPLOAD_MB

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

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
