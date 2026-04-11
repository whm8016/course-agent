from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from api.auth import get_current_user
from config import UPLOAD_DIR, MAX_UPLOAD_MB

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _safe_ext(filename: str | None) -> str:
    if not filename:
        return ".png"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return ".png"
    return ext


@router.post("/upload")
async def upload_image(file: UploadFile, user: dict = Depends(get_current_user)):
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}，仅允许图片")

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
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(content)

    logger.info("Upload by user=%s file=%s size=%d", user["id"], filename, len(content))
    return {"filename": filename, "path": filepath}
