"""附件模型 —— 多模态对话的统一附件抽象。

纯数据载体（Pydantic v2），无业务行为。type 是唯一分类开关：
- IMAGE 本期实现（视觉看图）
- FILE / PDF 预留给 P2 的文件解析（DOCX/TXT/PDF，属于深度解析，本期不做）

设计要点：url 与 base64 双存；持久化前 base64 必须清空。
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class AttachmentType(str, Enum):
    IMAGE = "image"
    FILE = "file"   # P2: DOCX/TXT 纯文本提取
    PDF = "pdf"     # P2: PDF / OCR 深度解析


class Attachment(BaseModel):
    """一次用户回合的附件（图片或文件）。

    Attributes:
        type: 分类开关，唯一决定附件如何被处理。
        url: 可访问 URL（/api/uploads/xxx 或 http(s)://）。
        file_path: 服务端磁盘绝对路径；注入图片时优先据此读 base64。
        base64: 已编码 base64（不含 data: 前缀）；持久化前必须清空。
        mime_type: image/jpeg | image/png | ...
        filename: 原始文件名。
        id: 稳定标识（P2 AttachmentStore 用）；MVP 可空。
    """

    type: AttachmentType = AttachmentType.IMAGE
    url: str | None = None
    file_path: str | None = None
    base64: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    id: str | None = None

    def is_image(self) -> bool:
        return self.type == AttachmentType.IMAGE


def from_image_path(path: str) -> Attachment:
    """把旧的单图路径（image_path: str）包成单个 IMAGE 附件，向后兼容。"""
    return Attachment(type=AttachmentType.IMAGE, file_path=path)
