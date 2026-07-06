"""步骤8：api.upload 附件解析 + 图片限流统一入口测试。

验证：
- resolve_attachments：attachments 列表优先；回退 image_path 填 url（非 file_path，
  走 materialize 归属校验，修 run.py 旧缺陷）；列表已有图不回退；无效 raw 容错忽略；
  接受已校验的 Attachment 实例（chat 入口）不重复校验
- enforce_image_limit：超 MAX_IMAGES_PER_TURN 抛 400；恰等于上限不抛；非图附件不计入
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.upload import (
    MAX_IMAGES_PER_TURN,
    enforce_image_limit,
    resolve_attachments,
)
from core.attachment import Attachment, AttachmentType


def test_resolve_list_takes_priority():
    atts = resolve_attachments(
        [Attachment(type=AttachmentType.IMAGE, url="/api/uploads/a.png")],
        "/api/uploads/legacy.png",
    )
    assert len(atts) == 1  # 列表已有图，不回退 image_path
    assert atts[0].url == "/api/uploads/a.png"


def test_resolve_falls_back_to_image_path_as_url():
    atts = resolve_attachments(None, "/api/uploads/legacy.png")
    assert len(atts) == 1
    assert atts[0].is_image()
    # 关键：回退填 url（走 materialize 归属校验），不是 file_path（run.py 旧缺陷）
    assert atts[0].url == "/api/uploads/legacy.png"
    assert atts[0].file_path is None


def test_resolve_falls_back_only_when_no_image_in_list():
    # 列表只有非图附件 → 仍回退 image_path
    atts = resolve_attachments(
        [Attachment(type=AttachmentType.FILE, url="/api/uploads/doc.pdf")],
        "/api/uploads/img.png",
    )
    assert len(atts) == 2
    assert atts[1].is_image() and atts[1].url == "/api/uploads/img.png"


def test_resolve_invalid_raw_ignored():
    # 非法 type 触发枚举校验失败 → 被容错忽略
    atts = resolve_attachments(
        [{"type": "image", "url": "ok"}, {"type": "not_a_valid_type"}], None
    )
    assert len(atts) == 1
    assert atts[0].url == "ok"


def test_resolve_accepts_attachment_instances():
    """chat 传入已校验的 Attachment 实例列表，直接复用不重复校验。"""
    inst = Attachment(type=AttachmentType.IMAGE, url="/api/uploads/x.png")
    atts = resolve_attachments([inst], None)
    assert atts[0] is inst


def test_enforce_limit_raises_over_cap():
    too_many = [
        Attachment(type=AttachmentType.IMAGE) for _ in range(MAX_IMAGES_PER_TURN + 1)
    ]
    with pytest.raises(HTTPException) as exc:
        enforce_image_limit(too_many)
    assert exc.value.status_code == 400


def test_enforce_limit_ok_at_cap():
    at_cap = [Attachment(type=AttachmentType.IMAGE) for _ in range(MAX_IMAGES_PER_TURN)]
    enforce_image_limit(at_cap)  # 恰等于上限不抛


def test_enforce_limit_ignores_non_image():
    mix = [Attachment(type=AttachmentType.IMAGE)] * MAX_IMAGES_PER_TURN + [
        Attachment(type=AttachmentType.FILE)
    ]
    enforce_image_limit(mix)  # 非图不计入，不抛
