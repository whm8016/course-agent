"""core/llm/multimodal.py 单元测试 —— prepare_multimodal_messages 纯函数。

覆盖 OpenAI / Anthropic 双分支图片注入（三层解耦第二层）。
"""
from __future__ import annotations

from pathlib import Path

from core.attachment import Attachment, AttachmentType, from_image_path
from core.llm.multimodal import prepare_multimodal_messages


def _make_image(tmp_path: Path, name: str = "a.png") -> str:
    """造一个最小伪 PNG 文件（内容任意，_image_to_data_url 只读字节做 base64），返回路径。"""
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")
    return str(p)


def _user_messages(text: str = "这张图是什么？") -> list[dict]:
    return [
        {"role": "system", "content": "你是助教"},
        {"role": "user", "content": text},
    ]


def test_prepare_no_attachments_returns_unchanged():
    """无附件 → 原样返回（零注入）。"""
    msgs = _user_messages()
    out = prepare_multimodal_messages(msgs, None, "dashscope")
    assert out is msgs
    assert out[-1]["content"] == "这张图是什么？"


def test_prepare_no_image_attachments_returns_unchanged():
    """附件存在但无图片 → 原样返回。"""
    msgs = _user_messages()
    atts = [Attachment(type=AttachmentType.FILE, url="/api/uploads/x.txt")]
    out = prepare_multimodal_messages(msgs, atts, "dashscope")
    assert out[-1]["content"] == "这张图是什么？"


def test_prepare_openai_image(tmp_path):
    """dashscope（openai_compat）binding → image_url content part。"""
    img = _make_image(tmp_path)
    msgs = _user_messages("图里写了什么？")
    atts = [from_image_path(img)]
    out = prepare_multimodal_messages(msgs, atts, "dashscope")

    content = out[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "图里写了什么？"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_prepare_anthropic_image(tmp_path):
    """anthropic binding → image source base64 block（非 image_url，修 Claude 400）。"""
    img = _make_image(tmp_path)
    msgs = _user_messages("图里写了什么？")
    atts = [from_image_path(img)]
    out = prepare_multimodal_messages(msgs, atts, "anthropic")

    content = out[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "图里写了什么？"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"]  # 非空 base64


def test_prepare_multi_images(tmp_path):
    """多图 → 多个 image part（多图能力）。"""
    img1 = _make_image(tmp_path, "a.png")
    img2 = _make_image(tmp_path, "b.png")
    msgs = _user_messages("对比这两张图")
    atts = [from_image_path(img1), from_image_path(img2)]
    out = prepare_multimodal_messages(msgs, atts, "dashscope")

    content = out[-1]["content"]
    assert isinstance(content, list)
    image_parts = [c for c in content if c.get("type") == "image_url"]
    assert len(image_parts) == 2


def test_prepare_empty_text_uses_fallback(tmp_path):
    """user content 为空 → 用 fallback_text 作为 text part。"""
    img = _make_image(tmp_path)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": ""}]
    atts = [from_image_path(img)]
    out = prepare_multimodal_messages(msgs, atts, "dashscope", fallback_text="请描述这张图片")

    content = out[-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "请描述这张图片"}


def test_is_image_input_unsupported_keywords():
    """Stage-2 判据：命中图片相关措辞为 True，无关错误为 False（对标 ）。"""
    from core.llm.multimodal import is_image_input_unsupported

    assert is_image_input_unsupported(RuntimeError("model does not support image input"))
    assert is_image_input_unsupported(RuntimeError("messages.1.content must be a string"))
    assert is_image_input_unsupported(RuntimeError("vision modality not supported"))
    assert is_image_input_unsupported(RuntimeError("invalid type for 'messages'"))
    # 非图片错误不命中（避免误降级）
    assert not is_image_input_unsupported(RuntimeError("rate limit exceeded"))
    assert not is_image_input_unsupported(RuntimeError("internal server error"))
