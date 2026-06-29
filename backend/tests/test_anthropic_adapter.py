"""core/llm/providers/anthropic_adapter.py 图片格式转换单元测试。

验证 OpenAI image_url data URL → Anthropic image source base64 的转换
（此前直接原样透传会导致 Claude API 400）。
"""
from __future__ import annotations

from core.llm.providers.anthropic_adapter import _convert_content_blocks_to_anthropic


def test_image_url_data_url_converted_to_image_source():
    """OpenAI image_url data URL → Anthropic image source base64。"""
    blocks = [
        {"type": "text", "text": "这张图是什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    out = _convert_content_blocks_to_anthropic(blocks)
    assert out[0] == {"type": "text", "text": "这张图是什么？"}
    assert out[1]["type"] == "image"
    assert out[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "AAAA"}


def test_jpeg_mime_extracted():
    """jpeg data URL 的 media_type 正确提取。"""
    blocks = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}}]
    out = _convert_content_blocks_to_anthropic(blocks)
    assert out[0]["source"]["media_type"] == "image/jpeg"


def test_plain_text_unchanged():
    """纯文本 content block 不被破坏。"""
    blocks = [{"type": "text", "text": "你好"}]
    out = _convert_content_blocks_to_anthropic(blocks)
    assert out == [{"type": "text", "text": "你好"}]


def test_http_url_image_skipped():
    """http URL 远程图（非 base64）本期跳过（不注入，不报错）。"""
    blocks = [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]
    out = _convert_content_blocks_to_anthropic(blocks)
    assert out == []  # 跳过 → 空
