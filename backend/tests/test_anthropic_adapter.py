"""core/llm/providers/anthropic_adapter.py 图片格式转换单元测试。

验证 OpenAI image_url data URL → Anthropic image source base64 的转换
（此前直接原样透传会导致 Claude API 400）。
"""
from __future__ import annotations

import pytest

from core.llm.providers.anthropic_adapter import (
    _convert_content_blocks_to_anthropic,
    _map_stop_reason,
)


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


def test_stop_reason_mapped_to_openai_finish():
    """Anthropic stop_reason → OpenAI finish_reason 词表归一（截断检测只认 OpenAI 方言）。"""
    assert _map_stop_reason("max_tokens") == "length"   # 撞输出上限（Anthropic 方言）
    assert _map_stop_reason("end_turn") == "stop"       # 正常结束
    assert _map_stop_reason("stop_sequence") == "stop"  # 命中停止序列
    assert _map_stop_reason("pause_turn") == "pause_turn"  # 未知值原样透传


# ── 流式事件链：stop_reason 从 message_delta 捕获（message_stop 事件本身无此字段）──

def _evt(etype: str, **kw):
    from types import SimpleNamespace
    return SimpleNamespace(type=etype, **kw)


@pytest.mark.asyncio
async def test_stream_finish_reason_taken_from_message_delta(anthropic_fake_client_cls):
    """SDK 事实：RawMessageStopEvent 只有 type 字段，真实 stop_reason 住在最后一个
    message_delta 的 delta.stop_reason——适配器须从那里捕获，message_stop 消费并翻译。
    （回归锁：首版实现误从 message_stop.message 取，恒得默认 'stop'，映射成死路径。）"""
    from types import SimpleNamespace
    from core.llm.providers.anthropic_adapter import _AnthropicStream

    events = [
        _evt("content_block_delta",
             delta=SimpleNamespace(type="text_delta", text="答案文本")),
        _evt("message_delta",
             delta=SimpleNamespace(stop_reason="max_tokens", stop_sequence=None),
             usage=SimpleNamespace(output_tokens=12)),
        _evt("message_stop"),  # 无 message/stop_reason 字段
    ]
    stream = _AnthropicStream(client=anthropic_fake_client_cls(events), create_kwargs={})

    chunks = [c async for c in stream]
    finishes = [
        ch.choices[0].finish_reason
        for ch in chunks
        if getattr(ch, "choices", None) and getattr(ch.choices[0], "finish_reason", None)
    ]
    assert finishes == ["length"], f"message_delta 的 max_tokens 应译为 length，实际 {finishes}"


def test_nonstream_finish_reason_mapped():
    """非流式 _response_to_openai 顶层 resp.stop_reason 同走词表归一
    （course_topic_builder 等按 OpenAI 方言判 length 做快速放弃）。"""
    from types import SimpleNamespace
    from core.llm.providers.anthropic_adapter import _response_to_openai

    resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="半截 JSON")],
        stop_reason="max_tokens",
        usage=None,
        model="claude-test",
    )
    out = _response_to_openai(resp)
    assert out.choices[0].finish_reason == "length"
