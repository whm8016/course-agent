"""recent_discussion 层（跨模式「刚才聊了什么」）单测。

Bug 场景：同一会话先 chat 讨论知识点，再切出题/解题/研究说「针对刚才讲的出几道题」——
M-6 把三条深度管线的 conversation_history 清成对话轮次空表后，模型看不到任何历史，
指代落空。修复：build_common_context_layers 为非 chat 模式渲染末尾几轮原文摘录进
system prompt 背景段；chat 模式门控不注入（其 history 本就原样进 _build_messages，
再注入会重复）。M-6 隔离语义不受影响（对话轮次与背景资料两条通道分离，
test_m6_solve_history_isolation 仍须通过）。

asyncio_mode=auto（pyproject.toml），async def test 自动按 asyncio 跑。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from core.context import UnifiedContext
from core.pipeline_common import (
    CommonContextLayers,
    assemble_common_context,
    build_common_context_layers,
)


def _history(*pairs: tuple[str, str]) -> list[dict]:
    """按 (user, assistant) 对构造 OpenAI message 格式历史。"""
    out: list[dict] = []
    for u, a in pairs:
        out.append({"role": "user", "content": u})
        out.append({"role": "assistant", "content": a})
    return out


async def _build(mode: str, history: list[dict]) -> CommonContextLayers:
    """无 user/course（跳过 stitch/mem0 DB 路径），mock get_course_prompt 后组装层。"""
    ctx = UnifiedContext(user_message="出题", mode=mode, conversation_history=history)
    with patch("core.llm.prompts.get_course_prompt", new=AsyncMock(return_value="")):
        return await build_common_context_layers(ctx)


@pytest.mark.parametrize("mode", ["deep_solve", "deep_research", "quiz"])
async def test_recent_discussion_rendered_for_non_chat(mode):
    """非 chat 三模式：末轮原文进 recent_discussion，带防注入口径标题。"""
    history = _history(
        ("戴维南等效怎么用？", "把复杂电路换成电源+内阻再看…"),
        ("针对这个出几道题", "好的"),
    )
    layers = await _build(mode, history)
    assert "戴维南等效怎么用？" in layers.recent_discussion
    assert "仅供参考背景" in layers.recent_discussion


async def test_recent_discussion_gated_for_chat():
    """chat 门控：history 本就进 _build_messages，再注入会重复，必须为空。"""
    layers = await _build("chat", _history(("刚才讲到哪了", "讲到线性电路")))
    assert layers.recent_discussion == ""


async def test_recent_discussion_keeps_tail_pairs_only():
    """5 轮历史只带末 3 轮（前 2 轮不进摘录）。"""
    history = _history(
        ("第1轮", "答1"), ("第2轮", "答2"), ("第3轮", "答3"), ("第4轮", "答4"), ("第5轮", "答5"),
    )
    layers = await _build("quiz", history)
    assert "第3轮" in layers.recent_discussion
    assert "第1轮" not in layers.recent_discussion


async def test_recent_discussion_truncates_long_message():
    """单条超 400 字截断加省略号（防一条长回答撑爆背景段）。"""
    layers = await _build("deep_solve", [{"role": "user", "content": "长" * 500}])
    # 标题(~30) + "用户："(3) + 400 + "…" ≈ 434，远小于 500 原文
    assert len(layers.recent_discussion) < 500
    assert layers.recent_discussion.endswith("…")


async def test_recent_discussion_skips_empty_messages():
    """空内容消息（如 quiz 结果消息 content=''）跳过，不占摘录名额。"""
    history = [
        {"role": "user", "content": "讲讲戴维南"},
        {"role": "assistant", "content": ""},
    ]
    layers = await _build("quiz", history)
    assert "讲讲戴维南" in layers.recent_discussion
    assert "用户：" in layers.recent_discussion  # 只剩 user 一条，仍正常渲染


async def test_recent_discussion_empty_history():
    """无历史（直接开新会话出题）：层为空串，被 assemble 过滤。"""
    layers = await _build("quiz", [])
    assert layers.recent_discussion == ""


@pytest.mark.parametrize(
    "bad_history",
    [
        "abc",  # WS 入口 history 无 pydantic 校验，字符串能活着走到这里
        [{"role": "user", "content": 123}, 5, None],  # 非 dict 元素 / 非字符串 content
    ],
)
async def test_recent_discussion_tolerates_malformed_history(bad_history):
    """畸形 history 不炸整轮（M-6 时代这些被丢弃无害，本层引入后须自我防御）。"""
    layers = await _build("quiz", bad_history)
    assert layers.recent_discussion == ""


async def test_recent_discussion_joins_multimodal_content_parts():
    """content 为多模态 parts 列表（合法形态，_build_messages 支持）时拼接 text part，不打 repr 噪声。"""
    history = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "看这张电路图"}, {"type": "image_url"}],
        }
    ]
    layers = await _build("quiz", history)
    assert "看这张电路图" in layers.recent_discussion
    assert "{" not in layers.recent_discussion and "[" not in layers.recent_discussion


def test_assemble_places_recent_after_summary_before_now():
    """拼接顺序契约：summary（压缩，较稳）→ recent（原文，较新）→ now。"""
    layers = CommonContextLayers(
        session_summary="## 前情摘要",
        recent_discussion="## 刚才的对话",
        now_text="【当前时间】x",
    )
    out = assemble_common_context(layers)
    assert out.index("前情摘要") < out.index("刚才的对话") < out.index("【当前时间】")
