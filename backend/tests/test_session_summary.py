"""P0-B：L2 摘要结构化 JSON 抽取 + 代码增量合并 + 降级兜底 单测。

不依赖真实 LLM：mock async_openai_client.chat.completions.create 控制输出，验证
解析/合并/格式化的纯逻辑 + _do_compress 的重试与降级路径 + get_summary 的 JSON→文本转换。
摘要质量（LLM 抽得好不好）待真环境；本文件只验「逻辑正确 + 降级不阻塞」。
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory import session_summary as ss
from core.memory.session_summary import SessionSummaryManager


# ── _parse_structured ───────────────────────────────────────────────────────

def test_parse_structured_valid_json():
    assert ss._parse_structured('{"topics": ["a"]}') == {"topics": ["a"]}


def test_parse_structured_invalid_returns_none():
    assert ss._parse_structured("这不是JSON") is None
    assert ss._parse_structured("") is None
    assert ss._parse_structured('["a","b"]') is None  # list 非 dict
    assert ss._parse_structured(None) is None


# ── _parse_json_loose（宽松解析 LLM 输出）──────────────────────────────────

def test_parse_json_loose_plain():
    assert ss._parse_json_loose('{"facts": ["x"]}') == {"facts": ["x"]}


def test_parse_json_loose_markdown_fence():
    raw = "```json\n{\"facts\": [\"y\"]}\n```"
    assert ss._parse_json_loose(raw) == {"facts": ["y"]}


def test_parse_json_loose_with_noise():
    raw = '好的，结果如下：\n{"facts": ["z"]}\n以上。'
    assert ss._parse_json_loose(raw) == {"facts": ["z"]}


def test_parse_json_loose_invalid():
    assert ss._parse_json_loose("没有json") is None
    assert ss._parse_json_loose("") is None


# ── _merge_structured（代码增量合并）──────────────────────────────────────

def test_merge_dedup_and_append():
    existing = {"facts": ["学生喜欢Python", "数学基础弱"], "topics": ["循环"]}
    # "学生喜欢python。" 归一化后 == "学生喜欢python"（lower + 去尾句号）→ 与首条重复
    new = {"facts": ["学生喜欢python。", "想学数据结构"], "topics": ["循环"]}
    merged = ss._merge_structured(existing, new)
    assert "学生喜欢Python" in merged["facts"]   # 保留首个原形
    assert "想学数据结构" in merged["facts"]
    assert len(merged["facts"]) == 3             # Python 去重 1 条
    assert merged["topics"] == ["循环"]          # topics 去重


def test_merge_caps_at_max_items():
    existing = {"facts": [f"旧事实{i}" for i in range(4)]}
    new = {"facts": [f"新事实{i}" for i in range(4)]}
    merged = ss._merge_structured(existing, new)
    assert len(merged["facts"]) == ss._MAX_ITEMS_PER_LIST   # 上限 5
    assert "新事实3" in merged["facts"]                     # 保留最新（new 在后优先）
    assert "旧事实0" not in merged["facts"]                 # 丢弃最旧


def test_merge_filters_non_str_and_empty():
    new = {"facts": ["有效", "", 123, None, "也有效"]}
    merged = ss._merge_structured(None, new)
    assert merged["facts"] == ["有效", "也有效"]


def test_merge_missing_keys_default_empty():
    merged = ss._merge_structured(None, {})
    assert all(merged[k] == [] for k in ss._SUMMARY_KEYS)


# ── _format_structured（JSON → 可读文本注入）──────────────────────────────

def test_format_structured_renders_sections():
    d = {"topics": ["T1"], "facts": ["F1", "F2"],
         "decisions": [], "open_questions": [], "action_items": []}
    text = ss._format_structured(d)
    assert "## 会话主题" in text
    assert "- T1" in text
    assert "## 确认的事实" in text
    assert "- F1" in text and "- F2" in text
    assert "关键结论" not in text   # 空节不渲染


def test_format_structured_empty_returns_empty():
    assert ss._format_structured({}) == ""


# ── _do_compress（mock LLM：重试 + 降级）──────────────────────────────────

def _mk_messages(n=3):
    msgs = []
    for i in range(n):
        m = MagicMock()
        m.role = "user" if i % 2 == 0 else "assistant"
        m.content = f"对话内容{i}" * 50
        msgs.append(m)
    return msgs


def _mock_client_with(fake_create):
    """构造 mock async_openai_client，其 chat.completions.create = fake_create。"""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return client


@pytest.mark.asyncio
async def test_do_compress_structured_success():
    """LLM 返回合法 JSON → 返回 merged JSON 字符串（existing + new 代码合并去重）。"""
    mgr = SessionSummaryManager()
    existing_json = json.dumps({"facts": ["旧事实"]}, ensure_ascii=False)

    async def fake_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = '{"facts": ["旧事实", "新事实"], "topics": ["T"]}'
        return resp

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        out = await mgr._do_compress(existing_json, _mk_messages())

    assert out is not None
    d = json.loads(out)   # 返回的是合法 JSON 字符串
    assert "新事实" in d["facts"]
    assert "T" in d["topics"]


@pytest.mark.asyncio
async def test_do_compress_retries_then_succeeds():
    """第一次非 JSON（temp=0.3），第二次合法 JSON（temp=0）→ 第二次成功，不降级。"""
    mgr = SessionSummaryManager()
    seq = ["不是json", '{"facts": ["ok"]}']

    async def fake_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = seq.pop(0)
        return resp

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        out = await mgr._do_compress("", _mk_messages())

    d = json.loads(out)
    assert d["facts"] == ["ok"]


@pytest.mark.asyncio
async def test_do_compress_falls_back_to_text_on_bad_json():
    """LLM 连续返回非 JSON → 重试一次（temp=0）后降级 _do_compress_text。"""
    mgr = SessionSummaryManager()
    temps = []

    async def fake_create(**kwargs):
        temps.append(kwargs.get("temperature"))
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "这不是JSON，重试也没用"
        return resp

    async def fake_text(existing, msgs):
        return "降级文本摘要"

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        with patch.object(mgr, "_do_compress_text", new=AsyncMock(side_effect=fake_text)):
            out = await mgr._do_compress("", _mk_messages())

    assert temps == [0.3, 0.0]          # 结构化试了两次（首试 + temp=0 重试）
    assert out == "降级文本摘要"          # 降级到旧文本逻辑


# ── get_summary（注入点拿到可读文本）──────────────────────────────────────

@pytest.mark.asyncio
async def test_get_summary_formats_json_to_text():
    mgr = SessionSummaryManager()
    session = MagicMock()
    session.summary = json.dumps({"topics": ["主题A"], "facts": ["事实B"]}, ensure_ascii=False)
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)

    out = await mgr.get_summary(db, "s1")
    assert "## 会话主题" in out
    assert "主题A" in out
    assert "事实B" in out


@pytest.mark.asyncio
async def test_get_summary_passthrough_legacy_text():
    """旧文本格式（非 JSON）原样返回，兼容历史 session。"""
    mgr = SessionSummaryManager()
    session = MagicMock()
    session.summary = "这是旧的自由文本摘要。"
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)

    out = await mgr.get_summary(db, "s1")
    assert out == "这是旧的自由文本摘要。"
