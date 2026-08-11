"""L2 摘要 v2（slot key + 时间戳裁决 + 显著度淘汰）单测。

不依赖真实 LLM：mock async_openai_client.chat.completions.create 控制输出，验证
解析/合并/淘汰/渲染的纯逻辑 + _do_compress 的重试与降级路径 + get_summary 的 v2->文本转换。
摘要质量（LLM 抽得好不好）待真环境；本文件只验「逻辑正确 + 降级不阻塞」。
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory import session_summary as ss
from core.memory.session_summary import SessionSummaryManager, SummaryItem


def _item(k: str, key: str, t: str, ts: float = 1000.0, n: int = 1) -> SummaryItem:
    return SummaryItem(k=k, key=key, t=t, ts=ts, n=n)


# ── _parse_structured（三格式识别）──────────────────────────────────────────

def test_parse_structured_v2_direct_read():
    raw = json.dumps({"v": 2, "items": [
        {"k": "fact", "key": "textbook_edition", "t": "第6版", "ts": 1.0, "n": 2},
    ]}, ensure_ascii=False)
    items = ss._parse_structured(raw)
    assert items is not None
    assert len(items) == 1
    assert items[0].k == "fact"
    assert items[0].key == "textbook_edition"
    assert items[0].t == "第6版"
    assert items[0].ts == 1.0
    assert items[0].n == 2


def test_parse_structured_v1_upgrade():
    """v1 五键 dict 就地升级：key 用文本哈希、ts=0、action_items->next_step。"""
    raw = json.dumps({"facts": ["旧事实"], "action_items": ["下次讲三相"]}, ensure_ascii=False)
    items = ss._parse_structured(raw)
    assert items is not None
    ks = {it.k for it in items}
    assert ks == {"fact", "next_step"}  # action_items 改名 next_step
    assert all(it.ts == 0.0 for it in items)  # legacy 最低显著度


def test_parse_structured_invalid_returns_none():
    assert ss._parse_structured("这不是JSON") is None
    assert ss._parse_structured("") is None
    assert ss._parse_structured('["a","b"]') is None  # list 非 dict
    assert ss._parse_structured(None) is None


def test_parse_structured_v2_drops_unknown_kind():
    raw = json.dumps({"v": 2, "items": [
        {"k": "fact", "key": "x", "t": "y"},
        {"k": "bogus", "key": "z", "t": "w"},  # 未知 kind 丢弃
        {"k": "topic", "t": ""},  # 空文本丢弃
    ]}, ensure_ascii=False)
    items = ss._parse_structured(raw)
    assert items is not None
    assert len(items) == 1
    assert items[0].k == "fact"


# ── _parse_json_loose（宽松解析 LLM 输出，逻辑未变）────────────────────────

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


# ── _parse_llm_output + _merge_items（裁决合并）────────────────────────────

def test_merge_single_value_overwrite():
    """单值槽 fact 同 key：新值覆盖旧值，旧值不并存（治 P0 矛盾并存）。"""
    existing = [_item("fact", "textbook_edition", "第5版", ts=1.0)]
    new = [_item("fact", "textbook_edition", "第6版", ts=2.0)]
    merged = ss._merge_items(existing, new)
    texts = [it.t for it in merged]
    assert "第6版" in texts
    assert "第5版" not in texts  # 旧值被覆盖丢弃


def test_merge_single_value_same_value_increments_n():
    """单值槽同 key 同值：n 续计（稳定事实更显著），ts 前移。"""
    existing = [_item("fact", "textbook_edition", "第6版", ts=1.0, n=1)]
    new = [_item("fact", "textbook_edition", "第6版", ts=2.0, n=1)]
    merged = ss._merge_items(existing, new)
    assert len(merged) == 1
    assert merged[0].n == 2
    assert merged[0].ts == 2.0


def test_merge_single_value_different_key_coexists():
    """单值槽不同 key 共存（教材版本 vs 年级是两个独立当前值）。"""
    existing = [_item("fact", "textbook_edition", "第6版", ts=1.0)]
    new = [_item("fact", "grade_level", "大二", ts=2.0)]
    merged = ss._merge_items(existing, new)
    assert len(merged) == 2


def test_merge_multi_value_same_key_merges():
    """多值槽同 key：文本取最新、ts 前移、n+=1。"""
    existing = [_item("topic", "thevenin", "戴维南求解（部分解答）", ts=1.0, n=2)]
    new = [_item("topic", "thevenin", "戴维南求解（已解答）", ts=2.0, n=1)]
    merged = ss._merge_items(existing, new)
    assert len(merged) == 1
    assert merged[0].t == "戴维南求解（已解答）"  # 最新措辞
    assert merged[0].ts == 2.0
    assert merged[0].n == 3  # 2 + 1


def test_merge_multi_value_different_key_coexists():
    """多值槽不同 key 共存。"""
    existing = [_item("topic", "thevenin", "戴维南（已解答）", ts=1.0)]
    new = [_item("topic", "kvl", "基尔霍夫电压（未解决）", ts=2.0)]
    merged = ss._merge_items(existing, new)
    assert len(merged) == 2


def test_merge_resolved_removes_open_question():
    """resolved 中的 key 删对应 open_question 条目（综述 filtering）。"""
    existing = [
        _item("open_question", "q1", "为什么并联这样算", ts=1.0),
        _item("open_question", "q2", "戴维南等效怎么求", ts=1.0),
    ]
    new = [_item("topic", "thevenin", "戴维南（已解答）", ts=2.0)]
    merged = ss._merge_items(existing, new, resolved=["q2"])
    keys = {it.key for it in merged if it.k == "open_question"}
    assert "q2" not in keys  # 被消除
    assert "q1" in keys  # 未在 resolved 中，保留


def test_merge_empty_inputs_returns_empty():
    assert ss._merge_items([], []) == []
    assert ss._merge_items([], [], resolved=["x"]) == []


# ── _evict_by_budget（显著度淘汰，替 combined[-5:]）────────────────────────

def test_evict_keeps_high_salience_under_budget():
    """预算紧张时保留 salience 高的（fact 权重 1.5 > topic 0.8），丢低 salience。"""
    items = [
        _item("topic", "t1", "主题一" * 20, ts=1.0),   # 低权重
        _item("fact", "f1", "约束一" * 20, ts=1.0),    # 高权重
    ]
    # 预算只够一条（每条约 100 估 token）
    kept = ss._evict_by_budget(items, token_budget=100, max_per_kind=8,
                               now_ts=1.0, half_life_s=3600)
    assert len(kept) == 1
    assert kept[0].k == "fact"  # fact salience 更高被保留


def test_evict_max_per_kind_cap():
    """每类上限独立计数，一类占不满预算挤掉其他类。"""
    items = [_item("topic", f"t{i}", f"主题{i}", ts=1.0) for i in range(10)]
    kept = ss._evict_by_budget(items, token_budget=10000, max_per_kind=3,
                               now_ts=1.0, half_life_s=3600)
    assert len(kept) == 3  # max_per_kind=3 截断


def test_evict_empty_returns_empty():
    assert ss._evict_by_budget([], token_budget=100, max_per_kind=8,
                               now_ts=1.0, half_life_s=3600) == []


# ── _render_items（v2 -> 可读文本注入）──────────────────────────────────────

def test_render_items_renders_sections():
    items = [
        _item("topic", "t1", "戴维南（已解答）", ts=1.0),
        _item("fact", "f1", "教材第6版", ts=1.0),
    ]
    text = ss._render_items(items, now_ts=1.0, half_life_s=3600)
    assert "## 会话主题" in text
    assert "戴维南（已解答）" in text
    assert "## 确认的事实" in text
    assert "教材第6版" in text


def test_render_items_empty_returns_empty():
    assert ss._render_items([], now_ts=1.0, half_life_s=3600) == ""


def test_render_items_skips_empty_kind():
    """无该 kind 条目则不渲染该节。"""
    items = [_item("fact", "f1", "教材第6版", ts=1.0)]
    text = ss._render_items(items, now_ts=1.0, half_life_s=3600)
    assert "## 确认的事实" in text
    assert "会话主题" not in text  # topic 无条目不渲染


# ── _format_msg_for_compress（metadata 输入源扩展）─────────────────────────

def _mk_msg(role="user", content="对话", meta=None):
    m = MagicMock()
    m.role = role
    m.content = content
    m.metadata_ = json.dumps(meta or {}, ensure_ascii=False)
    m.created_at = 1000.0
    return m


def test_format_msg_with_tools_and_refs():
    m = _mk_msg("assistant", "解答如下", {
        "tools_used": ["rag", "web_search"],
        "chunks": [{"source": "第3章 戴维南定理"}, {"source": "例题3-7"}],
    })
    line = SessionSummaryManager()._format_msg_for_compress(m)
    assert "[tools: rag, web_search]" in line
    assert "[refs: 第3章 戴维南定理, 例题3-7]" in line


def test_format_msg_with_attachments():
    m = _mk_msg("user", "这题怎么做", {"attachments": [{"filename": "电路图.png"}]})
    line = SessionSummaryManager()._format_msg_for_compress(m)
    assert "[attachments: 电路图.png]" in line


def test_format_msg_no_metadata_plain():
    m = _mk_msg("user", "纯文本消息", {})
    line = SessionSummaryManager()._format_msg_for_compress(m)
    assert line == "user: 纯文本消息"


# ── _do_compress（mock LLM：重试 + 降级）──────────────────────────────────

def _mk_messages(n=3):
    msgs = []
    for i in range(n):
        m = MagicMock()
        m.role = "user" if i % 2 == 0 else "assistant"
        m.content = f"对话内容{i}" * 50
        m.metadata_ = "{}"
        m.created_at = 1000.0 + i
    return msgs


def _mock_client_with(fake_create):
    """构造 mock async_openai_client，其 chat.completions.create = fake_create。"""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    return client


@pytest.mark.asyncio
async def test_do_compress_structured_success():
    """LLM 返回 v2 JSON -> 返回 merged v2 JSON 字符串。"""
    mgr = SessionSummaryManager()

    async def fake_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({
            "items": [{"k": "fact", "key": "textbook_edition", "t": "第6版"}],
            "resolved": [],
        }, ensure_ascii=False)
        return resp

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        out = await mgr._do_compress("", _mk_messages())

    assert out is not None
    d = json.loads(out)
    assert d["v"] == 2
    texts = [it["t"] for it in d["items"]]
    assert "第6版" in texts


@pytest.mark.asyncio
async def test_do_compress_merges_existing_and_overwrites():
    """existing v2 有 fact 旧值，LLM 抽同 key 新值 -> 旧值被覆盖（P0 修复端到端）。"""
    mgr = SessionSummaryManager()
    existing_json = json.dumps({"v": 2, "items": [
        {"k": "fact", "key": "textbook_edition", "t": "第5版", "ts": 1.0, "n": 1},
    ]}, ensure_ascii=False)

    async def fake_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = json.dumps({
            "items": [{"k": "fact", "key": "textbook_edition", "t": "第6版"}],
            "resolved": [],
        }, ensure_ascii=False)
        return resp

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        out = await mgr._do_compress(existing_json, _mk_messages())

    d = json.loads(out)
    texts = [it["t"] for it in d["items"]]
    assert "第6版" in texts
    assert "第5版" not in texts  # 单值槽覆盖


@pytest.mark.asyncio
async def test_do_compress_retries_then_succeeds():
    """第一次非 JSON（temp=0.3），第二次合法 v2（temp=0）-> 第二次成功，不降级。"""
    mgr = SessionSummaryManager()
    seq = ["不是json", json.dumps({"items": [{"k": "topic", "key": "t", "t": "主题"}]})]

    async def fake_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = seq.pop(0)
        return resp

    with patch.object(ss, "async_openai_client", _mock_client_with(fake_create)):
        out = await mgr._do_compress("", _mk_messages())

    d = json.loads(out)
    assert d["v"] == 2
    assert any(it["t"] == "主题" for it in d["items"])


@pytest.mark.asyncio
async def test_do_compress_falls_back_to_text_on_bad_json():
    """LLM 连续返回非 JSON -> 重试一次（temp=0）后降级 _do_compress_text。"""
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


# ── get_summary（注入点拿到可读文本）────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_summary_formats_v2_to_text():
    mgr = SessionSummaryManager()
    session = MagicMock()
    session.summary = json.dumps({"v": 2, "items": [
        {"k": "topic", "key": "t", "t": "主题A", "ts": 1.0, "n": 1},
        {"k": "fact", "key": "f", "t": "事实B", "ts": 1.0, "n": 1},
    ]}, ensure_ascii=False)
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)

    out = await mgr.get_summary(db, "s1")
    assert "## 会话主题" in out
    assert "主题A" in out
    assert "## 确认的事实" in out
    assert "事实B" in out


@pytest.mark.asyncio
async def test_get_summary_upgrades_v1_to_text():
    """v1 五键 dict 自动升级并渲染。"""
    mgr = SessionSummaryManager()
    session = MagicMock()
    session.summary = json.dumps({"topics": ["主题A"], "facts": ["事实B"]}, ensure_ascii=False)
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)

    out = await mgr.get_summary(db, "s1")
    assert "## 会话主题" in out
    assert "主题A" in out
    assert "## 确认的事实" in out
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
