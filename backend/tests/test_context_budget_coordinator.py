"""通用上下文管理：三级级联 enforce + 反应式 plan_turn 测试。

覆盖：
- evict_tool_results（L1）：超预算优先级清理、generic 先于 source、recent 轮不动、
  ask_user/exclude_tools 白名单永不清理、墓碑格式、捕获被清原文、达标即停、不超预算 noop
- enforce（级联）：L1 清 tool 结果、L2 LLM 摘要续接、L3 丢最旧 20% 消息组、未超软阈值 noop
- plan_turn（反应式）：未超软阈值原样返回、超了才裁历史、有摘要前插 <session_summary>
- drop_oldest_turn_group（L3）：丢最旧 20%、保留前导、轮少不动
"""
from __future__ import annotations

import pytest

from core.agentic import context_budget
from core.agentic.context_budget import (
    EnforceReport,
    TurnBudgetPlan,
    drop_oldest_turn_group,
    enforce,
    evict_tool_results,
    plan_turn,
)
from settings import get_settings


def _assistant_with_tools(call_id: str, name: str = "rag") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": "{}"}}],
    }


# ---------------------------------------------------------------------------
# evict_tool_results（L1）
# ---------------------------------------------------------------------------

def test_evict_noop_under_budget():
    msgs = [
        {"role": "system", "content": "sys"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": "short"},
    ]
    assert evict_tool_results(msgs, keep_recent_turns=1, target_tokens=100000) == 0
    assert msgs[2]["content"] == "short"  # 未动


def test_evict_clears_generic_before_source():
    generic = "普通结果" * 200   # ~800 token
    source = "内容\n\n来源: doc1"  # 带来源引用段，score +2
    msgs = [
        {"role": "system", "content": "sys"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": generic},   # older, generic
        {"role": "tool", "name": "rag", "content": source},    # older, source-bearing
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},  # recent 轮，保留
    ]
    # target=100：远小于 generic(800) 逼出 generic；又 > 清完 generic 后的余量
    #（~72 = sys+两 assistant tool_calls+tombstone+source+recent），保住 source。
    # 2.6 起 _messages_tokens 计入 tool_calls，旧 target=20 连 tool_calls 都装不下。
    target = 100
    cleared = evict_tool_results(msgs, keep_recent_turns=1, target_tokens=target)
    contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 1
    assert any(c.startswith("[cleared: rag]") for c in contents)  # generic 清成墓碑
    assert source in contents                                        # source 保留
    assert "recent" in contents                                      # recent 保留


def test_evict_keeps_recent_turns():
    big = "x" * 4000  # ~1000 token
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": big},  # older，应清
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": big},  # recent，必留
    ]
    cleared = evict_tool_results(msgs, keep_recent_turns=1, target_tokens=50)
    tool_contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 1
    assert sum(1 for c in tool_contents if c.startswith("[cleared:")) == 1
    assert big in tool_contents  # recent 未动


def test_evict_ask_user_never_cleared_by_default():
    """exclude_tools 默认含 ask_user -> 用户答复永不清理。"""
    big = "y" * 4000
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1", name="ask_user"),
        {"role": "tool", "name": "ask_user", "content": big},  # 用户答复，永不清理
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    cleared = evict_tool_results(msgs, keep_recent_turns=1, target_tokens=10)
    contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 0  # ask_user 是 older 唯一候选但被白名单排除 -> 无可清
    assert not any(c.startswith("[cleared:") for c in contents)
    assert big in contents  # ask_user 内容原样保留


def test_evict_exclude_tools_whitelist_param():
    """显式 exclude_tools 参数保护指定 tool 不被清理。"""
    big = "z" * 4000
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "search", "content": big},  # older，但被白名单保护
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    cleared = evict_tool_results(msgs, keep_recent_turns=1, target_tokens=10,
                                 exclude_tools=["search"])
    contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 0  # search 被白名单排除，older 无其他可清候选
    assert big in contents  # search 原样保留


def test_evict_captures_originals_for_l2():
    """captured_originals 收集被清原文，供 L2 摘要复用。"""
    big = "z" * 4000
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": big},
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    captured: list[str] = []
    evict_tool_results(msgs, keep_recent_turns=1, target_tokens=50,
                       captured_originals=captured)
    assert len(captured) == 1
    assert captured[0] == big  # 被清原文


def test_evict_tombstone_format():
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": "z" * 4000},
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "r"},
    ]
    evict_tool_results(msgs, keep_recent_turns=1, target_tokens=10)
    cleared = [str(m.get("content")) for m in msgs if m["role"] == "tool" and str(m.get("content", "")).startswith("[cleared:")]
    assert cleared
    assert cleared[0] == "[cleared: rag]"


# ---------------------------------------------------------------------------
# drop_oldest_turn_group（L3）
# ---------------------------------------------------------------------------

def test_drop_oldest_drops_20_percent():
    """5 个 tool 轮 -> 丢最旧 1 轮（20%），保留前导 + 最近 4 轮。"""
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(5):
        msgs.append(_assistant_with_tools(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "rag",
                     "content": f"result-{i}"})
    msgs.append({"role": "assistant", "content": "final"})
    before = len(msgs)
    dropped = drop_oldest_turn_group(msgs)
    assert dropped >= 2  # 至少丢一个 tool 轮（assistant+tool = 2 条）
    assert len(msgs) < before
    # 最旧的 result-0 被丢
    contents = [str(m.get("content")) for m in msgs]
    assert "result-0" not in contents
    # 最近的 result-4 保留
    assert "result-4" in contents
    # 前导 system 保留
    assert msgs[0]["role"] == "system"


def test_drop_oldest_noop_when_too_few_turns():
    """仅 1 个 tool 轮 + 前导 -> 无余量可丢。"""
    msgs = [
        {"role": "system", "content": "sys"},
        _assistant_with_tools("c0"),
        {"role": "tool", "tool_call_id": "c0", "name": "rag", "content": "r"},
    ]
    before = list(msgs)
    dropped = drop_oldest_turn_group(msgs)
    assert dropped == 0
    assert msgs == before


# ---------------------------------------------------------------------------
# enforce（三级级联）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_noop_under_soft(monkeypatch):
    """未超软阈值 -> 三级都不动作，tokens 不变。"""
    monkeypatch.setattr(context_budget, "compute_budgets",
                        lambda m: (100000, 200000))
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert isinstance(report, EnforceReport)
    assert report.cleared_tool_results == 0
    assert report.summary_added is False
    assert report.dropped_messages == 0
    assert report.tokens_after == report.tokens_before


@pytest.mark.asyncio
async def test_enforce_l1_clears(monkeypatch):
    """超软阈值 -> L1 清旧 tool 结果，L1 够则不升级 L2/L3。"""
    monkeypatch.setattr(context_budget, "compute_budgets", lambda m: (100, 200))
    monkeypatch.setattr(get_settings().context_budget, "keep_recent_turns", 1)
    big = "z" * 4000  # ~1000 token
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": big},  # older，应清
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.cleared_tool_results == 1
    assert report.tokens_after < report.tokens_before
    assert report.summary_added is False  # L1 够，不升级
    assert report.dropped_messages == 0


@pytest.mark.asyncio
async def test_enforce_l2_summary_when_l1_insufficient(monkeypatch):
    """L1 清完仍超软阈值 -> L2 LLM 摘要续接（mock 摘要）。"""
    from core.agentic import context_policy as cp

    monkeypatch.setattr(context_budget, "compute_budgets", lambda m: (100, 100000))
    monkeypatch.setattr(get_settings().context_budget, "keep_recent_turns", 1)

    async def fake_summarize(text, model):
        return "压缩摘要"
    monkeypatch.setattr(cp, "_summarize_masked_text", fake_summarize)

    big_tool = "z" * 4000       # ~1000 token，L1 会清
    big_user = "u" * 4000       # ~1000 token，L1 清不动（非 tool），保 total > soft
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": big_user},  # 非 tool，L1 不动 -> 保 total > 100
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": big_tool},  # older，L1 清
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.cleared_tool_results == 1  # L1 清了 big_tool
    assert report.summary_added is True       # 仍超 soft -> L2 摘要
    # 摘要以 <context_summary> system 消息插入
    assert any(m.get("role") == "system" and "<context_summary>" in str(m.get("content"))
               for m in msgs)


@pytest.mark.asyncio
async def test_enforce_l3_drop_when_over_hard_ceiling(monkeypatch):
    """仍超硬天花板 -> L3 丢最旧 20% 消息组。"""
    from core.agentic import context_policy as cp

    # soft 极小（L1/L2 都触发后仍超 hard），hard 也小 -> L3 必触发
    monkeypatch.setattr(context_budget, "compute_budgets", lambda m: (50, 100))
    monkeypatch.setattr(get_settings().context_budget, "keep_recent_turns", 1)

    async def fake_summarize(text, model):
        return ""  # L2 摘要返回空 -> 不续接，逼 L3
    monkeypatch.setattr(cp, "_summarize_masked_text", fake_summarize)

    # 构造 5 个 tool 轮，每轮 tool 结果大（total >> hard=100）
    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"u{i} " + "x" * 3000})
        msgs.append(_assistant_with_tools(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "rag",
                     "content": "z" * 3000})
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.dropped_messages > 0  # L3 丢了消息
    assert len(msgs) < 5 * 4 + 1  # 消息变少


# ---------------------------------------------------------------------------
# plan_turn（反应式）
# ---------------------------------------------------------------------------

def test_plan_turn_reactive_under_soft_no_trim():
    """未超软阈值 -> 原样返回，不裁、不续接。"""
    plan = plan_turn(history=[{"role": "user", "content": "hi"}],
                     model="qwen-plus", session_summary_text="摘要")
    assert isinstance(plan, TurnBudgetPlan)
    assert plan.dropped_count == 0
    assert plan.carry_forward_added is False
    assert plan.keep_history == [{"role": "user", "content": "hi"}]
    # 预算字段：target_input_tokens=soft_trigger，mecw=effective_window，hard_ceiling 在
    assert plan.target_input_tokens == 128000     # qwen-plus soft_trigger
    assert plan.mecw == 1_000_000                 # qwen-plus effective_window
    assert plan.hard_ceiling > 0


def test_plan_turn_reactive_over_soft_trims(monkeypatch):
    """超软阈值、无摘要 -> 裁历史到 soft_trigger 预算内，不续接（no-loss）。"""
    monkeypatch.setattr(get_settings().context_budget, "coordinator_enabled", True)
    big = "x" * 200000  # 远超 soft_trigger（qwen-max 比例线 16384）
    history = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "recent question"},
    ]
    plan = plan_turn(history=history, model="qwen-max", session_summary_text="")
    assert plan.dropped_count >= 1
    assert plan.carry_forward_added is False  # 无摘要不续接
    # recent 保留（从近到远裁）
    assert any(m.get("content") == "recent question" for m in plan.keep_history)


def test_plan_turn_carry_forward_with_summary(monkeypatch):
    """超软阈值 + 有摘要 + history_prefix -> 裁历史并前插 <session_summary> 续接。"""
    monkeypatch.setattr(get_settings().context_budget, "coordinator_enabled", True)
    monkeypatch.setattr(get_settings().context_budget, "carry_forward_location", "history_prefix")
    big = "x" * 200000
    history = [
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": "recent question"},
    ]
    plan = plan_turn(history=history, model="qwen-max", session_summary_text="早期对话摘要内容")
    assert plan.carry_forward_added is True
    assert plan.dropped_count >= 1
    assert plan.keep_history[0]["role"] == "system"
    assert "<session_summary>" in plan.keep_history[0]["content"]
    assert "早期对话摘要内容" in plan.keep_history[0]["content"]
    # recent 仍在（续接块之后）
    assert any(m.get("content") == "recent question" for m in plan.keep_history)


def test_plan_turn_no_drop_no_carry(monkeypatch):
    """超软阈值但历史很短（裁不掉）-> 不续接。"""
    monkeypatch.setattr(get_settings().context_budget, "coordinator_enabled", True)
    monkeypatch.setattr(get_settings().context_budget, "carry_forward_location", "history_prefix")
    history = [{"role": "user", "content": "hi"}]
    plan = plan_turn(history=history, model="qwen-max", session_summary_text="摘要")
    assert plan.dropped_count == 0
    assert plan.carry_forward_added is False
    assert plan.keep_history == history
