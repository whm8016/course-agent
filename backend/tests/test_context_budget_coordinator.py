"""通用上下文管理：轮内单 pass 压缩 enforce + 反应式 plan_turn 测试。

覆盖：
- clear_old_tool_results（①清理）：时间序最旧优先、recent 轮不动、ask_user/exclude_tools
  白名单永不清理、RAG 来源段跳过、clear_at_least 门限（可清量不足整体跳过）、
  captured 记 id->原文（供 ②折叠复用）、墓碑格式
- enforce（单 pass）：①清理够则不折叠、②折叠进唯一 <compacted>（替换非叠加，二次
  enforce 幂等续写）、assistant 中间结论进折叠输入（D3 优先级倒挂修复）、tools 计入
  预算、③收缩循环收敛到 hard 以下、无可收缩时显式 raise、tool_use<->tool_result
  配对不变式、answer_now 浅拷贝语义（折叠只影响副本、墓碑经共享 dict 可见）
- plan_turn（反应式）：未超软阈值原样返回、超了才裁历史、有摘要前插 <session_summary>
- drop_oldest_turn_group（413 兜底入口）：丢最旧 20%、保留前导、轮少不动
"""
from __future__ import annotations

import pytest

from core.agentic import context_budget
from core.agentic.context_budget import (
    EnforceReport,
    TurnBudgetPlan,
    clear_old_tool_results,
    drop_oldest_turn_group,
    enforce,
    plan_turn,
)
from settings import get_settings


def _assistant_with_tools(call_id: str, name: str = "rag", content: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": "{}"}}],
    }


def _assert_tool_pairing(messages: list[dict]) -> None:
    """结构不变式：每条 role=tool 前必是 assistant(tool_calls) 或同轮前一条 tool。"""
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            assert i > 0, "tool 消息不能是首条"
            prev = messages[i - 1]
            ok = (prev.get("role") == "assistant" and prev.get("tool_calls")) or (
                prev.get("role") == "tool"
            )
            assert ok, f"tool 消息 {i} 前出现孤儿配对: {prev.get('role')}"


def _enforce_env(monkeypatch, *, soft=100, hard=1_000_000, keep=1, summarize=""):
    """enforce 测试通用环境：双阈值 + 保护窗 + 两个防抖门限置 0（默认 2000 token 会
    静默跳过①清理/②折叠，测试要的是确定性触发）+ 摘要 mock。

    mock 靶点是 context_budget 自己的命名空间：模块顶层已绑定
    context_policy._summarize_masked_text，patch context_policy 侧不生效。
    summarize=None 表示"折叠不应发生，调了即 AssertionError"；传 callable 则收
    折叠输入文本、返回值作摘要正文（记录型测试用）。返回 mock 函数。
    """
    monkeypatch.setattr(context_budget, "compute_budgets", lambda m: (soft, hard))
    monkeypatch.setattr(get_settings().context_budget, "keep_recent_turns", keep)
    monkeypatch.setattr(context_budget, "_CLEAR_AT_LEAST_TOKENS", 0)
    monkeypatch.setattr(context_budget, "_MIN_FOLD_GAIN_TOKENS", 0)
    if summarize is None:
        async def fake(text, model):
            raise AssertionError("不应触发 LLM 折叠")
    else:
        async def fake(text, model):
            return summarize(text) if callable(summarize) else summarize
    monkeypatch.setattr(context_budget, "_summarize_masked_text", fake)
    return fake


def _foldable_msgs(aside: str = "") -> list[dict]:
    """①清后仍超 soft 的标准脚手架：big_user 非 tool 清不动，保 total > soft 逼出②折叠。"""
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u" * 4000},
        _assistant_with_tools("c1", content=aside),
        {"role": "tool", "name": "rag", "content": "z" * 4000},
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]


def _ck_count(messages: list[dict]) -> int:
    """checkpoint 条数（判定的唯一来源是 context_budget._is_checkpoint，测试不另写格式）。"""
    return sum(1 for m in messages if context_budget._is_checkpoint(m))


# ---------------------------------------------------------------------------
# clear_old_tool_results（①清理）
# ---------------------------------------------------------------------------

def test_clear_gate_skips_small_gain():
    """clear_at_least：可清量不足 2000 token 整体跳过（小步清理只打断缓存）。"""
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": "z" * 4000},  # ~1000 token < 2000 门限
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    assert clear_old_tool_results(msgs, keep_recent_turns=1) == 0
    assert not any(
        str(m.get("content")).startswith("[cleared:") for m in msgs if m["role"] == "tool"
    )


def test_clear_all_candidates_chronological():
    """时间序最旧优先：保护窗外候选一次性全清（已清区收敛成单调前缀，护 cache）。"""
    old_a, old_b = "a" * 12000, "b" * 12000  # 各 ~3000 token，过门限
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": old_a},   # 最旧
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": old_b},
        _assistant_with_tools("c3"),
        {"role": "tool", "name": "rag", "content": "recent"},  # keep=1 保护
    ]
    captured: dict[int, str] = {}
    cleared = clear_old_tool_results(msgs, keep_recent_turns=1, captured_originals=captured)
    contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 2
    assert contents[0] == "[cleared: rag]" and contents[1] == "[cleared: rag]"
    assert contents[2] == "recent"  # 保护窗内原文保留
    # captured 按时间序记 id -> 原文（②折叠用原文而非墓碑）
    assert list(captured.values()) == [old_a, old_b]


def test_clear_keeps_whitelist_and_source_para():
    """exclude_tools 白名单与 RAG 来源引用段永不清理（保留原地）。"""
    big = "x" * 12000
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1", name="ask_user"),
        {"role": "tool", "name": "ask_user", "content": big},     # 白名单
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "正文\n\n来源: doc2"},  # 来源段跳过
        _assistant_with_tools("c3"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    cleared = clear_old_tool_results(msgs, keep_recent_turns=1)
    contents = [str(m.get("content")) for m in msgs if m["role"] == "tool"]
    assert cleared == 0  # 唯一大体量候选是 ask_user（白名单），来源段跳过 -> 无可清
    assert big in contents
    assert any("来源: doc2" in c for c in contents)


def test_clear_exclude_tools_param():
    """显式 exclude_tools 参数保护指定 tool。"""
    big = "z" * 12000
    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "search", "content": big},  # 被白名单保护
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    assert clear_old_tool_results(msgs, keep_recent_turns=1, exclude_tools=["search"]) == 0
    assert big in [str(m.get("content")) for m in msgs if m["role"] == "tool"]


def test_clear_no_tool_turns_noop():
    """无 tool 轮（纯前导）-> 无可清。"""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert clear_old_tool_results(msgs, keep_recent_turns=3) == 0


# ---------------------------------------------------------------------------
# drop_oldest_turn_group（413 反应式兜底入口）
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
    contents = [str(m.get("content")) for m in msgs]
    assert "result-0" not in contents  # 最旧的被丢
    assert "result-4" in contents      # 最近的保留
    assert msgs[0]["role"] == "system"  # 前导保留
    _assert_tool_pairing(msgs)


def test_drop_oldest_noop_when_too_few_turns():
    """仅 1 个 tool 轮 + 前导 -> 无余量可丢。"""
    msgs = [
        {"role": "system", "content": "sys"},
        _assistant_with_tools("c0"),
        {"role": "tool", "tool_call_id": "c0", "name": "rag", "content": "r"},
    ]
    before = list(msgs)
    assert drop_oldest_turn_group(msgs) == 0
    assert msgs == before


# ---------------------------------------------------------------------------
# enforce（轮内单 pass 压缩）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_noop_under_soft(monkeypatch):
    """未超软阈值 -> 零干预，tokens 不变。"""
    _enforce_env(monkeypatch, soft=100000, hard=200000)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert isinstance(report, EnforceReport)
    assert report.cleared_tool_results == 0
    assert report.summary_added is False
    assert report.dropped_messages == 0
    assert report.tokens_after == report.tokens_before


@pytest.mark.asyncio
async def test_enforce_tools_counted_in_budget(monkeypatch):
    """tools 定义计入请求前预算（对标 LiteLLM _count_extra）：messages 未超但含 tools
    超 soft -> ①清理触发。修复前 schema 体积（教学 agent 十几个工具 ≈3-8k token）
    完全不计，soft 判断偏乐观。"""
    from core.agentic.context_window import count_tokens_messages

    # 本测关注预算口径（非 clear_at_least 行为）：门限置 0 让清理确定性触发；soft 稍后
    # 按 msgs/tools 实测重设
    _enforce_env(monkeypatch, soft=1, hard=200000, summarize="")

    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": "检索结果：" + "课程材料引用。 " * 200},
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": f"academic_tool_{i}",
            "description": "Search course materials and return relevant chunk references. " * 40,
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    } for i in range(4)]
    # soft 夹在 messages 与 messages+tools 之间（同源函数计算，免疫 tokenizer 波动）
    msgs_tokens = count_tokens_messages(msgs)
    joint_tokens = count_tokens_messages(msgs, tools=tools)
    tools_extra = count_tokens_messages([], tools=tools)
    assert joint_tokens == msgs_tokens + tools_extra - 4  # 联合调用含 system×tools 的 -4 折扣
    assert tools_extra > 100  # schema 确有体积
    soft = msgs_tokens + tools_extra // 2
    monkeypatch.setattr(context_budget, "compute_budgets", lambda m: (soft, 200000))

    # 对照：不带 tools 时 messages < soft，不触发
    msgs_plain = [dict(m) for m in msgs]
    report_plain = await enforce(msgs_plain, "qwen-plus", plan=None)
    assert report_plain.cleared_tool_results == 0

    # 带 tools：总量超 soft -> ①清 older tool 结果
    report = await enforce(msgs, "qwen-plus", plan=None, tools=tools)
    assert report.cleared_tool_results == 1
    assert report.tokens_before == joint_tokens  # 报告口径含 tools（请求真实体积）
    assert report_plain.tokens_before == msgs_tokens  # 对照组不含 tools


@pytest.mark.asyncio
async def test_enforce_clear_suffices_no_fold(monkeypatch):
    """①清理够（清后回到 soft 下）-> 不折叠、不收缩。"""
    _enforce_env(monkeypatch, soft=100, hard=200000, summarize=None)

    msgs = [
        {"role": "system", "content": "s"},
        _assistant_with_tools("c1"),
        {"role": "tool", "name": "rag", "content": "z" * 4000},  # ~1000 token，清后回到 soft 下
        _assistant_with_tools("c2"),
        {"role": "tool", "name": "rag", "content": "recent"},
    ]
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.cleared_tool_results == 1
    assert report.tokens_after < report.tokens_before
    assert report.summary_added is False
    assert report.dropped_messages == 0
    assert report.cleared_originals == [("z" * 4000)[:200]]
    assert report.summary_text == ""
    _assert_tool_pairing(msgs)


@pytest.mark.asyncio
async def test_enforce_fold_replaces_segment_with_single_checkpoint(monkeypatch):
    """①清后仍超 soft -> ②折叠：区段整段删除、唯一 <compacted> 顶替（替换非叠加），
    checkpoint_idx 指向其位置。"""
    _enforce_env(monkeypatch, summarize="压缩摘要")

    big_tool = "z" * 4000       # ①会清
    msgs = _foldable_msgs()
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.cleared_tool_results == 1
    assert report.summary_added is True
    assert report.summary_text == "压缩摘要"
    assert report.compacted_tokens > 0
    # 前导(system,user) + 唯一 checkpoint + 保护轮(A2,T2) = 5 条
    assert _ck_count(msgs) == 1
    assert len(msgs) == 5
    assert context_budget._is_checkpoint(msgs[2]) and "压缩摘要" in msgs[2]["content"]
    assert report.checkpoint_idx == 2                      # checkpoint 在前导之后、保护轮之前
    assert big_tool[:100] not in "".join(str(m.get("content")) for m in msgs)  # 区段原文已删除
    assert any(m.get("content") == "recent" for m in msgs)  # 保护轮原样
    assert report.cleared_originals == [big_tool[:200]]     # 观测 preview
    _assert_tool_pairing(msgs)


@pytest.mark.asyncio
async def test_enforce_fold_idempotent_and_rolling(monkeypatch):
    """二次 enforce 幂等：<compacted> 仍唯一（不累积），折叠输入含旧 checkpoint 正文
    （rolling 续写，非另起一条）。"""
    fold_inputs: list[str] = []

    def _record_summary(text):
        fold_inputs.append(text)
        return f"第{len(fold_inputs)}轮摘要"

    _enforce_env(monkeypatch, summarize=_record_summary)

    msgs = _foldable_msgs()
    msgs[5]["content"] = "r2"
    await enforce(msgs, "qwen-plus", plan=None)
    # 新一轮工具追加后再 enforce：旧 checkpoint 应被续写而非新增
    msgs.append(_assistant_with_tools("c3"))
    msgs.append({"role": "tool", "name": "rag", "content": "z" * 4000})
    report2 = await enforce(msgs, "qwen-plus", plan=None)
    assert _ck_count(msgs) == 1                  # 不累积（旧 L2 插入语义的病灶）
    assert "第1轮摘要" in fold_inputs[1]           # 二次折叠输入含旧 checkpoint 正文（续写）
    assert report2.checkpoint_idx >= 0
    _assert_tool_pairing(msgs)


@pytest.mark.asyncio
async def test_enforce_fold_input_covers_assistant_text(monkeypatch):
    """D3 优先级倒挂修复：assistant 中间结论进折叠摘要输入（旧 L3 裸丢的恰是它）。"""
    fold_inputs: list[str] = []
    _enforce_env(monkeypatch, summarize=lambda t: fold_inputs.append(t) or "摘要")

    msgs = _foldable_msgs(aside="我先查导数定义，再对比两种定义的适用范围。")
    await enforce(msgs, "qwen-plus", plan=None)
    assert "我先查导数定义" in fold_inputs[0]   # assistant 中间结论进摘要输入


@pytest.mark.asyncio
async def test_enforce_shrink_loop_converges_under_hard(monkeypatch):
    """③收缩循环：超 hard_ceiling 时逐级降级（截长 tool 结果 -> 丢最旧轮），构造性
    收敛到 hard 以下，且配对不变式保持。"""
    _enforce_env(monkeypatch, soft=50, hard=2000, summarize="")  # "" = 折叠降级，逼 ③ 收缩

    msgs: list[dict] = [{"role": "system", "content": "s"}]
    for i in range(5):
        msgs.append({"role": "user", "content": f"u{i} " + "x" * 3000})
        msgs.append(_assistant_with_tools(f"c{i}"))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "rag",
                     "content": "z" * 3000})
    report = await enforce(msgs, "qwen-plus", plan=None)
    assert report.tokens_after <= 2000          # 构造性收敛
    assert report.dropped_messages > 0          # 丢轮发生
    assert len(report.dropped_originals) >= report.dropped_messages
    assert report.dropped_originals[0].startswith("[")
    _assert_tool_pairing(msgs)


@pytest.mark.asyncio
async def test_enforce_shrink_exhausted_raises(monkeypatch):
    """收缩耗尽仍超 hard -> 显式 raise（fail-fast，不静默发注定 413 的请求）。"""
    _enforce_env(monkeypatch, soft=50, hard=100)
    # 前导只有巨型 system：①②③皆无动作（无 tool 轮可清/折/丢，system 不在 ③c 范围）
    msgs = [{"role": "system", "content": "s" * 100000}]
    with pytest.raises(RuntimeError, match="context shrink exhausted"):
        await enforce(msgs, "qwen-plus", plan=None)


@pytest.mark.asyncio
async def test_enforce_answer_now_shallow_copy_semantics(monkeypatch):
    """answer_now 副本语义：折叠删消息只影响副本列表（主列表长度不变），清墓碑经
    共享 dict 同步可见（无害且幂等）。"""
    _enforce_env(monkeypatch, summarize="摘要")

    msgs = _foldable_msgs()
    tool_msg = msgs[3]
    ans_messages = list(msgs) + [{"role": "system", "content": "用户请求立即回答…"}]
    report = await enforce(ans_messages, "qwen-plus", plan=None)
    assert report.summary_added is True
    assert len(msgs) == 6  # 折叠只删副本，主列表长度不变
    assert str(tool_msg["content"]).startswith("[cleared:")  # 墓碑经共享 dict 同步可见
    _assert_tool_pairing(ans_messages)


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
