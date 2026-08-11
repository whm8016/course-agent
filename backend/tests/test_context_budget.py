"""通用上下文管理：纯函数测试（token 记账 + cache_control T1 偏移 + T2 集体裁剪）。

覆盖：
- token_count_slices：逐切片计数、跳空切片
- system_t1_chars：T1 前缀字符偏移=拼装前缀长度、空段过滤、是拼装字符串的精确切点
- cap_dynamic_slices：超预算按优先级低->高裁、T1 永不裁、达标即停、不改入参

（mecw_for 已随百分比系数移除，其测试迁至 test_context_window.py 的 compute_budgets。）
"""
from __future__ import annotations

from services.session.context_builder import count_tokens
from core.agentic.context_budget import (
    cap_dynamic_slices,
    system_t1_chars,
    token_count_slices,
)


# ---------------------------------------------------------------------------
# token_count_slices
# ---------------------------------------------------------------------------

def test_token_count_slices_per_slice_count():
    slices = {"a": "你好世界", "b": "hello world", "c": ""}
    got = token_count_slices(slices)
    assert "c" not in got  # 空切片不计
    # 逐切片计数=该切片单独计数（注意：拼装后 "\n\n" 分隔符与 BPE 边界会额外增 token，
    # 故「逐切片之和」≠「拼装整段计数」，前者是预算估算口径，略低于实际整段）
    assert got["a"] == count_tokens("你好世界")
    assert got["b"] == count_tokens("hello world")


def test_token_count_slices_all_empty():
    assert token_count_slices({"a": "", "b": ""}) == {}


# ---------------------------------------------------------------------------
# system_t1_chars
# ---------------------------------------------------------------------------

def test_system_t1_chars_filters_empty_and_joins():
    t1 = ["loop", "course", "", "skills"]  # 中间空段过滤
    assert system_t1_chars(t1) == len("loop\n\ncourse\n\nskills")


def test_system_t1_chars_is_exact_offset_into_assembled_prompt():
    # 与 assemble_system_prompt 同序同过滤：T1 偏移须精确切齐拼装字符串
    t1_slices = ["AAA", "BBB"]
    t2_slices = ["CCC", "DDD"]
    full = "\n\n".join(t1_slices + t2_slices)
    off = system_t1_chars(t1_slices)
    assert full[:off] == "\n\n".join(t1_slices)
    assert full[off:].lstrip() == "\n\n".join(t2_slices)


def test_system_t1_chars_all_empty_is_zero():
    assert system_t1_chars(["", ""]) == 0


# ---------------------------------------------------------------------------
# cap_dynamic_slices
# ---------------------------------------------------------------------------

def test_cap_dynamic_slices_trims_low_priority_first():
    # memory 巨大（最低优先级），mastery/summary 小；预算只够 T1+mastery+summary
    slices = {
        "loop_system": "x",
        "memory": "y" * 1000,      # 最低优先级，应先被裁
        "mastery": "z",
        "session_summary": "w",    # 最高优先级，应保留
    }
    tier1 = frozenset({"loop_system"})
    priority = ["memory", "mastery", "session_summary"]  # 低 -> 高
    budget = count_tokens(slices["loop_system"]) + count_tokens(slices["mastery"]) + count_tokens(slices["session_summary"])
    out = cap_dynamic_slices(slices, budget, tier1, priority)
    assert out["loop_system"] == slices["loop_system"]      # T1 不动
    assert out["memory"] == ""                               # 最低优先级被裁
    assert out["mastery"] == slices["mastery"]               # 达标即停，保留
    assert out["session_summary"] == slices["session_summary"]


def test_cap_dynamic_slices_preserves_tier1_even_when_over_budget():
    slices = {"loop_system": "稳定" * 100, "memory": "记忆"}
    out = cap_dynamic_slices(slices, 0, frozenset({"loop_system"}), ["memory"])
    assert out["loop_system"] == slices["loop_system"]  # T1 永不裁
    assert out["memory"] == ""


def test_cap_dynamic_slices_noop_when_under_budget():
    slices = {"loop_system": "a", "memory": "b"}
    out = cap_dynamic_slices(slices, 1000, frozenset({"loop_system"}), ["memory"])
    assert out == slices


def test_cap_dynamic_slices_does_not_mutate_input():
    slices = {"loop_system": "a", "memory": "b" * 200}
    orig = dict(slices)
    cap_dynamic_slices(slices, 0, frozenset({"loop_system"}), ["memory"])
    assert slices == orig  # 入参未被改
