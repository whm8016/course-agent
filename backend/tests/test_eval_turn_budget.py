"""eval_turn_budget runner 纯逻辑单测（不烧 LLM、不依赖 DB/课程索引）。

run_case 的端到端（start_turn -> 真 turn -> LLM）需真机跑（LLM key + DB + 已索引课程），
本测试只覆盖 run_case 内部的两个纯函数：
  - _patch_settings / _restore_settings：四开关覆写 + 复原（全局单例，须保证复原）
  - _build_record：从 events + ctx.metadata 组装完整记录的字段抽取
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# 让 import 找到 backend 根 + scripts 包
_BACKEND_ROOT = str(Path(__file__).resolve().parents[1])
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from core.agentic.context_budget import TurnBudgetPlan  # noqa: E402
from core.context import UnifiedContext  # noqa: E402
from core.stream import StreamEventType  # noqa: E402
from scripts.eval_turn_budget import config  # noqa: E402
from scripts.eval_turn_budget.runner import _build_record, _patch_settings, _restore_settings  # noqa: E402
from settings import get_settings  # noqa: E402


def _arm(label: str) -> dict:
    return next(a for a in config.ARMS if a["label"] == label)


def test_patch_restore_settings_roundtrip():
    """四开关被正确覆写，复原后回到原值（全局单例不能被测试污染）。"""
    s = get_settings()
    orig_coordinator = s.context_budget.coordinator_enabled
    orig_eviction = s.context_budget.eviction_strategy
    orig_carry = s.context_budget.carry_forward_location
    orig_policy = s.context_policy.enabled

    arm = _arm("coordinator_priority")
    orig = _patch_settings(arm)
    try:
        assert s.context_budget.coordinator_enabled is True
        assert s.context_budget.eviction_strategy == "priority"
        assert s.context_budget.carry_forward_location == "history_prefix"
        assert s.context_policy.enabled is True
    finally:
        _restore_settings(orig)

    # 复原：回到原值（生产默认 coordinator_enabled=False）
    assert s.context_budget.coordinator_enabled == orig_coordinator
    assert s.context_budget.eviction_strategy == orig_eviction
    assert s.context_budget.carry_forward_location == orig_carry
    assert s.context_policy.enabled == orig_policy


def test_patch_restore_policy_arm():
    """policy_default 臂：coordinator 关、policy 开。"""
    s = get_settings()
    orig = _patch_settings(_arm("policy_default"))
    try:
        assert s.context_budget.coordinator_enabled is False
        assert s.context_policy.enabled is True
        assert s.context_budget.eviction_strategy == "mask"
    finally:
        _restore_settings(orig)


def _evt(event_type: StreamEventType, **payload) -> dict:
    d = {"type": event_type.value}
    d.update(payload)
    return d


def test_build_record_extracts_all_fields_coordinator_arm():
    """coordinator 臂：_budget_plan 有值，cleared_tool_results 累加，masked_turns=0。"""
    ctx = UnifiedContext(user_message="q", course_id="c")
    ctx.metadata["_budget_plan"] = TurnBudgetPlan(
        keep_history=[], dropped_count=3, carry_forward_added=True,
        target_input_tokens=9830, mecw=16384, hard_ceiling=20000,
    )
    ctx.metadata["llm_usage"] = {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 500}
    ctx.metadata["llm_cost_usd"] = 0.0123
    ctx.metadata["_cb_cleared_tool_results"] = 2  # coordinator 臂的压缩键

    events = [
        _evt(StreamEventType.THINKING, content="分析中"),
        _evt(StreamEventType.TOOL_CALL, tool="rag", input={"query": "KCL"}),
        _evt(StreamEventType.TOOL_RESULT, tool="rag", content="基尔霍夫电流定律..."),
        _evt(StreamEventType.TOKEN, content="答"),
        _evt(StreamEventType.ANSWER, content="案"),
        _evt(StreamEventType.DONE, metadata={"iterations": 4}),
    ]
    arm = _arm("coordinator_priority")
    case = {"id": "tb-01", "question": "q", "course_id": "c", "rag_mode": "naive", "history": []}
    rec = _build_record(case, arm, ctx, events, 24, 21, 120, 3500, "")

    # 裁剪侧：budget_plan 字段透传
    assert rec["dropped_count"] == 3
    assert rec["carry_forward_added"] is True
    assert rec["mecw"] == 16384
    assert rec["target_input_tokens"] == 9830
    assert rec["history_before_count"] == 24
    assert rec["history_after_count"] == 21
    # 过程侧
    assert rec["rounds"] == 4
    assert rec["tool_calls"] == [{"tool": "rag", "input": {"query": "KCL"}}]
    assert rec["tool_results"] == [{"tool": "rag", "content": "基尔霍夫电流定律..."}]
    assert "分析中" in rec["thinking"]
    assert rec["answer"] == "答案"
    assert rec["answer_chars"] == 2
    assert rec["events"] == events
    # 成本侧
    assert rec["input_tokens"] == 1000
    assert rec["output_tokens"] == 200
    assert rec["cache_read_tokens"] == 500
    assert rec["cost_usd"] == 0.0123
    # 压缩侧：coordinator 臂累加 cleared，masked 恒 0
    assert rec["cleared_tool_results"] == 2
    assert rec["masked_turns"] == 0
    # 时延侧
    assert rec["first_event_ms"] == 120
    assert rec["total_elapsed_ms"] == 3500
    assert rec["error"] is None


def test_build_record_policy_arm_no_budget_plan():
    """policy 臂：无 _budget_plan（dropped/carry/mecw/target 全 None），masked_turns 累加。"""
    ctx = UnifiedContext(user_message="q", course_id="c")
    ctx.metadata["llm_usage"] = {"input_tokens": 800, "output_tokens": 150, "cache_read_tokens": 0}
    ctx.metadata["_cp_masked_turns"] = 5  # policy 臂的压缩键
    # 无 _budget_plan、无 _cb_cleared_tool_results、无 llm_cost_usd

    events = [_evt(StreamEventType.ANSWER, content="回复"), _evt(StreamEventType.DONE, metadata={"iterations": 2})]
    rec = _build_record({"id": "tb-02", "question": "q", "course_id": "c", "rag_mode": "naive", "history": []},
                        _arm("policy_default"), ctx, events, 24, 24, 80, 2000, "")

    # 裁剪侧：policy 臂无 budget_plan -> None
    assert rec["dropped_count"] is None
    assert rec["carry_forward_added"] is None
    assert rec["mecw"] is None
    assert rec["target_input_tokens"] is None
    # 压缩侧：policy 臂累加 masked，cleared 恒 0
    assert rec["masked_turns"] == 5
    assert rec["cleared_tool_results"] == 0
    # cost_usd 缺失 -> None
    assert rec["cost_usd"] is None
    assert rec["answer"] == "回复"


def test_build_record_error_event_recorded():
    """turn 崩溃吐 ERROR 事件 -> error 字段记下，避免当正常样本。"""
    ctx = UnifiedContext(user_message="q", course_id="c")
    events = [_evt(StreamEventType.ERROR, message="LLM 服务异常")]
    rec = _build_record({"id": "tb-03", "question": "q", "course_id": "c", "rag_mode": "naive", "history": []},
                        _arm("policy_default"), ctx, events, 24, 24, None, 500, "LLM 服务异常")
    assert rec["error"] == "LLM 服务异常"
    assert rec["rounds"] == 0
    assert rec["answer"] == ""


def test_build_record_with_namespace_budget_plan():
    """budget_plan 非 dataclass（如 SimpleNamespace）也能 getattr 取字段。"""
    ctx = UnifiedContext(user_message="q", course_id="c")
    ctx.metadata["_budget_plan"] = SimpleNamespace(
        dropped_count=1, carry_forward_added=False, mecw=16384, target_input_tokens=9830,
    )
    events = [_evt(StreamEventType.DONE, metadata={"iterations": 1})]
    rec = _build_record({"id": "tb-04", "question": "q", "course_id": "c", "rag_mode": "naive", "history": []},
                        _arm("coordinator_priority"), ctx, events, 24, 23, 50, 1000, "")
    assert rec["dropped_count"] == 1
    assert rec["carry_forward_added"] is False
