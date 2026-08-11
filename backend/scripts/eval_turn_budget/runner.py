"""单 case 单臂执行器：走完整 turn_runtime.start_turn 链路，全量落盘。

为什么不复用 eval_context/ablation_runner：那条路直调 ``run_agent_loop`` 并用 ``set_arm`` 切臂，
``set_arm`` 会让 loop 走 ``arm is not None`` 分支、**短路 coordinator 分支**（loop.py:491-497），
测不到 coordinator_enabled 这条线。本 runner 一律不调 set_arm，改临时覆写 settings 三/四开关，
让 loop 自然回落到 coordinator / policy 分支。

为什么不复用 eval_capabilities/solver：solver 直调 ``orchestrator.handle``，**跳过
turn_runtime.start_turn**，而两臂的回合前分歧点（plan_turn vs ContextBuilder）就在 start_turn 里。
本 runner 必须经 start_turn，才能测到回合前历史裁剪。

执行流程（对齐 api/chat.py 的 ctx 构造 + 生产 SSE 订阅路径）：
  临时覆写 settings -> 构造 UnifiedContext -> start_turn(内部跑 plan_turn/ContextBuilder)
  -> subscribe_turn 收全量事件至 done/error -> 读 ctx.metadata(就地改写) -> 组装记录
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from . import config

logger = logging.getLogger(__name__)

# 单 turn 总超时（防 ask_user 等异常挂起；chat 模式无 ask_user，正常远低于此）。env 可调。
_TURN_TIMEOUT = float(__import__("os").getenv("EVAL_TURN_BUDGET_TURN_TIMEOUT", "180"))


def _patch_settings(arm: dict) -> dict:
    """临时覆写 settings 四开关，返回原值快照供 finally 复原。

    全局单例 ``get_settings()`` 被就地改写；评测必须串行跑（不能并行改同一单例）。
    TruthyBool 字段直接赋 bool 即可（``if`` 求值取其真假）。
    """
    from settings import get_settings
    s = get_settings()
    orig = {
        "coordinator_enabled": s.context_budget.coordinator_enabled,
        "eviction_strategy": s.context_budget.eviction_strategy,
        "carry_forward_location": s.context_budget.carry_forward_location,
        "policy_enabled": s.context_policy.enabled,
    }
    s.context_budget.coordinator_enabled = arm["coordinator_enabled"]
    s.context_budget.eviction_strategy = arm["eviction_strategy"]
    s.context_budget.carry_forward_location = arm["carry_forward_location"]
    s.context_policy.enabled = arm["policy_enabled"]
    return orig


def _restore_settings(orig: dict) -> None:
    from settings import get_settings
    s = get_settings()
    s.context_budget.coordinator_enabled = orig["coordinator_enabled"]
    s.context_budget.eviction_strategy = orig["eviction_strategy"]
    s.context_budget.carry_forward_location = orig["carry_forward_location"]
    s.context_policy.enabled = orig["policy_enabled"]


async def _collect_events(trm, turn_id: str, t0: float) -> tuple[list[dict], int | None, str]:
    """订阅 turn 全量事件至 bus 关闭（turn 结束自然终止）。

    返回 (events[to_dict], first_event_ms, error)。turn 是后台 task，异常被 _run_turn 吞成
    ERROR 事件而非抛出，故在此显式检测 ERROR 记进 error（避免跑挂的 case 当正常样本）。
    """
    from core.stream import StreamEventType
    events: list[dict] = []
    first_event_ms: int | None = None
    error = ""
    async for event in trm.subscribe_turn(turn_id):
        if first_event_ms is None:
            first_event_ms = int((time.perf_counter() - t0) * 1000)
        ed = event.to_dict()
        events.append(ed)
        if event.type == StreamEventType.ERROR:
            error = str(ed.get("content") or ed.get("message") or ed.get("error") or "")
    return events, first_event_ms, error


def _build_record(
    case: dict, arm: dict, ctx, events: list[dict],
    history_before: int, history_after: int,
    first_event_ms: int | None, total_elapsed_ms: int, error: str,
) -> dict[str, Any]:
    """从 events + ctx.metadata 组装单条完整记录（对齐 plan §4 字段表）。"""
    from core.stream import StreamEventType

    # 过程侧：按事件类型拆分
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    rounds = 0
    for ed in events:
        t = ed.get("type")
        if t == StreamEventType.TOOL_CALL.value:
            tool_calls.append({"tool": ed.get("tool"), "input": ed.get("input")})
        elif t == StreamEventType.TOOL_RESULT.value:
            tool_results.append({"tool": ed.get("tool"), "content": ed.get("content")})
        elif t in (StreamEventType.THINKING.value, StreamEventType.THINKING_CHUNK.value):
            c = ed.get("content")
            if c:
                thinking_parts.append(str(c))
        elif t in (StreamEventType.ANSWER.value, StreamEventType.TOKEN.value):
            c = ed.get("content")
            if c:
                answer_parts.append(str(c))
        elif t == StreamEventType.DONE.value:
            # done.metadata.iterations = loop 实际跑的轮数（权威 rounds 来源）
            rounds = int((ed.get("metadata") or {}).get("iterations") or 0)

    answer = "".join(answer_parts)

    # 成本侧：loop 写入 ctx.metadata（done 之前就写好，subscribe 结束时已就绪）
    usage = ctx.metadata.get("llm_usage") or {}
    cost_usd = ctx.metadata.get("llm_cost_usd")

    # 压缩侧：两臂互斥键（coordinator->_cb_cleared_tool_results；policy->_cp_masked_turns）
    # extra_llm_calls 仅 set_arm 路径累加（本 runner 不走），恒为 0，保留字段便于对照
    # budget_plan：TurnBudgetPlan dataclass（policy 臂无 _budget_plan -> None）；
    # getattr(None, ..., None) 兜底返 None，无需显式 None 守卫/中间字典
    budget_plan = ctx.metadata.get("_budget_plan")

    return {
        # 输入侧
        "case_id": case.get("id"),
        "arm": arm["label"],
        "question": case.get("question"),
        "history_before": case.get("history", []),  # 裁剪前原文（便于复盘两臂裁了什么）
        "course_id": case.get("course_id"),
        "rag_mode": case.get("rag_mode"),
        # 裁剪侧
        "history_before_count": history_before,
        "history_after_count": history_after,
        "dropped_count": getattr(budget_plan, "dropped_count", None),
        "carry_forward_added": getattr(budget_plan, "carry_forward_added", None),
        "mecw": getattr(budget_plan, "mecw", None),
        "target_input_tokens": getattr(budget_plan, "target_input_tokens", None),
        # 过程侧
        "events": events,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "thinking": "\n".join(thinking_parts),
        "rounds": rounds,
        # 输出侧
        "answer": answer,
        "answer_chars": len(answer),
        "error": error or None,
        # 成本侧
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_tokens", 0),
        "cost_usd": cost_usd,
        # 压缩侧
        "cleared_tool_results": ctx.metadata.get("_cb_cleared_tool_results", 0),
        "masked_turns": ctx.metadata.get("_cp_masked_turns", 0),
        "extra_llm_calls": ctx.metadata.get("_cp_extra_llm_calls", 0),
        # 时延侧
        "total_elapsed_ms": total_elapsed_ms,
        "first_event_ms": first_event_ms,
    }


async def run_case(case: dict, arm: dict) -> dict[str, Any]:
    """单 case 单臂：patch settings -> start_turn -> subscribe -> 组装记录。

    settings 覆写在 finally 复原；history 深拷避免两臂共用同一 case 时被就地裁剪污染。
    """
    from core.context import UnifiedContext
    from services.session.turn_runtime import get_turn_runtime_manager

    orig = _patch_settings(arm)
    try:
        # 深拷 history：start_turn 会把 ctx.conversation_history 指向裁剪后的新列表，
        # 虽不改原 list，但 message dict 可能被下游改--拷贝保两臂独立。
        history = [dict(m) for m in case.get("history", [])]
        ctx = UnifiedContext(
            course_id=case.get("course_id") or "general",
            user_id=config.EVAL_USER_ID,
            user_message=case.get("question", ""),
            conversation_history=history,
            session_summary=case.get("session_summary", ""),
            mode=case.get("mode", "chat"),
            enabled_tools=case.get("enabled_tools", ["rag"]),
            rag_mode=case.get("rag_mode", "naive"),
        )
        history_before = len(history)

        trm = get_turn_runtime_manager()
        t0 = time.perf_counter()
        turn_id = await trm.start_turn(ctx)
        # start_turn 同步完成 plan_turn/ContextBuilder（在 create_task 之前），返回时 history 已裁好
        history_after = len(ctx.conversation_history)

        try:
            events, first_event_ms, error = await asyncio.wait_for(
                _collect_events(trm, turn_id, t0), timeout=_TURN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[%s/%s] turn 超时 %ss，取消", case.get("id"), arm["label"], _TURN_TIMEOUT)
            await trm.cancel_turn(turn_id)
            events, first_event_ms, error = [], None, f"turn timeout ({_TURN_TIMEOUT}s)"
        finally:
            # turn 已自然结束则 cancel 是 no-op；超时则真正取消
            await trm.cancel_turn(turn_id)

        total_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        rec = _build_record(
            case, arm, ctx, events, history_before, history_after,
            first_event_ms, total_elapsed_ms, error,
        )
        logger.info(
            "[%s/%s] rounds=%s in=%s out=%s cost=%.6f cleared=%s masked=%s "
            "dropped=%s carry=%s elapsed=%sms ttft=%sms ans=%s",
            case.get("id"), arm["label"], rec["rounds"],
            rec["input_tokens"], rec["output_tokens"], rec["cost_usd"] or 0,
            rec["cleared_tool_results"], rec["masked_turns"],
            rec["dropped_count"], rec["carry_forward_added"],
            rec["total_elapsed_ms"], rec["first_event_ms"], rec["answer_chars"],
        )
        return rec
    finally:
        _restore_settings(orig)


__all__ = ["run_case"]
