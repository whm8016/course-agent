"""通用上下文管理：双阈值预算 + 三级级联执行器 + 反应式回合前裁剪。

复用 context_builder.count_tokens（tiktoken cl100k_base + len//4 兜底）做 token 记账。
窗口解析与双阈值预算在 context_window.py（resolve_effective_window 三级解析 +
compute_budgets 减法留白）。本模块提供：

- token_count_slices   逐切片 token 计数（system prompt 预算分解 + 超阈值告警）
- system_t1_chars      system prompt 稳定前缀（T1）的字符长度，供 cache_control 在 T1/T2 边界放断点
- cap_dynamic_slices   T2 动态层集体裁剪（按优先级整段丢弃最低优先级切片，直到回到预算内）
- compute_budgets      （转发 context_window）(soft_trigger, hard_ceiling) 双阈值
- enforce              轮内三级级联执行器：L1 清旧 tool 结果(免费)->L2 LLM 摘要->L3 丢最旧 20% 消息组
- plan_turn            回合前反应式裁剪：未超软阈值原样返回，超了才裁历史 + 摘要续接

分层（驱逐优先级从高到低）：
  T1 静态锚点（loop_system / course_prompt / skills / tool_hint…）永不裁、逐字稳定保 prefix cache；
  T2 易变层（memory / mastery / summary / now）承压时按优先级裁（KB-seed->memory->mastery->summary）。

调研依据：Claude Code 五级压缩级联（永远先用最轻的干预）+ Anthropic Context Management API
（compact 150k / tool clearing 100k keep=3 + exclude_tools）+ context rot 论文 arXiv:2606.29718
（compaction+trimming 组合最优）+ arXiv:2508.21433（Observation Masking 成本减半）+ Anthropic
Prompt Caching（cache_control read 0.1x）。
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from core.agentic.context_window import compute_budgets, count_tokens, resolve_effective_window
from services.session.context_builder import ContextBuilder
from settings import get_settings


def token_count_slices(slices: dict[str, str]) -> dict[str, int]:
    """逐切片 token 计数（空切片不计）。用于 system prompt 预算分解与超阈值告警。"""
    return {k: count_tokens(v) for k, v in slices.items() if v}


def system_t1_chars(t1_slices: list[str]) -> int:
    """system prompt 稳定前缀（T1）的字符长度。

    t1_slices 须与 assemble_system_prompt 的 T1 段「同序、同空段过滤」——返回值即 T1 前缀在拼装后
    system_prompt 字符串里的精确字符偏移。_build_messages 据此切 system_prompt[:n] 为 T1、
    system_prompt[n:] 为 T2，供 anthropic_adapter 在 T1 块放 cache_control 断点。用字符不用 token：
    拼装是按字符 "\\n\\n".join，字符偏移能精确切齐（token 边界会错位）。
    """
    return len("\n\n".join(p for p in t1_slices if p))


def cap_dynamic_slices(
    slices: dict[str, str],
    budget_tokens: int,
    tier1_keys: frozenset[str],
    priority_low_to_high: list[str],
) -> dict[str, str]:
    """T2 动态层集体裁剪：超预算时按优先级（低->高）整段丢弃最低优先级切片，直到回到预算内。

    tier1_keys 永不裁（逐字保留保 prefix cache）。priority_low_to_high 给 T2 切片的裁剪顺序
    （如 ["extra_context","memory_context","mastery_context","session_summary"]——KB-seed 最易重建
    先裁、L2 摘要不可重建最后裁）。返回新 dict（不改入参）。pipeline_common 在装配前调用。
    """
    out = {k: v for k, v in slices.items()}
    t2_keys = [k for k in priority_low_to_high if k in out and k not in tier1_keys]

    def _t2_tokens() -> int:
        return sum(count_tokens(out[k]) for k in t2_keys if out.get(k))

    for k in t2_keys:  # 低优先级在前，逐个丢弃直到达标
        if _t2_tokens() <= budget_tokens:
            break
        if out.get(k):
            out[k] = ""
    return out


# ---------------------------------------------------------------------------
# 协调器骨架（plan_turn 回合前反应式 + enforce 轮内三级级联）
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TurnBudgetPlan:
    """回合前预算规划产物，随 context.metadata["_budget_plan"] 跨调用传递。

    keep_history        裁剪后历史（carry_forward 时首位可能前插 <session_summary> system 消息）
    dropped_count       被裁掉的历史消息数
    carry_forward_added 是否前插了摘要续接块（是则 turn_runtime 清空 ctx.session_summary 避免双付）
    target_input_tokens 软阈值/质量线（= compute_budgets 的 soft_trigger；超了才裁）。eval getattr 兼容字段
    mecw                有效上下文窗口（= resolve_effective_window；三级解析）。eval getattr 兼容字段
    hard_ceiling        硬天花板/安全线（L3 降级与反应式兜底用）
    """

    keep_history: list[dict[str, Any]]
    dropped_count: int
    carry_forward_added: bool
    target_input_tokens: int
    mecw: int
    hard_ceiling: int


@dataclasses.dataclass
class EnforceReport:
    """轮内三级级联执行报告。"""

    tokens_before: int
    tokens_after: int
    cleared_tool_results: int   # L1 清理的 tool 结果条数
    summary_added: bool         # L2 是否做了 LLM 摘要续接
    dropped_messages: int       # L3 丢弃的最旧消息条数


def _messages_tokens(messages: list[dict[str, Any]]) -> int:
    """messages 列表的总 token 估算（content + tool_calls 序列化）。

    assistant 工具调用轮常是 content="" + 一大坨 tool_calls，旧实现只计 content ->
    软阈值判断偏乐观（实际已超窗仍判未超）。tool_calls 按发给 API 的 JSON 序列化计入。
    """
    total = 0
    for m in messages:
        total += count_tokens(str(m.get("content", "")))
        tcs = m.get("tool_calls")
        if tcs:
            total += count_tokens(json.dumps(tcs, ensure_ascii=False))
    return total


# -- L1：优先级清理 tool 结果（免费，无 LLM）------------------------------------

_CLEAR_TOMBSTONE = "[cleared: {name}]"


def _tool_result_priority(msg: dict[str, Any]) -> int:
    """单条 role=tool 结果的保留优先级（高=留，低=先驱逐）。

    带来源引用段 = +2（RAG 引用高价值，复用 context_policy._is_source_para/_split_paragraphs 判定）；
    空 = -3（无信息，但 0 token，evict_tool_results 直接跳过）；其余 = 0。
    ask_user 用户答复不在本函数打分——它由 evict_tool_results 的 exclude_tools 白名单直接排除。
    """
    content = str(msg.get("content") or "")
    if not content.strip():
        return -3
    from core.agentic.context_policy import _is_source_para, _split_paragraphs

    if any(_is_source_para(p) for p in _split_paragraphs(content)):
        return 2
    return 0


def evict_tool_results(
    messages: list[dict[str, Any]],
    keep_recent_turns: int,
    target_tokens: int,
    *,
    exclude_tools: list[str] | None = None,
    captured_originals: list[str] | None = None,
) -> int:
    """L1 优先级清理：总 token 超 target 时，按优先级（低->高）把 older 轮的 role=tool 结果清理为墓碑。

    保留最近 keep_recent_turns 轮不动（含本轮 live 工具结果）；exclude_tools 白名单（默认
    settings.context_budget.exclude_tools，对齐 Anthropic clear_tool_uses 语义，保护 ask_user 等
    不可重建输入）永不清理。复用 context_policy._split_turns 切轮。墓碑 ``[cleared: <tool>]`` ~15 字
    vs 原文数千字（Claude Code「最轻 compaction=旧工具结果清理」）。

    captured_originals 非空时，把被清理 tool 原文 append 进去，供 L2 摘要复用（避免 L1 清掉后 L2
    无原文可摘要）。返回被清理条数。就地改 messages。
    """
    total = _messages_tokens(messages)
    if total <= target_tokens:
        return 0
    from core.agentic.context_policy import _split_turns

    # 白名单：调用方显式传入优先；否则回落 settings.context_budget.exclude_tools（默认 ["ask_user"]）
    if exclude_tools is None:
        exclude_tools = get_settings().context_budget.exclude_tools
    exclude_lower = {(n or "").lower() for n in exclude_tools}

    turns = _split_turns(messages)
    if len(turns) <= 1:
        return 0
    older = turns[:-keep_recent_turns] if keep_recent_turns > 0 else turns
    # 收集 older 轮的可清理 tool 结果：白名单永不清理；空/已清理墓碑 跳过（清理省 0 token）
    candidates: list[tuple[int, dict[str, Any], int]] = []
    for turn in older:
        for msg in turn:
            if msg.get("role") != "tool":
                continue
            if (msg.get("name") or "").lower() in exclude_lower:
                continue  # 白名单 tool 永不清理（如 ask_user 用户答复）
            c = str(msg.get("content") or "")
            if not c or c.startswith("[cleared:"):
                continue
            candidates.append((_tool_result_priority(msg), msg, count_tokens(c)))
    candidates.sort(key=lambda x: x[0])  # 低优先级在前，先清理
    cleared = 0
    for _score, msg, ct in candidates:
        if total <= target_tokens:
            break
        if captured_originals is not None:
            captured_originals.append(str(msg.get("content") or ""))
        tomb = _CLEAR_TOMBSTONE.format(name=(msg.get("name") or "tool"))
        msg["content"] = tomb
        total -= ct - count_tokens(tomb)
        cleared += 1
    return cleared


# -- L3：丢最旧 20% 消息组（免费硬降级）-----------------------------------------

def drop_oldest_turn_group(messages: list[dict[str, Any]], *, fraction: float = 0.2) -> int:
    """L3 硬降级：丢弃最旧的 fraction（默认 20%）消息组（按 turn 切），保证请求一定发得出去。

    对齐 Claude Code PTL retry 策略（丢最旧 20%）。保留首轮（system + 首个 user 前导，_split_turns
    把它们归 turns[0]）不动——只丢最旧的 tool 轮。就地改 messages（用 survivors 覆盖）。返回丢弃条数。
    """
    from core.agentic.context_policy import _split_turns

    turns = _split_turns(messages)
    if len(turns) <= 2:  # 仅 1 个 tool 轮 + 前导，无余量可丢
        return 0
    # 可丢轮 = 排除前导 turns[0] 的 tool 轮；丢最旧的 fraction
    droppable = len(turns) - 1
    drop_n = max(1, int(droppable * fraction))
    drop_n = min(drop_n, droppable)  # 不超过可丢数
    survivors = turns[:1] + turns[1 + drop_n:]
    dropped = sum(len(t) for t in turns[1:1 + drop_n])
    messages[:] = [m for t in survivors for m in t]
    return dropped


async def enforce(
    messages: list[dict[str, Any]],
    model: str,
    plan: TurnBudgetPlan | None = None,
) -> EnforceReport:
    """轮内三级级联执行器（loop 每轮调一次）。永远先用最便宜的手段，只有不够才升级。

    L1 清旧 tool 结果（免费）：总 token 超 soft_trigger 时按优先级清理 older 轮 tool 结果为墓碑，
        保留最近 keep_recent_turns 轮，exclude_tools 白名单（ask_user）永不清理。
    L2 LLM 摘要（一次调用）：L1 后仍超 soft_trigger 才触发，把 L1 清掉的原文压成结构化摘要前插。
    L3 硬降级（免费）：仍超 hard_ceiling 时丢弃最旧 20% 消息组，保证请求一定发得出去。

    plan 当前未用（回合前 plan_turn 已做反应式裁剪），保留参数供后续 per-turn 策略。
    """
    cfg = get_settings().context_budget
    soft_trigger, hard_ceiling = compute_budgets(model)
    before = _messages_tokens(messages)
    total = before
    cleared = 0
    summary_added = False
    dropped = 0

    # L1：清旧 tool 结果（免费）。捕获被清原文供 L2 摘要复用。
    captured: list[str] = []
    if total > soft_trigger:
        cleared = evict_tool_results(
            messages, cfg.keep_recent_turns, soft_trigger,
            exclude_tools=cfg.exclude_tools, captured_originals=captured,
        )
        total = _messages_tokens(messages)

    # L2：LLM 摘要续接（仍超软阈值）。把 L1 清掉的原文压成摘要前插为 system 消息（carry-forward 式）。
    if total > soft_trigger and captured:
        from core.agentic.context_policy import _summarize_masked_text
        summary = await _summarize_masked_text("\n\n".join(captured)[:20000], model)
        if summary:
            # 前插位置：首个 system 消息之后（不破坏 system-first 不变式，模型仍先读 system）
            _insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
            messages.insert(_insert_idx, {
                "role": "system",
                "content": f"<context_summary>\n{summary}\n</context_summary>",
            })
            summary_added = True
            total = _messages_tokens(messages)

    # L3：硬降级（仍超硬天花板）。丢最旧 20% 消息组，保请求发得出去。
    if total > hard_ceiling:
        dropped = drop_oldest_turn_group(messages)
        total = _messages_tokens(messages)

    return EnforceReport(
        tokens_before=before,
        tokens_after=total,
        cleared_tool_results=cleared,
        summary_added=summary_added,
        dropped_messages=dropped,
    )


# -- 回合前反应式裁剪（plan_turn）-----------------------------------------------

def plan_turn(
    *,
    history: list[dict[str, Any]],
    model: str,
    session_summary_text: str,
) -> TurnBudgetPlan:
    """回合前预算规划：反应式——未超软阈值直接原样返回（不裁），超了才裁历史 + 摘要续接。

    反应式语义（对标 Claude Code「未超阈值不干预」）：history token <= soft_trigger 时零干预原样
    返回，避免无谓裁剪丢失上下文。超 soft_trigger 才裁：
      - 有摘要兜底（carry_forward_location="history_prefix" 且 session_summary_text 非空）：收紧裁到
        soft_trigger 的一半（丢的有摘要补），裁掉消息时把摘要前插为 <session_summary> system 消息
        续接（Claude Code 式 carry-forward），turn_runtime 据此清空 ctx.session_summary 避免 system 双付。
      - 无摘要兜底：裁到 soft_trigger（no-loss 倾向——不丢未覆盖消息，只裁到预算内）。
    """
    cfg = get_settings().context_budget
    soft_trigger, hard_ceiling = compute_budgets(model)
    eff_window = resolve_effective_window(model)

    # 反应式：未超软阈值直接原样返回（不裁，零信息损失）
    if _messages_tokens(history) <= soft_trigger:
        return TurnBudgetPlan(
            keep_history=list(history),
            dropped_count=0,
            carry_forward_added=False,
            target_input_tokens=soft_trigger,
            mecw=eff_window,
            hard_ceiling=hard_ceiling,
        )

    # 超软阈值：裁历史。有摘要兜底才收紧到一半（丢的有摘要补）；否则裁到 soft_trigger（no-loss）
    carry = cfg.carry_forward_location == "history_prefix" and bool(session_summary_text)
    hist_budget = soft_trigger // 2 if carry else soft_trigger
    builder = ContextBuilder(max_history_tokens=hist_budget)
    keep = builder.build(list(history))
    dropped = len(history) - len(keep)
    carry_added = False
    if carry and dropped > 0 and session_summary_text:
        keep = [
            {"role": "system", "content": f"<session_summary>\n{session_summary_text}\n</session_summary>"}
        ] + keep
        carry_added = True
    return TurnBudgetPlan(
        keep_history=keep,
        dropped_count=dropped,
        carry_forward_added=carry_added,
        target_input_tokens=soft_trigger,
        mecw=eff_window,
        hard_ceiling=hard_ceiling,
    )
