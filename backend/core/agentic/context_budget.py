"""通用上下文管理：双阈值预算 + 轮内单 pass 压缩执行器 + 反应式回合前裁剪。

复用 context_builder.count_tokens（tiktoken cl100k_base + len//4 兜底）做 token 记账。
窗口解析与双阈值预算在 context_window.py（resolve_effective_window 三级解析 +
compute_budgets 减法留白）。本模块提供：

- token_count_slices   逐切片 token 计数（system prompt 预算分解 + 超阈值告警）
- system_t1_chars      system prompt 稳定前缀（T1）的字符长度，供 cache_control 在 T1/T2 边界放断点
- cap_dynamic_slices   T2 动态层集体裁剪（按优先级整段丢弃最低优先级切片，直到回到预算内）
- compute_budgets      （转发 context_window）(soft_trigger, hard_ceiling) 双阈值
- enforce              轮内单 pass 压缩执行器：①清旧 tool 结果(免费) -> ②折叠进唯一
                      checkpoint(一次 LLM，替换非叠加) -> ③确定性收缩循环(hard 兜底)
- plan_turn            回合前反应式裁剪：未超软阈值原样返回，超了才裁历史 + 摘要续接

轮内压缩形态（取代旧 L1->L2->L3 三级级联）：一次通过按固定优先序动作，checkpoint
状态由 messages 自身携带（<compacted> system 消息自定位），enforce 无外部状态、幂等
可重入。soft~hard 之间且清理/折叠门限不满足 = 防抖带（滞回区间），静默放行；越 hard
由收缩循环构造性收敛。

分层（驱逐优先级从高到低）：
  T1 静态锚点（loop_system / course_prompt / skills / tool_hint…）永不裁、逐字稳定保 prefix cache；
  T2 易变层（memory / mastery / summary / now）承压时按优先级裁（KB-seed->memory->mastery->summary）。

调研依据：Anthropic Context Management API 两条——clear_tool_uses_20250919（时间序最旧
优先清 tool 结果、keep 最近 N 对、exclude_tools 白名单、clear_at_least 清不够宁可不clear
护缓存）与 compact_20260112（compaction block 之前的内容被忽略=替换语义、block 自居
历史、trigger 阈值触发）；OpenHands software-agent-sdk condenser（LLMSummarizingCondenser
的 forgotten_event_ids+summary_offset 顶替语义、minimum_progress 防抖、HARD 需求收缩
兜底）；Claude Code 五级压缩级联（反应式兜底+熔断，见 loop.py）+ context rot 论文
arXiv:2606.29718（compaction+trimming 组合最优）+ arXiv:2508.21433（Observation Masking
成本减半）+ Anthropic Prompt Caching（cache_control read 0.1x）。
"""
from __future__ import annotations

import dataclasses
from typing import Any

from core.agentic.context_window import (
    compute_budgets,
    count_tokens,
    count_tokens_messages,
    resolve_effective_window,
)
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
# 协调器骨架（plan_turn 回合前反应式 + enforce 轮内单 pass 压缩）
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TurnBudgetPlan:
    """回合前预算规划产物，随 context.metadata["_budget_plan"] 跨调用传递。

    keep_history           裁剪后历史（carry_forward 时首位可能前插 <session_summary> system 消息）
    dropped_count          被裁掉的历史消息数
    carry_forward_added    是否前插了摘要续接块（是则 turn_runtime 清空 ctx.session_summary 避免双付）
    dropped_beyond_cursor  本次裁剪是否丢掉摘要游标之后（未被压缩卡覆盖）的消息。
                           读侧（load_uncompressed_tail）只装游标之后的消息 ⇒ 被裁即越游标
                           ——那些消息既不在卡里也不在本轮原文里，违反不变式，调用方
                           （turn_runtime）据此触发当轮同步压缩兜底。游客/body.history 无
                           游标概念，恒 False（无会话可压）。默认 False 兼容旧构造。
    target_input_tokens    软阈值/质量线（= compute_budgets 的 soft_trigger；超了才裁）。eval getattr 兼容字段
    mecw                   有效上下文窗口（= resolve_effective_window；三级解析）。eval getattr 兼容字段
    hard_ceiling           硬天花板/安全线（L3 降级与反应式兜底用）
    """

    keep_history: list[dict[str, Any]]
    dropped_count: int
    carry_forward_added: bool
    target_input_tokens: int
    mecw: int
    hard_ceiling: int
    dropped_beyond_cursor: bool = False


@dataclasses.dataclass
class EnforceReport:
    """轮内单 pass 压缩执行报告。"""

    tokens_before: int
    tokens_after: int
    cleared_tool_results: int   # ①清理的 tool 结果条数
    summary_added: bool         # ②是否折叠续写了 checkpoint
    dropped_messages: int       # ③收缩循环丢弃的消息条数
    # 观测补正文（此前只留计数，LangSmith 上"删了什么/压成了什么"无从查证）。
    # cleared/dropped 均为 200 字 preview 口径——全文唯一消费者是 ②折叠的摘要输入，
    # 报告只服务上报，存全文是白搬（join 后仍被 [:800] 截掉）。
    cleared_originals: list[str] = dataclasses.field(default_factory=list)  # ①被清 tool 原文前 200 字
    summary_text: str = ""      # ②checkpoint 新正文（失败/未触发为空串）
    dropped_originals: list[str] = dataclasses.field(default_factory=list)  # ③被丢消息 preview（msg_preview）
    compacted_tokens: int = 0   # ②折叠回收的 token（区段+旧 checkpoint - 新 checkpoint）
    checkpoint_idx: int = -1    # ②折叠后 <compacted> 消息在 messages 中的索引（无 checkpoint=-1）


def _messages_tokens(messages: list[dict[str, Any]]) -> int:
    """messages 列表的总 token 估算。委托 context_window.count_tokens_messages（单一真相源）。

    旧本体内联只计 content + tool_calls JSON（与 context_policy.token_estimate 逐字节重复，
    存在静默漂移风险），且漏计每条消息 4-8 token 结构开销与图片 token。现统一走
    count_tokens_messages 口径：content/tool_calls/multimodal 逐 part + 每消息 +3/name +1。
    """
    return count_tokens_messages(messages)


# ---------------------------------------------------------------------------
# 轮内单 pass 压缩（①清理 -> ②折叠 -> ③收缩；取代旧 L1->L2->L3 级联）
# ---------------------------------------------------------------------------

# 防 thrash 门限（模块常量非配置——对标 Anthropic clear_at_least / OpenHands minimum_progress）
_CLEAR_AT_LEAST_TOKENS = 2000    # ①清理：可清量不足此值整体不执行（护 prefix cache，攒一波再清）
_MIN_FOLD_GAIN_TOKENS = 2000    # ②折叠：区段可回收 token 不足此值不调 LLM（minimum_progress 防抖）
_SHRINK_MAX_STEPS = 8           # ③收缩：循环步数上限（防 tokenizer 估算误差下死循环）
_SHRINK_TOOL_MIN_CHARS = 800    # ③a：tool 结果短于此不截（截了也没意义）
_SHRINK_HISTORY_KEEP_CHARS = 500  # ③c：历史消息最后手段截断保留前 N 字
_FOLD_PREVIEW_CHARS = 600       # ②折叠：单条消息进摘要输入的 preview 上限

_COMPACTED_TAG = "<compacted>"
_CLEAR_TOMBSTONE = "[cleared: {name}]"


def _is_checkpoint(msg: dict[str, Any]) -> bool:
    """消息是否是轮内压缩 checkpoint（唯一摘要载体）。"""
    return msg.get("role") == "system" and str(msg.get("content") or "").startswith(_COMPACTED_TAG)


def _locate_checkpoint(messages: list[dict[str, Any]]) -> tuple[int, str]:
    """定位 checkpoint 消息，返回 (索引, 摘要正文)。无则 (-1, "")。

    checkpoint 状态由 messages 自身携带（对标 Anthropic compact 的 compaction block
    自居历史、其前内容被忽略的语义）：enforce 每次重扫定位，无外部 state——413 兜底
    删消息后的索引漂移、answer_now 浅拷贝副本天然一致（重扫即对齐）。
    """
    for i, m in enumerate(messages):
        if _is_checkpoint(m):
            body = str(m.get("content") or "")
            text = body[len(_COMPACTED_TAG):].strip()
            if text.endswith("</compacted>"):
                text = text[: -len("</compacted>")].strip()
            return i, text
    return -1, ""


def clear_old_tool_results(
    messages: list[dict[str, Any]],
    keep_recent_turns: int,
    *,
    exclude_tools: list[str] | None = None,
    captured_originals: dict[int, str] | None = None,
) -> int:
    """①清理（免费）：把保护窗外的旧 tool 结果按时间序清成墓碑。

    Anthropic clear_tool_uses_20250919 客户端复刻：时间序最旧优先（已清区收敛成单调
    前缀，清过的墓碑逐字不再变 -> prefix cache 可重新命中）、keep 最近 N 轮原文、
    exclude_tools 白名单（ask_user 等不可重建输入）永不清理。与旧 evict_tool_results
    的差异：删 score 散点优先级（Anthropic/OpenHands/Claude Code 三家对标皆无此设计，
    散点清理会让已清区永不收敛）；RAG 来源引用段（_is_source_para）降级为跳过不清、
    保留在原地；一次性清掉全部候选——可清量不足 _CLEAR_AT_LEAST_TOKENS 时整体跳过
    （clear_at_least：小步清理只会反复打断缓存，攒够一波再清）。

    复用 context_policy._split_turns 切轮；前导 turns[0]（system+历史+当前 user）天然
    不含 role=tool 不受影响。captured_originals 非空时按时间序记 id(msg)->被清原文
    （供 ②折叠用原文而非墓碑）。就地改 messages，返回清理条数。
    """
    from core.agentic.context_policy import _is_source_para, _split_paragraphs, _split_turns

    turns = _split_turns(messages)
    if len(turns) <= 1:
        return 0
    protected_from = len(turns) - keep_recent_turns if keep_recent_turns > 0 else len(turns)
    protected_from = max(1, protected_from)
    if protected_from <= 1:
        return 0  # 无保护窗外的轮可清

    if exclude_tools is None:
        exclude_tools = get_settings().context_budget.exclude_tools
    exclude_lower = {(n or "").lower() for n in exclude_tools}

    candidates: list[dict[str, Any]] = []
    gain = 0
    for turn in turns[1:protected_from]:
        for msg in turn:
            if msg.get("role") != "tool":
                continue
            if (msg.get("name") or "").lower() in exclude_lower:
                continue
            c = str(msg.get("content") or "")
            if not c or c.startswith("[cleared:"):
                continue
            if any(_is_source_para(p) for p in _split_paragraphs(c)):
                continue  # RAG 来源引用段保留原地
            candidates.append(msg)
            gain += count_tokens(c) - count_tokens(_CLEAR_TOMBSTONE.format(name=msg.get("name") or "tool"))
    if not candidates or gain < _CLEAR_AT_LEAST_TOKENS:
        return 0
    for msg in candidates:
        if captured_originals is not None:
            captured_originals[id(msg)] = str(msg.get("content") or "")
        msg["content"] = _CLEAR_TOMBSTONE.format(name=(msg.get("name") or "tool"))
    return len(candidates)


async def _fold_uncompacted(
    messages: list[dict[str, Any]],
    model: str,
    protected_from: int,
    captured_map: dict[int, str],
) -> tuple[bool, str, int]:
    """②折叠（一次 LLM）：把 checkpoint 与保护窗之间的完整轮折叠续写进唯一 checkpoint。

    替换语义（对标 Anthropic compact「compaction block 之前的内容被忽略」+ OpenHands
    Condensation 的 forgotten_event_ids+summary 顶替）：删除整个区段（assistant 中间
    结论 + tool 结果/墓碑一并），在区段起点放唯一一条 <compacted> system 消息——不是
    新增第二条摘要（旧 L2 插入语义会累积多条 <context_summary> 且窗口回升）。assistant
    文本进摘要修优先级倒挂：旧 L3 裸丢的恰是模型自己的推理，比工具原始输出更值钱。

    区段按 _split_turns 整轮删，tool_use<->tool_result 配对完整性由构造保证；前导
    turns[0] 与保护窗永不动。区段可回收 token 不足 _MIN_FOLD_GAIN_TOKENS 跳过
    （minimum_progress 防抖）；摘要失败返回 False（区段保留原样，信息不丢）。

    返回 (是否折叠, checkpoint 新正文, 回收 token)。
    """
    from core.agentic.context_policy import _split_turns, _summarize_masked_text

    turns = _split_turns(messages)
    seg_msgs = [m for t in turns[1:protected_from] for m in t]
    if not seg_msgs:
        return False, "", 0
    ck_idx, ck_text = _locate_checkpoint(messages)
    seg_tokens = count_tokens_messages(seg_msgs)
    old_ck_tokens = count_tokens_messages([messages[ck_idx]]) if ck_idx >= 0 else 0
    if seg_tokens + old_ck_tokens < _MIN_FOLD_GAIN_TOKENS:
        return False, "", 0

    # 折叠输入 = 旧 checkpoint 正文 + 区段逐消息 preview（tool 用清理前原文，非墓碑）
    parts: list[str] = [ck_text] if ck_text else []
    for m in seg_msgs:
        if m.get("role") == "tool":
            text = captured_map.get(id(m)) or str(m.get("content") or "")
            parts.append(f"[{(m.get('name') or 'tool')}] {text[:_FOLD_PREVIEW_CHARS]}")
        else:
            parts.append(f"[{m.get('role')}] {str(m.get('content') or '')[:_FOLD_PREVIEW_CHARS]}")
    summary = await _summarize_masked_text("\n\n".join(parts), model)
    if not summary:
        return False, "", 0

    # 区段是 messages 的连续切片（_split_turns 保序）；旧 checkpoint 在前导内，换新
    start = next(i for i, m in enumerate(messages) if id(m) == id(seg_msgs[0]))
    end = start + len(seg_msgs)
    checkpoint_msg = {"role": "system", "content": f"{_COMPACTED_TAG}\n{summary}\n</compacted>"}
    prefix = [m for m in messages[:start] if not _is_checkpoint(m)]
    messages[:] = prefix + [checkpoint_msg] + messages[end:]
    reclaimed = seg_tokens + old_ck_tokens - count_tokens_messages([checkpoint_msg])
    return True, summary, max(reclaimed, 0)


def _drop_oldest_turn(
    messages: list[dict[str, Any]], captured_previews: list[str] | None = None
) -> int:
    """丢掉最旧的一个非前导 tool 轮（turns[1]），保 tool_use<->tool_result 配对。

    drop_oldest_turn_group 的单轮原语：③b 每次丢一轮、重算、再决定是否继续；前导
    turns[0]（system+历史+当前 user）不动；至少保留 1 个 tool 轮（全丢则本轮工具
    上下文清零）。返回丢弃条数（0=无可丢）。
    """
    from core.agentic.context_policy import _split_turns

    turns = _split_turns(messages)
    if len(turns) <= 2:
        return 0
    dropped = turns[1]
    if captured_previews is not None:
        captured_previews.extend(msg_preview(m) for m in dropped)
    survivors = turns[:1] + turns[2:]
    messages[:] = [m for t in survivors for m in t]
    return len(dropped)


def drop_oldest_turn_group(
    messages: list[dict[str, Any]],
    *,
    fraction: float = 0.2,
    captured_originals: list[str] | None = None,
) -> int:
    """L3 式硬降级：丢弃最旧的 fraction（默认 20%）消息组（按 turn 切）。

    loop.py 413 反应式兜底沿用本入口（语义不变：丢最旧 20%、保前导）。内部复用
    _drop_oldest_turn 单轮原语。captured_originals 非空时把被丢消息 preview append
    进去（msg_preview 格式），供 LangSmith 上报"删了什么"。返回丢弃条数。
    """
    from core.agentic.context_policy import _split_turns

    turns = _split_turns(messages)
    if len(turns) <= 2:  # 仅 1 个 tool 轮 + 前导，无余量可丢
        return 0
    droppable = len(turns) - 1
    drop_n = min(max(1, int(droppable * fraction)), droppable)
    previews: list[str] = []
    dropped = 0
    for _ in range(drop_n):
        dropped += _drop_oldest_turn(messages, captured_previews=previews)
    if captured_originals is not None:
        captured_originals.extend(previews)
    return dropped


def _shrink_cap_tool_results(messages: list[dict[str, Any]]) -> int:
    """③a：把最长的一条未清理 tool 结果截半（复用 cap_tool_result 段落边界感知截断）。

    排除墓碑与白名单（ask_user 用户答复不做硬截）；短于 _SHRINK_TOOL_MIN_CHARS 的
    不值得截。每步只截一条（循环重算决定下一条）。返回 1=截了 / 0=无可截。
    """
    from core.agentic.context_policy import cap_tool_result

    exclude_lower = {(n or "").lower() for n in get_settings().context_budget.exclude_tools}
    best: dict[str, Any] | None = None
    best_len = _SHRINK_TOOL_MIN_CHARS
    for m in messages:
        if m.get("role") != "tool":
            continue
        if (m.get("name") or "").lower() in exclude_lower:
            continue
        c = m.get("content")
        if not isinstance(c, str) or c.startswith("[cleared:"):
            continue
        if len(c) > best_len:
            best, best_len = m, len(c)
    if best is None:
        return 0
    best["content"] = cap_tool_result(best["content"], best_len // 2)
    return 1


def _shrink_truncate_history(
    messages: list[dict[str, Any]], captured_previews: list[str] | None = None
) -> int:
    """③c（最后手段）：把前导里最旧的一条超长历史消息截到 _SHRINK_HISTORY_KEEP_CHARS。

    只动 user/assistant 且 content 为 str 的历史消息（多模态 list 不动）；前导末尾的
    当前 user 消息（本轮问题）不动。每步只截一条。返回 1=截了 / 0=无可截。
    """
    lead_end = next(
        (i for i, m in enumerate(messages)
         if m.get("role") == "assistant" and m.get("tool_calls")),
        len(messages),
    )
    last_user_idx = max(
        (i for i, m in enumerate(messages[:lead_end]) if m.get("role") == "user"),
        default=-1,
    )
    for i, m in enumerate(messages[:lead_end]):
        if i == last_user_idx:
            continue
        if m.get("role") not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str) and len(c) > _SHRINK_HISTORY_KEEP_CHARS:
            if captured_previews is not None:
                captured_previews.append(msg_preview(m))
            m["content"] = c[:_SHRINK_HISTORY_KEEP_CHARS] + "…[早期历史已截断]"
            return 1
    return 0


def msg_preview(m: dict[str, Any]) -> str:
    """单条消息的观测 preview（``[role] 正文前 200 字``）。

    L3 dropped_originals 与 turn_runtime 回合前裁剪 preview 共用——把格式钉进代码，
    避免 LangSmith 上 in_turn 与 pretrim 的 preview 各改各的、静默漂移。
    """
    return f"[{m.get('role')}] {str(m.get('content') or '')[:200]}"


async def enforce(
    messages: list[dict[str, Any]],
    model: str,
    plan: TurnBudgetPlan | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> EnforceReport:
    """轮内单 pass 压缩执行器（loop 每轮调一次）：①清理 -> ②折叠 -> ③收缩，一次通过。

    ①清理（免费）：超 soft_trigger 时把保护窗（keep_recent_turns 轮）外的旧 tool 结果
        按时间序清成墓碑（clear_old_tool_results）；可清量不足 _CLEAR_AT_LEAST_TOKENS
        不执行。白名单 exclude_tools（ask_user）永不清理。
    ②折叠（一次 LLM）：清后仍超 soft_trigger 才触发，把 checkpoint 与保护窗之间的完整
        轮折叠续写进唯一 <compacted> system 消息（_fold_uncompacted，替换非叠加）；
        区段可回收 token 不足 _MIN_FOLD_GAIN_TOKENS 跳过（防抖）。
    ③收缩循环（超 hard_ceiling）：按优先序逐级降级——③a 截长 tool 结果 -> ③b 丢最旧
        一轮 -> ③c 截最旧历史，每步重算直到 fit（_SHRINK_MAX_STEPS 上限），构造性保证
        请求发得出去；耗尽仍超显式 raise（fail-fast，不静默发出注定 413 的请求）。

    未超 soft_trigger 零干预（反应式）。soft~hard 之间且 ①② 门限不满足 = 防抖带
    （滞回区间），静默放行——不是死区：越 hard 必被 ③ 收敛。压缩落点自然低于
    soft_trigger（非保护区段整段折走，剩余 = 前导 + checkpoint + 保护窗），下轮不会
    立即再触发。

    checkpoint 状态由 messages 自身携带（_locate_checkpoint 重扫定位），本函数无外部
    状态、幂等可重入；answer_now 浅拷贝副本上折叠只影响副本列表（清墓碑经共享 dict
    同步可见，无害且幂等）。

    plan 当前未用（回合前 plan_turn 已做反应式裁剪），保留参数供后续 per-turn 策略。
    tools：本轮请求携带的工具定义 schema（对标 LiteLLM _count_extra 把 tools 计入请求前
    预算——schema 是请求的真实组成部分，教学 agent 十几个工具 ≈3-8k token，不计则 soft
    判断偏乐观）。None=不带 tools 的请求（旧行为）；报告的 tokens_before/after 均含 tools。
    """
    cfg = get_settings().context_budget
    soft_trigger, hard_ceiling = compute_budgets(model)
    # 总量 = count_tokens_messages(messages, tools=tools)：messages+tools 联合计数正是
    # 该函数的设计用途（含 system×tools 的 -4 折扣）。
    before = count_tokens_messages(messages, tools=tools)
    total = before
    cleared = 0
    summary_added = False
    dropped = 0
    summary_text = ""
    compacted_tokens = 0
    checkpoint_idx = -1

    # captured：id(msg) -> 被清原文（时间序）。②折叠用原文而非墓碑；报告 preview 取 values()。
    captured: dict[int, str] = {}
    dropped_previews: list[str] = []

    if total > soft_trigger:
        # ①清理（免费）。门限不够 2000 token 时 noop（clear_at_least 护缓存）。
        cleared = clear_old_tool_results(
            messages, cfg.keep_recent_turns,
            exclude_tools=cfg.exclude_tools, captured_originals=captured,
        )
        if cleared:
            total = count_tokens_messages(messages, tools=tools)

        # ②折叠（仍超软阈值才花 LLM 钱）。
        if total > soft_trigger:
            from core.agentic.context_policy import _split_turns

            turns = _split_turns(messages)
            protected_from = (
                len(turns) - cfg.keep_recent_turns if cfg.keep_recent_turns > 0 else len(turns)
            )
            protected_from = max(1, protected_from)
            folded, summary_text, compacted_tokens = await _fold_uncompacted(
                messages, model, protected_from, captured,
            )
            if folded:
                summary_added = True
                total = count_tokens_messages(messages, tools=tools)
                checkpoint_idx = _locate_checkpoint(messages)[0]

    # ③收缩循环（超硬天花板）：逐级降级，每步恰好一个动作后重算，构造性收敛到 fit。
    # dropped 只计 ③b 丢的消息条数；③c 是截断不是丢弃（preview 仍入观测）。
    if total > hard_ceiling:
        for _ in range(_SHRINK_MAX_STEPS):
            if _shrink_cap_tool_results(messages):
                pass
            elif (n_dropped := _drop_oldest_turn(messages, captured_previews=dropped_previews)) > 0:
                dropped += n_dropped
            elif _shrink_truncate_history(messages, captured_previews=dropped_previews):
                pass
            else:
                break  # 三级都无动作：无可收缩（如 system prompt 本身超限）
            total = count_tokens_messages(messages, tools=tools)
            if total <= hard_ceiling:
                break
        if total > hard_ceiling:
            raise RuntimeError(
                f"context shrink exhausted: {total} tokens > hard_ceiling {hard_ceiling} "
                f"after {_SHRINK_MAX_STEPS} steps (model={model}); system prompt 或单条 "
                "内容超出窗口预留，请检查 T2 切片预算与 cap_tool_result 单条上限"
            )

    return EnforceReport(
        tokens_before=before,
        tokens_after=total,
        cleared_tool_results=cleared,
        summary_added=summary_added,
        dropped_messages=dropped,
        cleared_originals=[v[:200] for v in captured.values()],
        summary_text=summary_text,
        dropped_originals=dropped_previews,
        compacted_tokens=compacted_tokens,
        checkpoint_idx=checkpoint_idx,
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
            dropped_beyond_cursor=False,
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
        # 读侧只装游标之后的消息 ⇒ 被裁即越游标（详见 TurnBudgetPlan 字段注释）
        dropped_beyond_cursor=dropped > 0,
        target_input_tokens=soft_trigger,
        mecw=eff_window,
        hard_ceiling=hard_ceiling,
    )
