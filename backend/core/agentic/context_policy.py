"""轮内上下文预算策略：单条工具结果上限 + 掩码窗口化 + hybrid 兜底摘要。

把旧 ``_snip_tool_results``（按全局字符 > 80000 从最早 role=tool 替换）升级为三段式：

  1. ``cap_tool_result``：单条工具结果段落边界感知截断（按空行切段、最后一段在句号处截断、
     尾部「来源」段优先保留）；单段无边界退化为头尾各半。消除「前端截 2000 字、
     messages 收全量」的不对称。
  2. ``mask_old_observations``：按 **assistant-with-tool_calls 边界**切轮，保留最近 M 轮
     role=tool 原文，更早的替换为占位符。对标 SWE-agent/OpenHands 的 ``f_mask(trajectory, M)``，
     按轮不按全局字符——这是与旧实现及业界的关键差异。
  3. hybrid 兜底摘要（可选）：被掩码轮数超阈值时，用 fast LLM 把被掩码原文压成摘要，
     塞进首个被掩码位（结构不变，仅改 tool content，避免破坏 tool_call_id 匹配）。

【调研依据】
- arXiv:2508.21433（The Complexity Trap, JetBrains, SWE-bench Verified ×5 模型）：
  Observation Masking 相对 Raw 成本减半、解题率持平或略高；纯摘要引发 trajectory elongation
  （多跑 13-15% 轮）。最优是 hybrid——掩码为主、摘要为最后手段。M 推荐取较小值。
- Anthropic context engineering：compaction / structured note-taking；摘要阈值触发 + 可注入
  形态照抄 compaction API 的 trigger 接口。

【token 计数】复用 ``services/session/context_builder.count_tokens``（tiktoken + len//4 降级），
不自造。阈值判定按字符（与旧 _snip_tool_results 同口径，便于行为对照）。

【默认行为】全部由 ``settings.context_policy`` 控制，``enabled=False`` 时 loop 仍走旧
``_snip_tool_results``（行为零变化），便于做 raw vs masking vs masking+summary 消融对照。
"""
from __future__ import annotations

import contextvars
import logging
import re
from typing import Any

from services.session.context_builder import count_tokens
from settings import get_settings

logger = logging.getLogger(__name__)

# 掩码占位符（与旧 _snip_tool_results 的 _MARKER 区分，便于日志区分新旧路径）
_MASK_MARKER = "[早期工具结果已折叠以节省上下文]"
_OMIT_PLACEHOLDER = "…[已省略 {n} 字]…"

# 段落边界感知截断（cap_tool_result）配置
_PARA_SPLIT = re.compile(r"\n\s*\n")  # 空行（含连续空白行）切段
# 尾部来源引用段前缀：截断时优先保留 RAG 来源列表，不被中段省略吃掉
_SOURCE_PREFIXES = ("来源", "source", "参考", "reference", "引用", "references")
# 省略标注长度上界（omitted 取 7 位数 9999999，覆盖千万字级，给预算预留够）
_OMIT_PAD = len(_OMIT_PLACEHOLDER.format(n=9999999))
# 句末/分句边界，最后一段放不下时在这些边界截断（按优先级从前到后）
_SENTENCE_SEPS = ("。", "\n", "；", ";", ".", "!", "?", "！", "？")


def _split_paragraphs(content: str) -> list[str]:
    """按空行把 content 切成段落列表，过滤纯空白段。无空行 → 单元素列表。"""
    return [p for p in _PARA_SPLIT.split(content) if p.strip()]


def _is_source_para(para: str) -> bool:
    """段落是否是尾部来源引用（以「来源/Source/参考」等开头，不区分大小写）。"""
    head = para.lstrip().lower()[:24]
    return any(head.startswith(p) for p in _SOURCE_PREFIXES)


def _truncate_at_sentence(text: str, budget: int) -> str:
    """在 budget 字符内、最近的句末/换行边界处截断；无标点则硬截 budget。

    保证返回长度 <= budget，且不从一句话/一个词中间砍断。
    """
    if len(text) <= budget:
        return text
    for sep in _SENTENCE_SEPS:
        idx = text.rfind(sep, 0, budget)
        if idx > 0:
            return text[: idx + 1]
    return text[:budget]


def _cap_head_tail(content: str, max_chars: int) -> str:
    """头尾各半 + 中间省略标注（旧逻辑）。

    单段（无段落边界）输入退化走此路——纯长文本无段落可按边界切时，保持旧行为零回归。
    """
    omitted = len(content) - max_chars
    avail = max_chars
    while True:
        placeholder = _OMIT_PLACEHOLDER.format(n=omitted)
        avail = max_chars - len(placeholder)
        if avail <= 0:
            return content[:max_chars]
        real_omitted = len(content) - avail
        if real_omitted == omitted:
            break
        omitted = real_omitted
    head = avail // 2
    tail = avail - head
    return f"{content[:head]}{placeholder}{content[-tail:] if tail else ''}"


def cap_tool_result(content: str, max_chars: int) -> str:
    """单条工具结果段落边界感知截断（替代旧头尾各半硬截）。

    多段输入：按空行切段，从头贪心选完整段至超预算，最后一段放不下时在句号/换行处截断
    取前缀（不从一句话中间砍字）；尾部「来源/Source/参考」段优先保留（RAG 来源列表显式
    保护，不被中段省略吃掉）。省略处标注省略字数便于排查。

    单段输入（无段落边界）：退化为头尾各半（``_cap_head_tail``），行为等价旧实现、零回归。

    max_chars<=0 表示不限（原样返回）。结果总长严格 <= max_chars（末尾兜底硬截保不变式）。

    调研依据：RECOMP extractive compressor（Xu et al., ICLR 2024, arXiv:2310.04408）
    思路——按语义边界选关键内容、不破坏句子完整性，纯字符串操作零额外 LLM 调用。
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content

    paras = _split_paragraphs(content)
    if len(paras) <= 1:
        # 无段落边界 → 头尾各半（旧行为，零回归）
        return _cap_head_tail(content, max_chars)

    # 尾部连续来源段优先保留（从末段往前扫，遇到非来源段即停）
    tail_idx = len(paras)
    for i in range(len(paras) - 1, -1, -1):
        if _is_source_para(paras[i]):
            tail_idx = i
        else:
            break
    tail = paras[tail_idx:]
    tail_text = "\n\n".join(tail)
    # tail 占用 = 文本 + 与前文的 \n\n 分隔（tail 非空时算 2）
    tail_block = len(tail_text) + (2 if tail else 0)

    head_paras = paras[:tail_idx]

    # 头部预算 = 总预算 - 尾部 - 省略标注上界 - 分隔符（head↔标注、标注↔tail 两个 \n\n = 4）
    budget = max_chars - tail_block - _OMIT_PAD - 4
    if budget <= 0:
        # 尾部来源段已占满 → 仅保尾部并截到预算内（极端情况，省略全部 head）
        return tail_text[:max_chars]

    # 从头贪心选完整段；最后一段放不下时在句号处截断补前缀
    kept: list[str] = []
    used = 0
    for p in head_paras:
        if used + len(p) + 2 <= budget:
            kept.append(p)
            used += len(p) + 2
        else:
            remain = budget - used
            if remain > 0:
                snippet = _truncate_at_sentence(p, remain)
                if snippet:
                    kept.append(snippet)
            break
    head_text = "\n\n".join(kept)

    omitted = len(content) - len(head_text) - len(tail_text)
    omit_str = _OMIT_PLACEHOLDER.format(n=max(omitted, 0))

    chunks = [c for c in (head_text, omit_str, tail_text) if c]
    result = "\n\n".join(chunks)

    # 兜底：省略标注位数变化或边界估算偏差导致超长时硬截（保证 len <= max_chars 不变式）
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


def _split_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """按 assistant-with-tool_calls 边界把 messages 切成「轮」。

    一轮 = 一个 assistant(tool_calls) + 其后所有 role=tool 结果（直到下一个 assistant）。
    首个 assistant(tool_calls) 之前的消息（system/user）单独成首轮（掩码只动 role=tool，
    前导 system/user 不受影响）。
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            if current is not None:
                turns.append(current)
            current = [msg]
        else:
            if current is None:
                current = []
            current.append(msg)
    if current is not None:
        turns.append(current)
    return turns


def _total_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def mask_old_observations(
    messages: list[dict[str, Any]],
    keep_recent_turns: int,
    budget_chars: int,
    masked_originals: list[str] | None = None,
) -> int:
    """掩码：保留最近 ``keep_recent_turns`` 轮的 role=tool 原文，更早的替换占位符。

    仅在总字符超 ``budget_chars`` 时实际掩码（未超返回 0，行为等价）。贪心：从最早轮掩码，
    总量降到预算内即提前停（少掩码多保信息）。改写前把被掩码 tool 原文 append 到
    ``masked_originals``（供 hybrid 摘要使用）。返回被掩码的轮数。
    """
    if _total_chars(messages) <= budget_chars:
        return 0
    turns = _split_turns(messages)
    if len(turns) <= 1:
        return 0
    # 最近 keep_recent_turns 轮保真；切片特判 0（=全掩码，但 mask 只动 tool，最终答案不受影响）
    older = turns[:-keep_recent_turns] if keep_recent_turns > 0 else turns
    masked_turns = 0
    for turn in older:
        touched = False
        for msg in turn:
            if msg.get("role") == "tool" and msg.get("content") != _MASK_MARKER:
                if masked_originals is not None:
                    masked_originals.append(str(msg.get("content", "")))
                msg["content"] = _MASK_MARKER
                touched = True
        if touched:
            masked_turns += 1
            if _total_chars(messages) <= budget_chars:  # 贪心：够省就停
                break
    return masked_turns


async def _summarize_masked_text(masked_text: str, model: str) -> str:
    """用 fast LLM 把被掩码的多轮工具结果压成摘要。失败返回空串（降级，不阻塞 loop）。"""
    from core.llm.llm import chat_complete
    try:
        return await chat_complete(
            system_prompt=(
                "你是上下文压缩助手。把下方多轮工具调用结果压成一段 300 字以内的关键信息摘要，"
                "保留事实、数据、结论，丢弃冗余与重复。直接输出摘要，不要前缀。"
            ),
            history=[],
            user_message=masked_text[:20000],  # 防止输入过长
            model=model,
            temperature=0.3,
            max_tokens=512,
        )
    except Exception:
        logger.warning("context_policy hybrid 摘要失败，降级为占位符", exc_info=True)
        return ""


async def apply(messages: list[dict[str, Any]], model: str) -> None:
    """上下文预算治理入口（loop 每轮调用一次）：掩码窗口化 + （可选）hybrid 摘要。

    单条 cap（``cap_tool_result``）已在 ``tool_dispatch`` 入口完成——单条结果产生即裁，
    本处不重复（避免冗余）。本函数就地修改 messages。被 ``settings.context_policy.enabled``
    守卫；关闭时 loop 不调本函数（走旧 _snip_tool_results）。
    """
    cfg = get_settings().context_policy

    # 掩码窗口化（仅在超预算时动作；改写前收集被掩码原文供摘要）
    masked_originals: list[str] = []
    masked_turns = mask_old_observations(
        messages, cfg.keep_recent_turns, cfg.budget_chars, masked_originals,
    )

    # hybrid 兜底摘要：被掩码轮数达阈值时，把摘要塞进首个被掩码位（结构不变，仅改 content）
    if cfg.summary_enabled and masked_turns >= cfg.summary_threshold and masked_originals:
        summary = await _summarize_masked_text("\n\n".join(masked_originals)[:20000], model)
        if summary:
            for m in messages:
                if m.get("content") == _MASK_MARKER:
                    m["content"] = f"[早期工具结果摘要] {summary}"
                    break


# ---------------------------------------------------------------------------
# 评测覆盖层：contextvar 指定臂（harness per-task 切换），未设→回落 settings
# （生产零变化）。与 dynamic_tools._CURRENT_LOADER 的注入范式同构：contextvar 是
# task-local，串行评测 set→跑→reset 不污染下一个组合，也不碰任何生产 settings。
# ---------------------------------------------------------------------------
_ARM: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "context_policy_arm", default=None
)

# 四臂枚举（对照 arXiv:2508.21433）
ARMS = ("raw", "masking", "summary_only", "hybrid")


def set_arm(arm: str | None) -> contextvars.Token:
    """harness 跑某臂前设置当前策略，返回 reset 用 token。"""
    return _ARM.set(arm)


def current_arm() -> str | None:
    """当前 contextvar 指定的臂；None=未覆盖（loop 回落 settings 开关）。"""
    return _ARM.get()


def reset_arm(token: contextvars.Token) -> None:
    """harness 跑完某臂后复位（同构 dynamic_tools.reset_deferred_loader）。"""
    if token is not None:
        _ARM.reset(token)


async def _summarize_old_turns(
    messages: list[dict[str, Any]], keep_recent_turns: int, model: str
) -> int:
    """summary_only 臂：窗口外每一轮 tool 结果调 LLM 压成摘要塞回原位（不掩码）。

    论文 H2 的关键臂——摘要丢信息，模型要多调工具补救（trajectory elongation）。
    每轮一次 LLM 调用（extra cost）：摘要塞回该轮首个 tool 消息，其余置占位以保持
    tool_call_id 结构不破坏。返回 LLM 调用次数。
    """
    turns = _split_turns(messages)
    if len(turns) <= keep_recent_turns:
        return 0
    older = turns[:-keep_recent_turns] if keep_recent_turns > 0 else turns
    calls = 0
    for turn in older:
        tool_msgs = [m for m in turn if m.get("role") == "tool" and m.get("content")]
        if not tool_msgs:
            continue
        joined = "\n\n".join(str(m["content"]) for m in tool_msgs)[:20000]
        summary = await _summarize_masked_text(joined, model)
        calls += 1
        if summary:
            tool_msgs[0]["content"] = f"[早期工具结果摘要] {summary}"
            for extra in tool_msgs[1:]:
                extra["content"] = "[已并入上方摘要]"
    return calls


async def apply_arm(messages: list[dict[str, Any]], model: str, arm: str) -> int:
    """按指定臂应用上下文策略，返回本次额外 LLM 调用次数（summary/hybrid 的压缩调用）。

    - raw：完全不裁（论文真基线，可能撑爆 context——这正是 complexity trap 要展示的成本爆炸）
    - masking：按轮掩码（保留最近 M 轮，更早替换占位），不摘要
    - summary_only：窗口外每轮 tool 结果 LLM 摘要塞回（不掩码），测 H2
    - hybrid：先掩码，被掩码轮数≥阈值才整体摘要（论文最优组合）

    masking/hybrid 复用现有 mask_old_observations/_summarize_masked_text；raw/summary_only
    为评测新增。就地修改 messages。
    """
    cfg = get_settings().context_policy
    if arm == "raw":
        return 0
    if arm == "masking":
        mask_old_observations(messages, cfg.keep_recent_turns, cfg.budget_chars)
        return 0
    if arm == "summary_only":
        return await _summarize_old_turns(messages, cfg.keep_recent_turns, model)
    if arm == "hybrid":
        masked_originals: list[str] = []
        masked_turns = mask_old_observations(
            messages, cfg.keep_recent_turns, cfg.budget_chars, masked_originals,
        )
        if masked_turns >= cfg.summary_threshold and masked_originals:
            summary = await _summarize_masked_text(
                "\n\n".join(masked_originals)[:20000], model
            )
            if summary:
                for m in messages:
                    if m.get("content") == _MASK_MARKER:
                        m["content"] = f"[早期工具结果摘要] {summary}"
                        break
            return 1
        return 0
    logger.warning("context_policy.apply_arm: 未知臂 %s，按 raw 处理", arm)
    return 0


def token_estimate(messages: list[dict[str, Any]]) -> int:
    """便利函数：估算 messages 总 token（供测试/日志断言落在窗口预算内）。"""
    return sum(count_tokens(str(m.get("content", ""))) for m in messages)
