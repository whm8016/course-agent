"""LLM 掌握度判定器：闭集序数判定，一段一次调用（plan「掌握度估计器重构」阶段B）。

感知与记账分离：LLM 只判「这一段里学生在这个主题上表现如何」（poor/fair/good/
unknown + confidence + 原文引用），数字由 mastery_state 的确定性 Elo 折叠算出。
输出契约借 ParLD（AAAI 2026）的「闭集 KC + 序数等级 + 文本解释」；不借其四智能体链。

prompt 口径：学生原话全文 + 教师回复每轮截前 100 字（依据见 judge_segment docstring）。

三条硬约束（plan §3c）：
1. topic_id 只能取自传入闭集（id 天然对齐，消除 hash 漂移），越界丢弃；
2. unknown 是一等选项——判不出就弃权，不写观测（用弃权率换 precision，
   对齐 Goel 协议「含糊或双向一律丢弃」）；
3. evidence_quote 必须是该段子串，否则该条判定作废并计入 hallucination 指标。

开关条件 = 该课程 course_topic 非空，与 LIGHTRAG_ENABLED 解耦（判定走课程主题
闭集，不依赖 LightRAG 实体目录）。失败 fail-safe 返回空（不阻塞巩固）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.llm.json_extract import extract_json_from_llm

logger = logging.getLogger(__name__)

# 判定等级合法集
LEVELS = ("poor", "fair", "good", "unknown")

# 教师回复每轮进 prompt 的截断长度
_TEACHER_TRUNCATE_CHARS = 100

_JUDGE_SYSTEM_PROMPT = """你是学习分析判定器。下面给出一个主题连续段的对话（学生原话全文 + 教师回复每轮截断）和该段命中的课程主题列表。

对每个主题，只依据对话判定该生在此主题上的表现等级。输出严格 JSON（不要 markdown 围栏）：
{
  "judgements": [
    {"topic_id": "必须是主题列表中的 id", "level": "poor|fair|good|unknown",
     "confidence": 0.0到1.0, "evidence_quote": "必须逐字取自对话原文"}
  ],
  "error_patterns": [
    {"label": "错误模式简述", "related_topic_ids": ["主题列表中的 id"],
     "evidence_quote": "必须逐字取自对话原文"}
  ]
}

等级定义：
- poor：对话中有明确证据该生未掌握（答错、概念混淆、被纠正后仍错、自述不懂）
- fair：表现平平或证据含糊——无法确认掌握也无法确认未掌握
- good：有明确证据已掌握（答对、正确应用、清晰复述）
- unknown：该段没有涉及此主题的有效证据（含糊或双向证据一律弃权，宁弃勿猜）

规则：
- **提问不等于不会**：学生单纯提出概念性问题（「X 是什么/怎么用/为什么」）不构成
  未掌握证据——从零学起问概念是该主题学习的第一步，一律判 unknown 弃权。只有
  学生实际作答/应用出错（答错题、算错数、被纠正后仍错）才可判 poor，只有学生
  实际答对/正确应用/准确复述才可判 good。
- topic_id 只能取自给定主题列表，不得编造
- evidence_quote 必须是对话原文的逐字子串（引用哪句就抄哪句，不改写不翻译）
- 拿不准就 unknown：弃权不扣分，猜错会污染掌握度记录
- error_patterns 只记对话中实际发生的错误；没有就空数组"""


@dataclass(frozen=True)
class Judgement:
    """一条通过校验的判定（unknown 的不产出观测，只进弃权统计）。"""

    topic_id: str
    level: str
    confidence: float
    evidence_quote: str


@dataclass(frozen=True)
class JudgeResult:
    """一段的判定结果（校验后）+ 质量指标（评测口径）。"""

    judgements: list[Judgement]      # 已滤掉 unknown 与校验失败条目
    abstained: list[str]             # unknown 的 topic_id（弃权统计）
    error_patterns: list[dict]       # {"label", "related_topic_ids", "evidence_quote"}（闭集 id）
    dropped_invalid_topic: int       # topic_id 越界丢弃数
    dropped_bad_quote: int           # 引用非子串丢弃数


def _validate(
    raw: dict, topic_ids: list[str], segment_text: str
) -> JudgeResult:
    """校验 LLM 返回：闭集过滤 + 引用子串校验。纯函数，可单测。"""
    valid = set(topic_ids)
    judgements: list[Judgement] = []
    abstained: list[str] = []
    bad_topic = bad_quote = 0

    for j in raw.get("judgements") or []:
        if not isinstance(j, dict):
            continue
        tid = str(j.get("topic_id") or "").strip()
        if tid not in valid:
            bad_topic += 1
            continue
        level = str(j.get("level") or "").strip().lower()
        if level not in LEVELS:
            bad_topic += 1
            continue
        if level == "unknown":
            abstained.append(tid)
            continue
        quote = str(j.get("evidence_quote") or "")
        if quote not in segment_text:
            bad_quote += 1
            continue
        try:
            conf = max(0.0, min(1.0, float(j.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.5
        judgements.append(Judgement(tid, level, conf, quote))

    # error_patterns 同口径：闭集过滤 + 引用子串校验（挂闭集 id 解决旧对齐问题）
    patterns: list[dict] = []
    for ep in raw.get("error_patterns") or []:
        if not isinstance(ep, dict):
            continue
        rel = [t for t in (ep.get("related_topic_ids") or []) if t in valid]
        if not rel:
            bad_topic += 1
            continue
        quote = str(ep.get("evidence_quote") or "")
        if quote and quote not in segment_text:
            bad_quote += 1
            continue
        label = str(ep.get("label") or "").strip()
        if label:
            patterns.append(
                {"label": label, "related_topic_ids": rel, "evidence_quote": quote}
            )

    return JudgeResult(judgements, abstained, patterns, bad_topic, bad_quote)


async def judge_segment(
    segment_turns: list[tuple[str, str]],
    topics: list[tuple[str, str, str]],
    track: dict[str, str] | None = None,
) -> JudgeResult | None:
    """一段一次 LLM 闭集序数判定。学生原话全文 + 教师回复每轮截断前 100 字。

    教师回复全量不进 prompt（省 token 大头），但完全去掉会丢「被纠正后仍错」
    这类只有教师回应里才可见的证据——run_judge_context_ab 双臂实测（一致率
    0.708 < 0.85），只拼学生的臂在学生只言片语的段上系统性激进误判（abstain→
    poor/good）。折中：每轮教师回复只保留开头 100 字（纠正/答疑的关键句通常
    在开头），既能看见「不对，应该是…」类证据，又砍掉长篇讲解的主体。

    Args:
        segment_turns: [(user_msg, assistant_msg)]，段内全部轮次原文。
        topics: [(topic_id, label, definition)]，仅该段 match_topics 命中的主题（≤3）。
            空列表 = 无可判主题，直接返回 None（不调 LLM）。
        track: 可选 usage 埋点维度（user_id/course_id/session_id），传入则 LLM 用量
            落 Prometheus + 明细（mode 固定 memory_judge）；不传零变化（评测脚本用）。

    Returns:
        JudgeResult；LLM 不可用/解析失败返回 None（fail-safe，调用方跳过掌握度
        路径，不阻塞巩固）。
    """
    if not topics:
        return None
    lines = [
        f"- {tid}：{label}。{definition or ''}".strip() for tid, label, definition in topics
    ]
    # 截断推导一次，dialog 与 segment_text 共用：模型看得到什么就只能引用什么
    # （segment_text 是 evidence_quote 子串校验基准），两处口径不同会导致
    # 「引用被截掉的后文」白白计入 bad_quote
    truncated = [
        (u or "", (a or "")[:_TEACHER_TRUNCATE_CHARS]) for u, a in segment_turns
    ]
    dialog = "\n".join(f"学生: {u}\n教师: {a}" for u, a in truncated)
    prompt = (
        f"主题列表（只可从中选择）：\n{''.join(lines)}\n\n对话原文：\n{dialog}"
    )
    segment_text = "\n".join(
        filter(None, (f"{u}\n{a}" for u, a in truncated))
    )

    try:
        from core.llm.llm import client as async_openai_client
        from settings import get_settings

        model = get_settings().llm.text_model
        resp = await async_openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            # 推理模型的 reasoning_content 与正文共享 max_tokens（实测该模型推理
            # 可写到 1.4 万字：800/4000 均被吃完、finish=length 正文为空判定全挂，
            # 8000 实测 finish=stop）；max_tokens 是上限非花费，判定正文本身很短
            max_tokens=8000,
            stream=False,
        )
        if track is not None:
            # 用量埋点：判定器是后台巩固的主力 LLM 消耗（每段一次），best-effort 不阻塞
            from core.analytics.token_usage import track_llm_response

            await track_llm_response(resp, model=model, mode="memory_judge", **track)
        raw = (resp.choices[0].message.content or "").strip()
        data = extract_json_from_llm(raw)
        if not data:
            logger.warning("[judge] JSON 解析失败，raw 前200字：%s", raw[:200])
            return None
    except Exception as exc:
        logger.warning("[judge] LLM 判定失败（fail-safe 返回 None）：%s", exc, exc_info=True)
        return None

    result = _validate(data, [t[0] for t in topics], segment_text)
    logger.info(
        "[judge] OK judgements=%d abstain=%d bad_topic=%d bad_quote=%d errors=%d",
        len(result.judgements), len(result.abstained),
        result.dropped_invalid_topic, result.dropped_bad_quote,
        len(result.error_patterns),
    )
    return result


__all__ = ["Judgement", "JudgeResult", "judge_segment", "LEVELS"]
