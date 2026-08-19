"""简答题 LLM 判分器：闭集等级 + 弃权 + 引用校验（judge.py 三件套同构，A3）。

感知与记账分离：LLM 只判「学生作答对照参考答案的等级」（good/fair/poor/unknown
+ confidence + 原文引用 + 一句话反馈），等级→outcome 数字由
mastery_state.LEVEL_OUTCOME（good=1.0/fair=0.5/poor=0.0）承担，不在这里算。

三条硬约束（与 memory/judge.py 同口径）：
1. grade 只能取闭集，越界丢弃；
2. unknown 是一等选项——判不出就弃权，不写观测（用弃权率换 precision）；
3. evidence_quote 必须是「学生作答」或「参考答案」的逐字子串，否则整条作废
   （校验基准 = 两段拼接全文，无截断——模型看得到什么就只能引用什么）。

失败 fail-safe 返回 None（调用方按 ungraded 处理，不阻塞提交主流程）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.llm.json_extract import extract_json_from_llm

logger = logging.getLogger(__name__)

# 判分等级合法集（与 mastery LEVELS 对齐，多一个 unknown 弃权位）
GRADES = ("good", "fair", "poor", "unknown")

_ANSWER_JUDGE_SYSTEM_PROMPT = """你是答题判分器。给定题目、参考答案和学生作答，判定学生作答的等级。

输出严格 JSON（不要 markdown 围栏）：
{"grade": "good|fair|poor|unknown", "confidence": 0.0到1.0,
 "evidence_quote": "必须逐字取自「学生作答」或「参考答案」原文",
 "feedback": "一句话反馈，面向学生"}

等级定义（以参考答案要点为唯一标准）：
- good：要点基本覆盖，核心概念/步骤正确
- fair：部分正确——方向对但缺关键要点，或有小错不影响主干
- poor：错误、答非所问、或与参考答案要点基本不沾边
- unknown：无法判定（作答空泛到无从对照，或参考答案本身无明确要点）——宁弃勿猜

规则：
- 只依据题面与参考答案判断，不引入外部标准；不因表达风格/错别字扣等级
- evidence_quote 必须是「学生作答」或「参考答案」的逐字子串（引用哪句就抄哪句，不改写不翻译）
- 拿不准就 unknown：弃权不扣分，猜错会污染掌握度记录
- feedback 不超过 60 字，good 时也写（指出做对了什么）"""


@dataclass(frozen=True)
class AnswerJudgeResult:
    """一条通过校验的判分结果。"""

    grade: str
    confidence: float
    evidence_quote: str
    feedback: str


def _validate(
    raw: dict, user_answer: str, correct_answer: str
) -> AnswerJudgeResult | None:
    """校验 LLM 返回：等级闭集 + 引用子串校验。纯函数，可单测。

    返回 None 的三种情形（调用方统一按 ungraded 处理，区分只进日志）：
    unknown 弃权 / grade 越闭集 / evidence_quote 非子串（hallucination）。
    """
    grade = str(raw.get("grade") or "").strip().lower()
    if grade not in GRADES:
        return None
    if grade == "unknown":
        return None
    quote = str(raw.get("evidence_quote") or "")
    basis = f"{user_answer}\n{correct_answer}"
    if quote not in basis:
        return None
    try:
        conf = max(0.0, min(1.0, float(raw.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.5
    feedback = str(raw.get("feedback") or "").strip()[:200]
    return AnswerJudgeResult(grade, conf, quote, feedback)


async def judge_answer(
    question: str,
    correct_answer: str,
    user_answer: str,
    track: dict[str, str] | None = None,
) -> AnswerJudgeResult | None:
    """一道题一次 LLM 判分（short_answer/written/fill_in_blank）。

    Args:
        track: 可选 usage 埋点维度（user_id/course_id/session_id），传入则 LLM 用量
            落 Prometheus + 明细（mode 固定 question_judge）；不传零变化（评测脚本用）。

    Returns:
        AnswerJudgeResult；LLM 不可用/解析失败/校验不过返回 None（fail-safe）。
    """
    prompt = (
        f"题目：\n{question}\n\n参考答案：\n{correct_answer}\n\n学生作答：\n{user_answer}"
    )
    try:
        from core.llm.llm import client as async_openai_client
        from settings import get_settings

        model = get_settings().llm.text_model
        resp = await async_openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ANSWER_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            # max_tokens=8000 同 judge.py：推理模型 reasoning 与正文共享额度，短了正文为空
            max_tokens=8000,
            stream=False,
        )
        if track is not None:
            from core.analytics.token_usage import track_llm_response

            await track_llm_response(resp, model=model, mode="question_judge", **track)
        raw = (resp.choices[0].message.content or "").strip()
        data = extract_json_from_llm(raw)
        if not data:
            logger.warning("[answer_judge] JSON 解析失败，raw 前200字：%s", raw[:200])
            return None
    except Exception as exc:
        logger.warning("[answer_judge] LLM 判分失败（fail-safe 返回 None）：%s", exc)
        return None

    result = _validate(data, user_answer, correct_answer)
    if result is None:
        logger.info("[answer_judge] 判分被弃（unknown/越闭集/引用非子串）")
        return None
    logger.info("[answer_judge] OK grade=%s conf=%.2f", result.grade, result.confidence)
    return result


__all__ = ["AnswerJudgeResult", "GRADES", "judge_answer"]
