"""判分编排：coding→沙箱 run_tests（确定性），简答→LLM judge_answer，统一 GradeResult（A3）。

并发闸：asyncio.Semaphore(4) 排队在事件循环（不占线程池线程）——100 人同时交卷
最坏排队 15-40s 由前端「判分中」态 + 60s 超时兜住；沙箱内部另有 _EXEC_GATE
（threading）防线程层超发。规模再上一个量级时迁 ARQ 队列（consolidate_memory 同
模式），grade_answer 本体不用改。

coding 契约：user_answer 是完整可运行代码（前端 textarea 预填 starter_code，
学生内容含签名骨架）；带 ```python 围栏时抽块（学生可能夹解释文字）。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from core.question.answer_judge import judge_answer
from core.question.sandbox import extract_code_blocks, run_tests

logger = logging.getLogger(__name__)

# coding 判分通过率→等级阈值（模块常量不做 settings：避免过度配置；≥0.7 good）
_GOOD_RATIO = 0.7

# 判分并发闸（见模块 docstring）
_GRADING_SEMAPHORE = asyncio.Semaphore(4)

# 走 LLM 判分的题型（fill_in_blank 变体多，本地字符串比对误伤高——plan §A3 裁决）
_LLM_GRADED_TYPES = ("short_answer", "written", "fill_in_blank")


@dataclass
class GradeResult:
    """一道题的判分结果（端点响应/notebook 字段的统一来源）。"""

    __test__ = False  # 防 pytest 把 Test 开头类误收集

    question_id: str = ""
    grade: str = "ungraded"  # good|fair|poor|ungraded
    score: float | None = None  # coding 通过率 passed/total；其余题型 None
    source: str = ""  # sandbox|llm_judge|""（ungraded）
    feedback: str = ""
    test_results: list[dict] = field(default_factory=list)

    @property
    def is_correct(self) -> bool | None:
        """good→True；fair/poor→False（fair=部分正确=未全对→进错题本）；ungraded→None。"""
        if self.grade == "good":
            return True
        if self.grade in ("fair", "poor"):
            return False
        return None


def _ratio_to_grade(ratio: float) -> str:
    if ratio >= _GOOD_RATIO:
        return "good"
    return "fair" if ratio > 0 else "poor"


async def grade_answer(payload: dict, track: dict[str, str] | None = None) -> GradeResult:
    """判一道题。空作答/不支持题型/判分失败 → ungraded（不写假观测）。

    Args:
        payload: {question_id, question_type, question, correct_answer,
                  user_answer, test_cases, starter_code, language}
        track: 可选 usage 埋点维度（透传 judge_answer，mode=question_judge）。
    """
    qid = str(payload.get("question_id", ""))
    qtype = str(payload.get("question_type", "")).strip().lower()
    user_answer = str(payload.get("user_answer", "") or "")
    if not user_answer.strip():
        return GradeResult(question_id=qid)

    if qtype == "coding":
        test_cases = payload.get("test_cases") or []
        if not test_cases:
            return GradeResult(question_id=qid, feedback="该题暂无自动判分用例")
        code = user_answer
        blocks = extract_code_blocks(user_answer)
        if blocks:
            code = "\n\n".join(blocks)
        # user_answer 自带骨架（textarea 预填），不再前插 starter_code——双重签名是
        # IndentationError
        async with _GRADING_SEMAPHORE:
            r = await asyncio.to_thread(run_tests, code, test_cases)
        ratio = r.pass_ratio
        failed_parts = [f"第{i + 1}条：{d[:80]}" for i, d in r.failed]
        feedback = f"通过 {len(r.passed)}/{r.total} 条用例"
        if failed_parts:
            feedback += "；未通过 " + "；".join(failed_parts)
        results = (
            [{"index": i, "passed": True} for i in r.passed]
            + [{"index": i, "passed": False, "error": d[:200]} for i, d in r.failed]
        )
        results.sort(key=lambda t: t["index"])
        return GradeResult(
            question_id=qid,
            grade=_ratio_to_grade(ratio),
            score=round(ratio, 4),
            source="sandbox",
            feedback=feedback[:500],
            test_results=results,
        )

    if qtype in _LLM_GRADED_TYPES:
        result = await judge_answer(
            str(payload.get("question", "")),
            str(payload.get("correct_answer", "")),
            user_answer,
            track=track,
        )
        if result is None:
            return GradeResult(question_id=qid, feedback="未能自动判分")
        return GradeResult(
            question_id=qid,
            grade=result.grade,
            source="llm_judge",
            feedback=result.feedback,
        )

    return GradeResult(question_id=qid, feedback="该题型暂不支持自动判分")


__all__ = ["GradeResult", "grade_answer"]
