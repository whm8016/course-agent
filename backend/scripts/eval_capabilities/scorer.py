"""四能力判分 scorer（Inspect AI）。

设计原则（调研落地）：
  - 纯程序维度（quiz schema 校验、solve 状态机轨迹）用代码断言，不浪费 LLM judge——
    确定性逻辑用确定性判断。
  - 主观维度用 LLM judge（G-Eval 式 CoT：rubric 注入 + 先点评后给分）。
  - 异家族 ensemble（_ensemble_score）：多个不同家族 judge 各打分取均值，治自增强偏置
    （arXiv:2410.02736）与 agreement trap（同家族 91% 一致却全错）。
  - 分数解析 _extract_score_01 兼容 0-100 / X/10 / 小数，judge 输出格式不严也能抽。

LLM judge 部分真实跑需 OPENAI_* key/base_url（config 已从项目 LLM__* 注入）。mock 测试
monkeypatch _llm_judge 即可不依赖真实 LLM 验证判分/聚合逻辑。
"""
from __future__ import annotations

import json
import re
from typing import Any

from inspect_ai.scorer import Score, Target, mean, model_graded_qa, scorer
from inspect_ai.solver import TaskState

from . import config


# ── LLM judge 公共能力 ──────────────────────────────────────────────────────

async def _llm_judge(prompt: str, model_spec: str) -> str:
    """调 judge 模型生成，返回 completion 文本。异常→空串（调用方降级，不崩）。"""
    from inspect_ai.model import get_model

    try:
        model = get_model(model_spec)
        result = await model.generate(prompt)
        return (getattr(result, "completion", "") or "").strip()
    except Exception:
        return ""


def _extract_score_01(text: str) -> float:
    """从 judge 文本提取 0-1 分数，兼容 X/100、X/10、0-1 小数、纯整数。"""
    if not text:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*100", text)
    if m:
        return min(1.0, float(m.group(1)) / 100)
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10\b", text)
    if m:
        return min(1.0, float(m.group(1)) / 10)
    m = re.search(r"(\d+\.\d+)", text)
    if m:
        v = float(m.group(1))
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        v = float(m.group(1))
        return v / 100 if v > 1 else v
    return 0.0


async def _ensemble_score(prompt: str) -> tuple[float, str]:
    """异家族 ensemble：每个 judge 独立打分取均值。返回 (均值, 说明)。"""
    scores: list[float] = []
    for spec in config.JUDGE_MODELS:
        txt = await _llm_judge(prompt, spec)
        if txt:
            scores.append(_extract_score_01(txt))
    if not scores:
        return 0.0, "judge 无有效输出（可能判崩，已降级为 0）"
    return sum(scores) / len(scores), f"{len(scores)} judge 均值"


# ── chat：model_graded_qa（rubric 写进 Sample.target，model=list 异家族 ensemble）──

def chat_scorer():
    """课程问答：用 inspect 内置 model_graded_qa，target 当 rubric(criterion)。

    faithfulness（忠于检索材料）依赖检索 context，Phase 1 先评 answer 质量(accuracy)，
    faithfulness 可后续从 _trace 抽检索内容加自定义 judge。
    """
    return model_graded_qa(model=config.JUDGE_MODELS)


# ── quiz：结构有效性(纯程序) + 出题质量(LLM 7维 rubric) ──────────────────────

def _valid_quiz_item(item: Any) -> bool:
    """零成本 schema 校验：题目有题干 + 答案。字段名宽松匹配 core/question 真实 payload。"""
    if not isinstance(item, dict):
        return False
    has_q = any(k in item for k in ("question", "stem", "题干", "content", "prompt"))
    has_a = any(k in item for k in ("answer", "correct", "answer_text", "答案", "solution"))
    return has_q and has_a


@scorer(metrics=[mean()])
def quiz_validity():
    async def score(state: TaskState, target: Target) -> Score:
        items = state.metadata.get("_quiz") or []
        if not items:
            return Score(value=0.0, explanation="未生成题目", metadata={"quiz_count": 0})
        ok = sum(1 for it in items if _valid_quiz_item(it))
        return Score(
            value=ok / len(items),
            explanation=f"{ok}/{len(items)} 题结构有效",
            metadata={"quiz_count": len(items)},
        )

    return score


def _quiz_quality_prompt(items: list[dict], question: str) -> str:
    items_str = json.dumps(items, ensure_ascii=False, indent=2)[:4000]
    return (
        "你是课程出题质量评审。对自动生成的题目按 7 维评分（EdTech MCQ rubric）：\n"
        "1 答案正确性 2 干扰项质量(错误选项是否有迷惑性) 3 主题相关性 4 表达清晰度"
        " 5 难度匹配 6 知识覆盖 7 可答性(能否从课程材料答出、不超纲)\n\n"
        f"出题要求：{question}\n生成的题目：\n{items_str}\n\n"
        "请逐维度简评，最后一行输出：总分：XX/100"
    )


@scorer(metrics=[mean()])
def quiz_quality():
    async def score(state: TaskState, target: Target) -> Score:
        items = state.metadata.get("_quiz") or []
        if not items:
            return Score(value=0.0, explanation="未生成题目")
        val, expl = await _ensemble_score(_quiz_quality_prompt(items, state.input_text))
        return Score(value=val, explanation=f"出题质量 7维 rubric：{expl}")

    return score


# ── deep_solve：状态机轨迹合法性(纯程序) + 答案正确性(LLM) ──────────────────

def _trajectory_legal(trace: list[dict]) -> bool:
    """状态机脊柱合法性（纯程序断言）：solve 应出现 solve_plan → solve_finish_step 推进序列。

    trace 是 event.to_dict() 列表；工具调用在 type=tool_call 事件，工具名在 tool/name 字段。
    状态机走错一步全盘错，这部分用代码断言而非 LLM（确定性逻辑不浪费 judge，τ-bench 思路）。
    """
    tool_seq = []
    for ev in trace or []:
        if ev.get("type") == "tool_call":
            tool_seq.append(str(ev.get("tool") or ev.get("name", "")))
    has_plan = any("solve_plan" in t or ("plan" in t and "replan" not in t) for t in tool_seq)
    has_finish = any("finish_step" in t for t in tool_seq)
    return has_plan and has_finish


@scorer(metrics=[mean()])
def solve_trajectory():
    async def score(state: TaskState, target: Target) -> Score:
        trace = state.metadata.get("_trace") or []
        legal = _trajectory_legal(trace)
        return Score(
            value=1.0 if legal else 0.0,
            explanation="solve 状态机：solve_plan→solve_finish_step 合法序列",
            metadata={"tool_count": len(trace)},
        )

    return score


def _solve_answer_prompt(answer: str, question: str, target: Target) -> str:
    return (
        "你是课程解题评审。判断最终答案是否正确、解题是否完整。\n"
        f"题目：{question}\n评分标准：{target.text}\n学生答案：{answer[:4000]}\n\n"
        "若答案正确且完整给高分，错误或遗漏给低分。最后一行输出：分数：XX/100"
    )


@scorer(metrics=[mean()])
def solve_answer():
    async def score(state: TaskState, target: Target) -> Score:
        val, expl = await _ensemble_score(
            _solve_answer_prompt(state.output.completion, state.input_text, target)
        )
        return Score(value=val, explanation=f"最终答案正确性：{expl}")

    return score


# ── deep_research：报告 RACE 4维 + 引用真实性 FACT ───────────────────────────

def _race_prompt(report: str, question: str, target: Target) -> str:
    return (
        "你是研究报告评审（RACE 裁剪4维）。逐维评分：\n"
        "1 覆盖广度与深度(Comprehensiveness) 2 分析洞见(Insight) "
        "3 是否紧扣主题(Relevance) 4 结构清晰(Structure)\n"
        f"研究要求：{question}\n评分标准：{target.text}\n报告：\n{report[:6000]}\n\n"
        "逐维点评，最后一行输出：总分：XX/100"
    )


@scorer(metrics=[mean()])
def research_race():
    async def score(state: TaskState, target: Target) -> Score:
        val, expl = await _ensemble_score(
            _race_prompt(state.output.completion, state.input_text, target)
        )
        return Score(value=val, explanation=f"报告 RACE 4维：{expl}")

    return score


def _fact_prompt(report: str, trace: list[dict]) -> str:
    # 从 trace 抽检索/引用来源摘要（tool_result / result 事件）
    sources = [
        str(ev.get("content") or ev.get("result") or "")
        for ev in (trace or [])
        if ev.get("type") in ("tool_result", "result") and (ev.get("content") or ev.get("result"))
    ]
    src_str = "\n".join(sources)[:6000] or "(未捕获到检索来源)"
    return (
        "你是引用真实性评审(FACT)。判断报告中的关键论断是否被其引用的来源真实支撑。\n"
        f"可用来源/检索摘要：\n{src_str}\n\n报告：\n{report[:6000]}\n\n"
        "逐条核验关键论断，最后一行输出：真实率：XX/100"
    )


@scorer(metrics=[mean()])
def research_fact():
    async def score(state: TaskState, target: Target) -> Score:
        trace = state.metadata.get("_trace") or []
        val, expl = await _ensemble_score(_fact_prompt(state.output.completion, trace))
        return Score(value=val, explanation=f"引用真实性 FACT：{expl}")

    return score
