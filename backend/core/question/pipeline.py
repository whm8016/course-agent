"""QuizPipeline — 3-stage quiz generation（对标 DeepTutor agents/question）。

Stages（代码编排，不靠 label）：
  explore → run_agent_loop 多轮（rag/web_search tool_calls）检索素材，产出探索摘要
  plan    → 单次 LLM 出 templates JSON（analysis + N 个，topic 不重复，6 题型合法）
  quiz    → 每题 run_agent_loop 出 QAPair JSON + 基础 schema 校验

对标 DeepTutor 的 explore→plan→quiz 三阶段，去 label：tool_calls loop 驱动 explore，
plan/quiz 用 run_agent_loop 单轮 + prompt 要求 JSON + 解析校验（provider 保证结构化
是 tool_calls 版取舍；强 repair / Tool Summarizer 留后续优化，避免侵入 run_agent_loop）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.agentic.loop import _get_tool_schemas, run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.prompt_loader import load_prompt_dict
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

_QUIZ_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "pipeline.yaml"

_VALID_TYPES = ("choice", "concept", "fill_in_blank", "short_answer", "written", "coding")
_VALID_DIFFICULTY = ("easy", "medium", "hard")


class QuizPipeline:
    """3-stage quiz pipeline: explore → plan → quiz."""

    async def run(
        self,
        requirement: str,
        context: UnifiedContext,
        stream: StreamBus,
        count: int = 1,
    ) -> dict[str, Any]:
        cfg = load_prompt_dict(_QUIZ_PROMPT_PATH)
        count = max(1, int(count or 1))
        log_flow("question.pipeline.start", count=count, requirement=requirement[:120])
        import time as _time
        _t_total = _time.perf_counter()

        # explore 可用工具：仅 rag/web_search（出题不需要 ask_user）
        explore_tools = [t for t in context.enabled_tools if t in ("rag", "web_search")]

        # ── Stage 1: explore — 多轮检索收集素材 ──────────────────────────
        _t_stage = _time.perf_counter()
        async with stream.stage("explore", source="quiz"):
            explore_ctx = replace(
                context,
                user_message=f"出题要求：{requirement}",
                conversation_history=[],
                enabled_tools=explore_tools,
                mode="quiz",
            )
            explore_outcome = await run_agent_loop(
                context=explore_ctx,
                stream=stream,
                system_prompt=(cfg.get("explore") or {}).get("system", ""),
                tool_schemas=_get_tool_schemas(explore_ctx),
                max_iterations=5,
            )
            exploration_trace = explore_outcome.final_text
        log_flow("question.stage.explore",
                 elapsed_ms=int((_time.perf_counter() - _t_stage) * 1000),
                 tools_used=explore_outcome.tools_used)

        # ── Stage 2: plan — 单次 LLM 出 templates 蓝图 ───────────────────
        _t_stage = _time.perf_counter()
        async with stream.stage("plan", source="quiz"):
            plan_system = (cfg.get("plan") or {}).get("system", "").format(
                count=count,
                valid_types="/".join(_VALID_TYPES),
            )
            plan_ctx = replace(
                context,
                user_message=(
                    f"出题要求：{requirement}\n\n"
                    f"探索摘要：\n{exploration_trace}\n\n"
                    f"请规划 {count} 道题目的蓝图（JSON）。"
                ),
                conversation_history=[],
                enabled_tools=[],
                mode="quiz",
            )
            plan_outcome = await run_agent_loop(
                context=plan_ctx,
                stream=stream,
                system_prompt=plan_system,
                tool_schemas=None,
                max_iterations=1,
            )
            templates = _parse_templates(plan_outcome.final_text, count)
        log_flow("question.stage.plan",
                 elapsed_ms=int((_time.perf_counter() - _t_stage) * 1000),
                 templates=len(templates))

        # ── Stage 3: quiz — 每题生成 + schema 校验 ───────────────────────
        questions: list[dict[str, Any]] = []
        _t_stage = _time.perf_counter()
        async with stream.stage("quiz", source="quiz"):
            quiz_system = (cfg.get("quiz") or {}).get("system", "")
            for idx, tmpl in enumerate(templates, 1):
                quiz_ctx = replace(
                    context,
                    user_message=(
                        f"题目蓝图：{json.dumps(tmpl, ensure_ascii=False)}\n\n"
                        f"探索摘要：\n{exploration_trace}\n\n"
                        f"请生成第 {idx} 道题（JSON）。"
                    ),
                    conversation_history=[],
                    enabled_tools=[],
                    mode="quiz",
                )
                quiz_outcome = await run_agent_loop(
                    context=quiz_ctx,
                    stream=stream,
                    system_prompt=quiz_system,
                    tool_schemas=None,
                    max_iterations=1,
                )
                q = _parse_question(quiz_outcome.final_text, tmpl)
                questions.append(q)
                await stream.emit({
                    "type": "quiz_question",
                    "index": idx - 1,
                    "question": q,
                })

        log_flow("question.stage.quiz",
                 elapsed_ms=int((_time.perf_counter() - _t_stage) * 1000),
                 generated=len(questions))
        log_flow("question.pipeline.complete",
                 elapsed_ms=int((_time.perf_counter() - _t_total) * 1000),
                 questions=len(questions))
        return {
            "questions": questions,
            "metadata": {
                "tools_used": explore_outcome.tools_used,
                "count_requested": count,
                "count_generated": len(questions),
                "stages": ["explore", "plan", "quiz"],
            },
        }


# ── JSON 解析 + schema 校验（tool_calls 版基础校验，强 repair 留后续）─────────

def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出提取 JSON 对象，容忍 markdown fence / 前后噪声。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        return {}
    try:
        data, _end = json.JSONDecoder().raw_decode(cleaned, start)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_templates(text: str, count: int) -> list[dict[str, Any]]:
    data = _extract_json(text)
    raw = data.get("templates", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []
    templates: list[dict[str, Any]] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        qt = str(item.get("question_type", "written")).strip().lower()
        qt = qt if qt in _VALID_TYPES else "written"
        diff = str(item.get("difficulty", "medium")).strip().lower()
        diff = diff if diff in _VALID_DIFFICULTY else "medium"
        templates.append({
            "question_id": str(item.get("question_id") or f"q_{i}"),
            "topic": topic,
            "question_type": qt,
            "difficulty": diff,
        })
        if len(templates) >= count:
            break
    return templates


def _parse_question(text: str, template: dict[str, Any]) -> dict[str, Any]:
    data = _extract_json(text)
    expected_type = str(template.get("question_type", "written")).strip().lower()
    expected_type = expected_type if expected_type in _VALID_TYPES else "written"
    qt = str(data.get("question_type", expected_type)).strip().lower()
    qt = qt if qt in _VALID_TYPES else expected_type

    q: dict[str, Any] = {
        "question_id": template.get("question_id", ""),
        "question_type": qt,
        "question": str(data.get("question", "")).strip(),
        "correct_answer": str(data.get("correct_answer", "")).strip(),
        "explanation": str(data.get("explanation", "")).strip(),
        "options": None,
        "difficulty": template.get("difficulty", ""),
        "topic": template.get("topic", ""),
    }

    if qt == "choice":
        raw_opts = data.get("options")
        clean: dict[str, str] = {}
        if isinstance(raw_opts, dict):
            for k, v in raw_opts.items():
                key = str(k).strip().upper()[:1]
                if key in {"A", "B", "C", "D"} and str(v).strip():
                    clean[key] = str(v).strip()
        q["options"] = clean if clean else None

    return q


__all__ = ["QuizPipeline"]
