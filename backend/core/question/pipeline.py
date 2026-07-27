"""QuizPipeline — 3-stage quiz generation（agents/question）。

Stages（代码编排，不靠 label）：
  explore → run_agent_loop 多轮（rag/web_search tool_calls）检索素材，产出探索摘要
  plan    → 单次 LLM 出 templates JSON（analysis + N 个，topic 不重复，6 题型合法）
  quiz    → 每题 run_agent_loop 出 QAPair JSON + 基础 schema 校验

的 explore→plan→quiz 三阶段，去 label：tool_calls loop 驱动 explore，
plan/quiz 用 run_agent_loop 单轮 + prompt 要求 JSON + 解析校验（provider 保证结构化
是 tool_calls 版取舍；强 repair / Tool Summarizer 留后续优化，避免侵入 run_agent_loop）。
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.agent.registry import get_tool_schemas
from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.llm.json_extract import extract_json_from_llm
from core.observability import log_flow
from core.pipeline_common import (
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
    with_common_prompt,
)
from core.prompt_loader import load_prompt_dict
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

_QUIZ_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "pipeline.yaml"

_VALID_TYPES = ("choice", "concept", "fill_in_blank", "short_answer", "written", "coding")
_VALID_DIFFICULTY = ("easy", "medium", "hard")

# Stage 3 并行生成上限：N 道题相互独立，用 Semaphore 限并发 LLM 调用，避免打满供应商速率限制
# （对标 research.DEFAULT_MAX_PARALLEL_TOPICS）
_MAX_PARALLEL_QUESTIONS = 3


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

        # 解析对话供应商 runtime + 通用上下文层（四条 pipeline 共享步骤，pipeline_common）
        rt = await resolve_profile_runtime(context.llm_profile_id, context.user_id)
        layers = await build_common_context_layers(context)

        # explore 可用工具：仅 rag/web_search（出题不需要 ask_user）
        explore_tools = [t for t in context.enabled_tools if t in ("rag", "web_search")]

        # ── Stage 1: explore — 多轮检索收集素材 ──────────────────────────
        _t_stage = _time.perf_counter()
        async with stream.stage("explore", source="quiz"):
            explore_ctx = replace(
                context,
                user_message=await describe_images(
                    context, f"出题要求：{requirement}", rt
                ),
                conversation_history=[],
                enabled_tools=explore_tools,
                mode="quiz",
            )
            explore_outcome = await run_agent_loop(
                context=explore_ctx,
                stream=stream,
                system_prompt=with_common_prompt((cfg.get("explore") or {}).get("system", ""), layers),
                tool_schemas=get_tool_schemas(explore_ctx),
                max_iterations=5,
                emit_terminal_events=False,
                client=rt.client,
                model=rt.text_model,
                binding=rt.binding,
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
                system_prompt=with_common_prompt(plan_system, layers),
                tool_schemas=None,
                max_iterations=1,
                emit_terminal_events=False,
                client=rt.client,
                model=rt.text_model,
                binding=rt.binding,
            )
            templates = _parse_templates(plan_outcome.final_text, count)
        log_flow("question.stage.plan",
                 elapsed_ms=int((_time.perf_counter() - _t_stage) * 1000),
                 templates=len(templates))

        # ── Stage 3: quiz — 每题并行生成 + schema 校验（单题容错）────────
        # 各题相互独立（第 N 题不需要第 N-1 题的结果），是 N 个完全独立的 run_agent_loop
        # （max_iterations=1）。用 asyncio.gather + Semaphore 并行跑，把 N 次串行 LLM 调用压成
        # ≈ 最慢一题的耗时（对标 research._drive_queue 的并行模式，扁平 DAG 的全量并行）。
        # 容错（M-8/M-9）语义不变：单题无论是 LLM 异常还是内容校验失败，都只记 error 事件、
        # 跳过该题，其余题照常生成。
        questions: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        _t_stage = _time.perf_counter()
        async with stream.stage("quiz", source="quiz"):
            quiz_system = (cfg.get("quiz") or {}).get("system", "")
            sem = asyncio.Semaphore(_MAX_PARALLEL_QUESTIONS)

            async def _gen(
                idx: int, tmpl: dict[str, Any]
            ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
                """生成单题（并发安全）：fork 隔离 context → run_agent_loop → 解析 + 校验。

                返回 (idx, question, error)：成功则 question 非 None，失败则 error 非 None。
                事件（quiz_question / quiz_question_error）按完成顺序直接发到主 stream——前端按
                到达顺序追加、不依赖 index 排序，故完成顺序乱序无害；questions / errors 列表在
                gather 后由主协程按 idx 排序汇总，保证结果载荷顺序确定（与串行时一致）。
                """
                async with sem:
                    # 并发隔离：replace 是浅拷贝，attachments / metadata 仍指向原 context 同一对象；
                    # _build_messages 会原地清空 doc 附件 base64（loop.py:96-97）并可能写 metadata，
                    # 多题共享同一份会让先跑的题清空后跑题的附件。深拷贝隔离（同 research._fork_for_block）。
                    quiz_ctx = _fork_for_quiz(
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
                    try:
                        quiz_outcome = await run_agent_loop(
                            context=quiz_ctx,
                            stream=stream,
                            system_prompt=with_common_prompt(quiz_system, layers),
                            tool_schemas=None,
                            max_iterations=1,
                            emit_terminal_events=False,
                            client=rt.client,
                            model=rt.text_model,
                            binding=rt.binding,
                        )
                    except Exception:
                        # M-8：单题 LLM 调用异常（超时 / 网络 / provider 抖动）不中断整批
                        logger.exception("quiz 第 %d 题生成异常", idx)
                        err = {
                            "index": idx - 1,
                            "question_id": tmpl.get("question_id", ""),
                            "topic": tmpl.get("topic", ""),
                            "reason": "generation_error",
                        }
                        await stream.emit({"type": "quiz_question_error", **err})
                        return (idx, None, err)

                    q = _parse_question(quiz_outcome.final_text, tmpl)
                    # M-9：内容级校验——空题干/空答案/choice 答案不命中 options 视为无效
                    if not _validate_question(q):
                        logger.warning("quiz 第 %d 题内容校验失败，跳过", idx)
                        err = {
                            "index": idx - 1,
                            "question_id": q.get("question_id", ""),
                            "topic": q.get("topic", ""),
                            "reason": "schema_invalid",
                        }
                        await stream.emit({"type": "quiz_question_error", **err})
                        return (idx, None, err)

                    await stream.emit({
                        "type": "quiz_question",
                        "index": idx - 1,
                        "question": q,
                    })
                    return (idx, q, None)

            # 并行跑全部题（_gen 内已全捕获异常，正常不会抛到 gather）
            raw = await asyncio.gather(
                *[_gen(idx, tmpl) for idx, tmpl in enumerate(templates, 1)]
            )
            # 按 idx 排序汇总 → questions / errors 顺序确定（与串行一致），与各题完成顺序无关
            for _idx, q, err in sorted(raw, key=lambda r: r[0]):
                if q is not None:
                    questions.append(q)
                elif err is not None:
                    errors.append(err)

        log_flow("question.stage.quiz",
                 elapsed_ms=int((_time.perf_counter() - _t_stage) * 1000),
                 generated=len(questions), errors=len(errors))
        log_flow("question.pipeline.complete",
                 elapsed_ms=int((_time.perf_counter() - _t_total) * 1000),
                 questions=len(questions))
        return {
            "questions": questions,
            "metadata": {
                "tools_used": explore_outcome.tools_used,
                "count_requested": count,
                "count_generated": len(questions),
                "count_failed": len(errors),
                "errors": errors,
                "stages": ["explore", "plan", "quiz"],
            },
        }


# ── JSON 解析 + schema 校验（tool_calls 版基础校验，强 repair 留后续）─────────

def _parse_templates(text: str, count: int) -> list[dict[str, Any]]:
    data = extract_json_from_llm(text)
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
    data = extract_json_from_llm(text)
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


def _validate_question(q: dict[str, Any]) -> bool:
    """M-9：对 _parse_question 的产物做内容级 schema 校验，判定该题是否有效可用。

    _parse_question 只做题型/options 字段的格式归一化（保已有行为与签名不变，不破坏旧测试）；
    真正的「内容是否可用」在这里判：
      - question（题干）、correct_answer（答案）必须非空——空题干/空答案无法作答。
      - choice 题：options 至少 2 项（单项无意义）、correct_answer 必须命中 options 的某个 key
        （否则答案指向不存在的选项，前端无法高亮正确项）。

    返回 False 的题由 pipeline 按 M-8 单题容错路径处理（标记 error，继续其它题）。
    """
    if not str(q.get("question", "")).strip():
        return False
    if not str(q.get("correct_answer", "")).strip():
        return False
    if q.get("question_type") == "choice":
        opts = q.get("options")
        if not isinstance(opts, dict) or len(opts) < 2:
            return False
        if str(q.get("correct_answer", "")).strip().upper() not in opts:
            return False
    return True


def _fork_for_quiz(context: UnifiedContext, **overrides: Any) -> UnifiedContext:
    """为并行 quiz 题目派生隔离 context（并发安全）。

    dataclasses.replace 是浅拷贝：attachments / metadata 仍指向原 context 的同一对象。
    run_agent_loop → _build_messages 会原地改 Attachment（doc 文件清空 base64，见 loop.py
    :96-97）并可能写 metadata；并行多题共享同一份时，先跑的题清空 base64 会让后跑的题丢掉
    附件内容。这里对 attachments + metadata 深拷贝，每题拿到独立副本互不影响（同 research
    的 _fork_for_block）。其余字段由 replace 正常处理；仅 Stage 3 并行入口用。
    """
    forked = replace(
        context,
        attachments=copy.deepcopy(context.attachments),
        metadata=copy.deepcopy(context.metadata),
    )
    if overrides:
        forked = replace(forked, **overrides)
    return forked


__all__ = ["QuizPipeline"]
