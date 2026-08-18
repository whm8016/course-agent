"""QuizPipeline — 3-stage quiz generation（agents/question）。

Stages（代码编排，不靠 label）：
  explore → run_agent_loop 多轮（rag/web_search tool_calls）检索素材，产出探索摘要
  plan    → 单次 LLM 出 templates JSON（analysis + N 个，topic 不重复，6 题型合法）
  quiz    → 每题 run_agent_loop 出 QAPair JSON + 基础 schema 校验 + 独立 critic 校验轮

的 explore→plan→quiz 三阶段，去 label：tool_calls loop 驱动 explore，
plan/quiz 用 run_agent_loop 单轮 + prompt 要求 JSON + 解析校验（provider 保证结构化
是 tool_calls 版取舍；强 repair / Tool Summarizer 留后续优化，避免侵入 run_agent_loop）。

成本提示：quiz 阶段每题在生成轮之后多一次 critic 校验轮（生成与校验拆成两轮独立
调用，critic 看不到生成器的探索摘要与推理过程）。题间并行不变，单题串行等效延迟
约 +1 轮（critic 未过触发重试时最多 +2×2 轮，见 _CRITIC_MAX_RETRIES）。
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

# critic 失败后的重试上限：初始 1 轮生成 + 最多 2 次重试；仍失败走 M-8/M-9 同款
# 单题容错（reason="critic_failed"，其余题照常）
_CRITIC_MAX_RETRIES = 2


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

        # explore 可用工具：rag/web_search 检索素材 + ask_user（出题要求不清时让学生澄清，仅 WS 入口可暂停）
        explore_tools = [t for t in context.enabled_tools if t in ("rag", "web_search")] + ["ask_user"]

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

        # ── 薄弱点清单（出题定向）：explore 后、plan 前查一次 ────────────────
        # user+course 齐备才查（匿名/无课场景保持旧行为）。fail-safe：查询失败按
        # 无薄弱点继续，不阻塞出题主流程（与 pipeline_common 门控同款容错口径）。
        weak_topics: list[dict[str, Any]] = []
        if context.user_id and context.course_id:
            try:
                from core.db.database import AsyncSessionLocal
                from core.memory.mastery import get_weak_topics

                async with AsyncSessionLocal() as db:
                    weak_topics = await get_weak_topics(
                        db, context.user_id, context.course_id
                    )
            except Exception:
                logger.exception("[quiz] get_weak_topics 失败（按无薄弱点继续）")
                weak_topics = []

        # ── Stage 2: plan — 单次 LLM 出 templates 蓝图 ───────────────────
        _t_stage = _time.perf_counter()
        async with stream.stage("plan", source="quiz"):
            plan_system = (cfg.get("plan") or {}).get("system", "").format(
                count=count,
                valid_types="/".join(_VALID_TYPES),
            )
            weak_block = ""
            if weak_topics:
                lines = ["该生薄弱知识点（按风险排序，出题时优先覆盖）："]
                for w in weak_topics:
                    lines.append(
                        f"- {w['label']}（风险{w['risk']:.2f}，"
                        f"{w['observation_count']}次观测，topic_id={w['topic_id']}）"
                    )
                weak_block = "\n".join(lines) + "\n\n"
            plan_ctx = replace(
                context,
                user_message=(
                    f"出题要求：{requirement}\n\n"
                    f"{weak_block}"
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
        # 蓝图命中薄弱点的题数（评测对账用；weak_ids 空时恒 0）
        weak_ids = {w["topic_id"] for w in weak_topics}
        weak_targeted = sum(1 for t in templates if t.get("target_topic_id") in weak_ids)
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
            critic_system = (cfg.get("critic") or {}).get("system", "")
            sem = asyncio.Semaphore(_MAX_PARALLEL_QUESTIONS)
            # critic 计数（metadata/评测读）：最终过 critic / 触发过重试 / 重试用尽失败
            critic_counts = {"passed": 0, "regenerated": 0, "failed": 0}

            async def _gen(
                idx: int, tmpl: dict[str, Any]
            ) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
                """生成单题（并发安全）：fork 隔离 context → run_agent_loop → 解析 + 校验 + critic。

                返回 (idx, question, error)：成功则 question 非 None，失败则 error 非 None。
                事件（quiz_question / quiz_question_error）按完成顺序直接发到主 stream——前端按
                到达顺序追加、不依赖 index 排序，故完成顺序乱序无害；questions / errors 列表在
                gather 后由主协程按 idx 排序汇总，保证结果载荷顺序确定（与串行时一致）。
                """
                async with sem:
                    # 并发隔离：replace 是浅拷贝，attachments / metadata 仍指向原 context 同一对象；
                    # _build_messages 会原地清空 doc 附件 base64（loop.py:96-97）并可能写 metadata，
                    # 多题共享同一份会让先跑的题清空后跑题的附件。深拷贝隔离（同 research._fork_for_block）。
                    base_prompt = (
                        f"题目蓝图：{json.dumps(tmpl, ensure_ascii=False)}\n\n"
                        f"探索摘要：\n{exploration_trace}\n\n"
                        f"请生成第 {idx} 道题（JSON）。"
                    )
                    quiz_ctx = _fork_for_quiz(
                        context,
                        user_message=base_prompt,
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

                    # ── critic 校验轮：schema 关过后独立复核（看不到生成过程）──
                    # 未过 → issues 拼回生成器重试（最多 _CRITIC_MAX_RETRIES 次）；
                    # 重试产物重新走 schema 校验再进 critic。仍失败走 M-8/M-9 同款
                    # 单题容错（reason="critic_failed"），其余题照常。
                    critic_attempts = 0
                    while True:
                        issues = await _run_critic(
                            q, critic_system, context, layers, rt, stream
                        )
                        if not issues:
                            break
                        if critic_attempts >= _CRITIC_MAX_RETRIES:
                            logger.warning(
                                "quiz 第 %d 题 critic 未通过（重试用尽），跳过", idx
                            )
                            err = {
                                "index": idx - 1,
                                "question_id": q.get("question_id", ""),
                                "topic": q.get("topic", ""),
                                "reason": "critic_failed",
                                "critic_issues": issues[:5],
                            }
                            critic_counts["failed"] += 1
                            await stream.emit({"type": "quiz_question_error", **err})
                            return (idx, None, err)
                        critic_attempts += 1
                        critic_counts["regenerated"] += 1
                        retry_ctx = _fork_for_quiz(
                            context,
                            user_message=(
                                f"{base_prompt}\n\n"
                                f"上一版未通过校验，问题：{'；'.join(issues)}。"
                                f"请修正后重新给出完整 JSON。"
                            ),
                            conversation_history=[],
                            enabled_tools=[],
                            mode="quiz",
                        )
                        try:
                            quiz_outcome = await run_agent_loop(
                                context=retry_ctx,
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
                            # 重试轮的 M-8 同款：异常不中断整批
                            logger.exception("quiz 第 %d 题重试生成异常", idx)
                            err = {
                                "index": idx - 1,
                                "question_id": tmpl.get("question_id", ""),
                                "topic": tmpl.get("topic", ""),
                                "reason": "generation_error",
                            }
                            await stream.emit({"type": "quiz_question_error", **err})
                            return (idx, None, err)

                        q = _parse_question(quiz_outcome.final_text, tmpl)
                        if not _validate_question(q):
                            # 重试版反而格式坏了：M-9 语义（schema_invalid），非 critic 账
                            logger.warning("quiz 第 %d 题重试后内容校验失败，跳过", idx)
                            err = {
                                "index": idx - 1,
                                "question_id": q.get("question_id", ""),
                                "topic": q.get("topic", ""),
                                "reason": "schema_invalid",
                            }
                            await stream.emit({"type": "quiz_question_error", **err})
                            return (idx, None, err)
                    critic_counts["passed"] += 1

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
                # 出题定向薄弱点（评测读）：注入清单的条数 / 蓝图命中清单的题数
                "weak_topics_considered": len(weak_topics),
                "weak_topics_targeted": weak_targeted,
                # critic 校验轮计数（评测读）：最终过 / 触发过重试 / 重试用尽失败
                "critic_passed": critic_counts["passed"],
                "critic_regenerated": critic_counts["regenerated"],
                "critic_failed": critic_counts["failed"],
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
            # 定向命中（出题定向薄弱点）：plan 给了就透传，没给空串（旧测试/旧行为兼容）
            "target_topic_id": str(item.get("target_topic_id") or "").strip(),
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
        # 蓝图定向命中的主题 id（出题定向薄弱点）：透传给前端/评测对账，无则空串
        "target_topic_id": template.get("target_topic_id", ""),
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


async def _run_critic(
    q: dict[str, Any],
    critic_system: str,
    context: UnifiedContext,
    layers,
    rt,
    stream: StreamBus,
) -> list[str]:
    """独立 critic 校验轮：只给题目本身（看不到探索摘要/生成过程），返回问题清单。

    两路检查：
      1. coding 题沙箱（程序化，零 LLM）：correct_answer 的围栏代码块逐块执行，
         跑不过计入 issues（详见 sandbox.py 风险披露）。explanation 里的代码常是
         故意的错误示范——跑了必误杀，不吃。子进程异常（编码/杀软锁临时目录等）
         也计 issue 走重试，不穿透 gather 炸整批（M-8 单题容错口径）。
      2. LLM critic（run_agent_loop 单轮）：事实正确性 / 蓝图对齐 / 干扰项唯一性，
         输出 {"pass": bool, "issues": [...]}。

    fail-open：critic LLM 调用异常 / 输出不可解析时按「仅沙箱结论」放行——
    生成侧已有 schema 校验（_validate_question）兜底，校验器不可用不误杀好题。
    边界：已解析出 pass=false 但 issues 缺失/为空**不属于** fail-open——模型明确
    判否却拿不到理由，兜底一条 issue 触发重试，不静默放行。
    """
    issues: list[str] = []
    if q.get("question_type") == "coding":
        from core.question.sandbox import extract_code_blocks, run_code

        blocks = extract_code_blocks(q.get("correct_answer", "") or "")
        for i, block in enumerate(blocks, 1):
            try:
                # to_thread：subprocess.run 是同步阻塞，直跑事件循环会让同 worker
                # 的全部并发会话冻结到超时（生产 gunicorn -w4 每 worker 单 loop）
                ok, detail = await asyncio.to_thread(run_code, block)
            except Exception as exc:
                logger.warning("[quiz] 沙箱执行异常（计 issue 走重试）", exc_info=True)
                ok, detail = False, f"sandbox_error: {exc}"
            if not ok:
                issues.append(f"第{i}个代码块执行失败（{detail}）")

    payload = json.dumps(
        {
            "question_type": q.get("question_type", ""),
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", ""),
            "question": q.get("question", ""),
            "options": q.get("options"),
            "correct_answer": q.get("correct_answer", ""),
            "explanation": q.get("explanation", ""),
        },
        ensure_ascii=False,
    )
    critic_ctx = _fork_for_quiz(
        context,
        user_message=f"待校验题目：\n{payload}\n\n请校验并输出 JSON。",
        conversation_history=[],
        enabled_tools=[],
        mode="quiz",
    )
    try:
        outcome = await run_agent_loop(
            context=critic_ctx,
            stream=stream,
            system_prompt=with_common_prompt(critic_system, layers),
            tool_schemas=None,
            max_iterations=1,
            emit_terminal_events=False,
            client=rt.client,
            model=rt.text_model,
            binding=rt.binding,
        )
    except Exception:
        logger.warning("[quiz] critic 调用失败（fail-open，仅沙箱结论生效）", exc_info=True)
        return issues

    data = extract_json_from_llm(outcome.final_text)
    if not isinstance(data, dict) or "pass" not in data:
        logger.warning(
            "[quiz] critic 输出不可解析（fail-open 放行）：%s",
            (outcome.final_text or "")[:200],
        )
        return issues
    if data.get("pass") is not True:
        raw = data.get("issues")
        if isinstance(raw, list):
            issues.extend(str(x) for x in raw if str(x).strip())
        if not issues:
            # 判否但没给理由：不算 fail-open（那是给"不知道"的），兜底 issue 触发重试
            logger.warning("[quiz] critic 判 pass=false 但无 issues，按兜底 issue 重试")
            issues.append("critic 判定未通过，但未给出具体问题清单")
    return issues


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
