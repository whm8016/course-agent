"""
MainSolver — Plan -> ReAct -> Write pipeline (DeepTutor deep_solve port).
"""

from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from config import TEXT_MODEL

from .agents import PlannerAgent, SolverAgent, WriterAgent
from .memory import Scratchpad, Source
from .tool_runtime import SolveToolRuntime
from .utils.trace import derive_trace_metadata, new_call_id

logger = logging.getLogger(__name__)


def _parse_language(language: str | None) -> str:
    if not language:
        return "zh"
    lang = str(language).strip().lower()
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    return lang[:2] if len(lang) >= 2 else "zh"


class MainSolver:
    """Problem-Solving System Controller — Plan -> ReAct -> Write."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        language: str | None = None,
        kb_name: str | None = None,
        output_base_dir: str | None = None,
        enabled_tools: list[str] | None = None,
        disable_memory: bool = False,
        disable_planner_retrieve: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.disable_memory = disable_memory
        self.disable_planner_retrieve = disable_planner_retrieve
        self._max_tokens_override = max_tokens
        self._temperature_override = temperature

        lang = _parse_language(language)
        self.config: dict[str, Any] = dict(config or {})
        self.config.setdefault(
            "system",
            {"language": lang, "output_base_dir": output_base_dir or "./data/solve/output"},
        )
        self.config["system"]["language"] = lang
        if output_base_dir:
            self.config["system"]["output_base_dir"] = str(output_base_dir)

        _solve_defaults: dict[str, Any] = {
            "max_react_iterations": 10,
            "max_plan_steps": 10,
            "max_replans": 2,
            "observation_max_tokens": 2000,
            "enable_citations": True,
            "save_intermediate_results": True,
            "detailed_answer": True,
        }
        incoming_solve = self.config.get("solve")
        if isinstance(incoming_solve, dict):
            self.config["solve"] = {**_solve_defaults, **incoming_solve}
        else:
            self.config["solve"] = dict(_solve_defaults)
        self.config.setdefault("llm", {"model": model or TEXT_MODEL})

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self.kb_name = (kb_name or "").strip()

        self.logger = logging.getLogger("solve.MainSolver")
        self._trace_callback: Any = None
        self._conversation_context: str = ""

        self.planner_agent: PlannerAgent | None = None
        self.solver_agent: SolverAgent | None = None
        self.writer_agent: WriterAgent | None = None
        self.tool_runtime: SolveToolRuntime | None = None

        self._init_agents(
            enabled_tools=enabled_tools,
            lang=lang,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            model=model,
        )

    def set_trace_callback(self, callback: Any) -> None:
        self._trace_callback = callback
        for agent in (self.planner_agent, self.solver_agent, self.writer_agent):
            if agent is not None:
                agent.set_trace_callback(callback)

    async def _emit_trace_event(self, payload: dict[str, Any]) -> None:
        callback = self._trace_callback
        if callback is None:
            return
        try:
            result = callback(payload)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass

    def _init_agents(
        self,
        *,
        enabled_tools: list[str] | None,
        lang: str,
        api_key: str | None,
        base_url: str | None,
        api_version: str | None,
        model: str | None,
    ) -> None:
        self.tool_runtime = SolveToolRuntime(
            enabled_tools=enabled_tools or ["rag"],
            language=lang,
        )
        common: dict[str, Any] = dict(
            config=self.config,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            model=model,
            token_tracker=None,
            language=lang,
        )
        self.planner_agent = PlannerAgent(
            **common,
            tool_runtime=self.tool_runtime,
            enable_pre_retrieve=not self.disable_planner_retrieve,
        )
        self.solver_agent = SolverAgent(**common, tool_runtime=self.tool_runtime)
        self.writer_agent = WriterAgent(**common)

        if self._max_tokens_override is not None or self._temperature_override is not None:
            for agent in (self.planner_agent, self.solver_agent, self.writer_agent):
                if agent is None:
                    continue
                if self._max_tokens_override is not None:
                    agent.agent_config["max_tokens"] = self._max_tokens_override
                if self._temperature_override is not None:
                    agent.agent_config["temperature"] = self._temperature_override

        self.logger.info(
            "Agents initialised (lang=%s), tools: %s",
            lang,
            self.tool_runtime.tool_names,
        )

    async def solve(
        self,
        question: str,
        image_url: str | None = None,
        attachments: list[Any] | None = None,
        verbose: bool = True,
        detailed: bool | None = None,
        conversation_context: str = "",
    ) -> dict[str, Any]:
        del image_url, attachments

        if detailed is None:
            detailed = bool(self.config.get("solve", {}).get("detailed_answer", False))
        self._detailed = detailed
        self._conversation_context = conversation_context.strip()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = self.config.get("system", {}).get("output_base_dir", "./data/solve/output")
        output_dir = os.path.join(str(output_base), f"solve_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        self._log_section("Problem Solving Started")
        if verbose:
            self.logger.info(
                "Question: %s%s",
                question[:100],
                "..." if len(question) > 100 else "",
            )
            self.logger.info("Output: %s", output_dir)

        try:
            result = await self._run_pipeline(question, output_dir)
            result["metadata"] = {
                **result.get("metadata", {}),
                "mode": "plan_react_write",
                "timestamp": timestamp,
                "output_dir": output_dir,
            }
            self.logger.info("Problem solving completed")
            return result

        except Exception as exc:
            self.logger.error("Solving failed: %s", exc)
            self.logger.error(traceback.format_exc())
            raise

    async def _run_pipeline(self, question: str, output_dir: str) -> dict[str, Any]:
        solve_cfg = self.config.get("solve", {})
        max_react = int(solve_cfg.get("max_react_iterations", 5))
        max_replans = int(solve_cfg.get("max_replans", 2))

        scratchpad = Scratchpad.load_or_create(output_dir, question)

        # Phase 1: PLAN
        self._emit_progress("plan", {"status": "planning"})
        memory_ctx = ""
        if not self.disable_memory:
            memory_ctx = await self._get_planner_memory_context(question)

        plan = await self.planner_agent.process(  # type: ignore[union-attr]
            question=question,
            scratchpad=scratchpad,
            kb_name=self.kb_name,
            memory_context=memory_ctx,
        )
        scratchpad.set_plan(plan)
        scratchpad.save(output_dir)

        # Phase 2: SOLVE
        self._emit_progress("solve", {"status": "starting"})
        replan_count = 0
        safety_limit = (len(plan.steps) + max_replans) * (max_react + 1)
        iterations = 0

        while not scratchpad.is_all_completed():
            iterations += 1
            if iterations > safety_limit:
                self.logger.warning("Safety iteration limit reached")
                break

            step = scratchpad.get_next_pending_step()
            if step is None:
                break

            scratchpad.mark_step_status(step.id, "in_progress")
            step_memory_context = ""
            if not self.disable_memory:
                step_memory_context = await self._get_solver_memory_context(step.goal)

            step_index = (
                next(
                    (i + 1 for i, s in enumerate(scratchpad.plan.steps) if s.id == step.id),
                    0,
                )
                if scratchpad.plan
                else 0
            )
            self._emit_progress(
                "solve",
                {
                    "step_id": step.id,
                    "step_index": step_index,
                    "step_target": step.goal,
                },
            )

            for round_num in range(max_react):
                decision = await self.solver_agent.process(  # type: ignore[union-attr]
                    question=question,
                    current_step=step,
                    scratchpad=scratchpad,
                    memory_context=step_memory_context,
                    round_index=round_num + 1,
                )

                action = decision["action"]
                action_input = decision["action_input"]
                thought = decision["thought"]
                self_note = decision["self_note"]
                trace_meta = decision.get("_trace", {})

                if action == "done":
                    if self_note:
                        await self._emit_trace_event(
                            {
                                "event": "llm_observation",
                                "state": "complete",
                                "response": self_note,
                                **trace_meta,
                            }
                        )
                    scratchpad.add_entry(
                        step_id=step.id,
                        round_num=round_num,
                        thought=thought,
                        action="done",
                        action_input="",
                        observation="",
                        self_note=self_note,
                    )
                    scratchpad.mark_step_status(step.id, "completed")
                    scratchpad.save(output_dir)
                    break

                if action == "replan":
                    if self_note:
                        await self._emit_trace_event(
                            {
                                "event": "llm_observation",
                                "state": "complete",
                                "response": self_note,
                                **trace_meta,
                            }
                        )
                    replan_count += 1
                    scratchpad.add_entry(
                        step_id=step.id,
                        round_num=round_num,
                        thought=thought,
                        action="replan",
                        action_input=action_input,
                        observation="",
                        self_note=self_note,
                    )
                    if replan_count <= max_replans:
                        replan_memory = ""
                        if not self.disable_memory:
                            replan_memory = await self._get_planner_memory_context(question)
                        new_plan = await self.planner_agent.process(  # type: ignore[union-attr]
                            question=question,
                            scratchpad=scratchpad,
                            kb_name=self.kb_name,
                            replan=True,
                            memory_context=replan_memory,
                        )
                        scratchpad.update_plan(new_plan)
                        scratchpad.save(output_dir)
                    else:
                        scratchpad.mark_step_status(step.id, "completed")
                        scratchpad.save(output_dir)
                    break

                await self._emit_trace_event(
                    {
                        "event": "tool_call",
                        "state": "running",
                        "tool_name": action,
                        "tool_args": {"input": action_input},
                        **trace_meta,
                    }
                )
                observation, sources = await self._execute_tool(
                    action=action,
                    action_input=action_input,
                    output_dir=output_dir,
                    question=question,
                    scratchpad=scratchpad,
                    trace_meta=trace_meta,
                )
                await self._emit_trace_event(
                    {
                        "event": "tool_result",
                        "state": "complete",
                        "tool_name": action,
                        "result": observation,
                        "sources": [s.to_dict() for s in sources],
                        **trace_meta,
                    }
                )
                if self_note:
                    await self._emit_trace_event(
                        {
                            "event": "llm_observation",
                            "state": "complete",
                            "response": self_note,
                            **trace_meta,
                        }
                    )

                scratchpad.add_entry(
                    step_id=step.id,
                    round_num=round_num,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                    self_note=self_note,
                    sources=sources,
                )
                scratchpad.save(output_dir)
            else:
                self.logger.warning("Max ReAct iterations reached for %s", step.id)
                scratchpad.mark_step_status(step.id, "completed")
                scratchpad.save(output_dir)

        completed = scratchpad.get_completed_steps()
        total = len(scratchpad.plan.steps) if scratchpad.plan else 0

        # Phase 3: WRITE
        detailed = getattr(self, "_detailed", False)
        self._emit_progress("write", {"status": "writing", "detailed": detailed})

        language = self.config.get("system", {}).get("language", "zh")
        lang_code = _parse_language(str(language))

        preference = "" if self.disable_memory else self._get_user_preference()

        content_cb = getattr(self, "_content_callback", None)

        if detailed:
            final_answer = await self.writer_agent.process_iterative(  # type: ignore[union-attr]
                question=question,
                scratchpad=scratchpad,
                language=lang_code,
                preference=preference,
                on_content_chunk=content_cb,
            )
        else:
            final_answer = await self.writer_agent.process(  # type: ignore[union-attr]
                question=question,
                scratchpad=scratchpad,
                language=lang_code,
                preference=preference,
                on_content_chunk=content_cb,
            )

        answer_file = Path(output_dir) / "final_answer.md"
        answer_file.write_text(final_answer, encoding="utf-8")
        self.logger.info("Final answer saved: %s", answer_file)

        return {
            "question": question,
            "output_dir": output_dir,
            "final_answer": final_answer,
            "output_md": str(answer_file),
            "output_json": str(Path(output_dir) / "scratchpad.json"),
            "formatted_solution": final_answer,
            "citations": [s["id"] for s in scratchpad.get_all_sources()],
            "pipeline": "plan_react_write",
            "total_steps": total,
            "completed_steps": len(completed),
            "total_react_entries": len(scratchpad.entries),
            "plan_revisions": scratchpad.metadata.get("plan_revisions", 0),
            "metadata": {
                "total_steps": total,
                "completed_steps": len(completed),
                "plan_revisions": scratchpad.metadata.get("plan_revisions", 0),
            },
        }

    async def _execute_tool(
        self,
        action: str,
        action_input: str,
        output_dir: str,
        question: str = "",
        scratchpad: Scratchpad | None = None,
        trace_meta: dict[str, Any] | None = None,
    ) -> tuple[str, list[Source]]:
        obs_max = int(self.config.get("solve", {}).get("observation_max_tokens", 2000))
        sources: list[Source] = []
        retrieve_trace = self._build_retrieve_trace_meta(action, action_input, trace_meta)

        try:
            if self.tool_runtime is None:
                raise RuntimeError("Solve tool runtime is not initialised.")
            if action not in self.tool_runtime.valid_actions:
                observation = f"Unknown action: {action}"
                return observation, sources

            async def _event_sink(
                event_type: str,
                message: str = "",
                metadata: dict[str, Any] | None = None,
            ) -> None:
                if retrieve_trace is None or not message:
                    return
                await self._emit_trace_event(
                    {
                        "event": "tool_log",
                        "message": message,
                        **derive_trace_metadata(
                            retrieve_trace,
                            trace_kind=str(event_type or "tool_log"),
                            **(metadata or {}),
                        ),
                    }
                )

            if retrieve_trace is not None:
                await self._emit_trace_event(
                    {
                        "event": "tool_log",
                        "message": f"Query: {action_input}"
                        if action_input
                        else "Starting retrieval",
                        **derive_trace_metadata(
                            retrieve_trace,
                            trace_kind="call_status",
                            call_state="running",
                        ),
                    }
                )

            result = await self.tool_runtime.execute(
                action,
                action_input,
                kb_name=self.kb_name or None,
                output_dir=output_dir,
                reason_context=self._build_reason_context(question, scratchpad),
                model=self.solver_agent.get_model() if self.solver_agent else None,
                event_sink=_event_sink if retrieve_trace is not None else None,
            )
            if retrieve_trace is not None:
                await self._emit_trace_event(
                    {
                        "event": "tool_log",
                        "message": f"Retrieve complete ({len(result.content)} chars)",
                        **derive_trace_metadata(
                            retrieve_trace,
                            trace_kind="call_status",
                            call_state="complete",
                        ),
                    }
                )
            observation = self._format_tool_observation(
                action, result.content, result.metadata, obs_max
            )
            sources = self._convert_tool_sources(result.sources, result.metadata)
        except Exception as exc:
            observation = f"Tool error ({action}): {exc}"
            self.logger.warning("Tool error: %s", exc)
            if retrieve_trace is not None:
                await self._emit_trace_event(
                    {
                        "event": "tool_log",
                        "message": f"Retrieve failed: {exc}",
                        **derive_trace_metadata(
                            retrieve_trace,
                            trace_kind="call_status",
                            call_state="error",
                            error=str(exc),
                        ),
                    }
                )

        return observation, sources

    def _build_retrieve_trace_meta(
        self,
        action: str,
        action_input: str,
        trace_meta: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self.tool_runtime is None or not trace_meta:
            return None
        resolved_tool = self.tool_runtime.resolve_tool_name(action)
        if resolved_tool != "rag":
            return None
        return derive_trace_metadata(
            trace_meta,
            call_id=new_call_id("solve-retrieve"),
            label="Retrieve",
            call_kind="rag_retrieval",
            trace_role="retrieve",
            trace_group="retrieve",
            trace_id=f"{trace_meta.get('trace_id', 'solve')}-retrieve",
            query=action_input,
            tool_name=resolved_tool,
        )

    def _build_reason_context(
        self,
        question: str,
        scratchpad: Scratchpad | None,
    ) -> str:
        context_parts: list[str] = []
        if question:
            context_parts.append(f"Original question: {question}")
        if scratchpad:
            if scratchpad.plan:
                context_parts.append(f"Plan:\n{scratchpad._format_plan()}")
            completed = scratchpad.get_completed_steps()
            if completed:
                notes: list[str] = []
                for step in completed:
                    entries = scratchpad.get_entries_for_step(step.id)
                    step_notes = [e.self_note for e in entries if e.self_note]
                    if step_notes:
                        notes.append(f"[{step.id}] {step.goal}: {' '.join(step_notes)}")
                if notes:
                    context_parts.append("Knowledge from previous steps:\n" + "\n".join(notes))

        return "\n\n".join(context_parts)

    @staticmethod
    def _format_tool_observation(
        action: str,
        content: str,
        metadata: dict[str, Any],
        max_chars: int,
    ) -> str:
        text = (content or "").strip() or "(no results)"
        if action in {"code_execution", "code_execute", "run_code"}:
            code = (metadata.get("code") or "").strip()
            if code:
                text = f"Code:\n```python\n{code}\n```\n\n{text}"
        return text[: max_chars * 4]

    @staticmethod
    def _convert_tool_sources(
        tool_sources: list[dict[str, Any]] | None,
        metadata: dict[str, Any],
    ) -> list[Source]:
        sources: list[Source] = []
        for item in tool_sources or []:
            if not isinstance(item, dict):
                continue
            sources.append(
                Source(
                    type=str(item.get("type", "reference")),
                    file=item.get("file") or item.get("title") or item.get("kb_name"),
                    page=item.get("page"),
                    url=item.get("url"),
                    chunk_id=item.get("chunk_id") or item.get("query") or item.get("identifier"),
                )
            )

        for artifact_path in metadata.get("artifact_paths", []):
            sources.append(Source(type="code", file=Path(str(artifact_path)).name))
        return sources

    def _log_section(self, title: str) -> None:
        self.logger.info("%s", "=" * 60)
        self.logger.info("%s", title)
        self.logger.info("%s", "=" * 60)

    def _emit_progress(self, stage: str, progress: dict[str, Any]) -> None:
        cb = getattr(self, "_send_progress_update", None)
        if callable(cb):
            try:
                cb(stage, progress)
            except Exception:
                pass

    def _get_user_preference(self) -> str:
        return ""

    async def _get_planner_memory_context(self, question: str) -> str:
        _ = question
        return self._merge_memory_context("")

    async def _get_solver_memory_context(self, step_goal: str) -> str:
        _ = step_goal
        return self._merge_memory_context("", include_conversation=False)

    def _merge_memory_context(
        self,
        memory_context: str,
        include_conversation: bool = True,
    ) -> str:
        parts = []
        if include_conversation and self._conversation_context:
            parts.append(f"Conversation context:\n{self._conversation_context}")
        if memory_context:
            parts.append(memory_context)
        return "\n\n".join(part for part in parts if part)


__all__ = ["MainSolver"]
