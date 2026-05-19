"""ResearchAgent - Research Agent (faithful port from DeepTutor).

Adaptation: ``get_tool_registry()`` is replaced by a local ``ToolRegistry``
instance passed via config, keeping the same interface.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable
from string import Template
from typing import Any

from ..base_agent import BaseAgent
from ..data_structures import DynamicTopicQueue, TopicBlock
from ..tool_registry import ToolRegistry
from ..trace import build_trace_metadata, new_call_id
from ..utils.json_utils import extract_json_from_text


class ResearchAgent(BaseAgent):
    """Research Agent"""

    _MODE_TO_STYLE = {
        "notes": "study_notes",
        "report": "report",
        "comparison": "comparison",
        "learning_path": "learning_path",
    }

    @staticmethod
    def _build_trace_meta(
        *,
        label: str,
        iteration: int,
        block_id: str = "",
        trace_role: str = "thought",
    ) -> dict[str, Any]:
        return build_trace_metadata(
            call_id=new_call_id("research-step"),
            phase="researching",
            label=label,
            call_kind="llm_reasoning",
            trace_role=trace_role,
            trace_kind="llm_reasoning",
            iteration=iteration,
            block_id=block_id or None,
            trace_group="research_round" if block_id else None,
        )

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
    ):
        language = config.get("system", {}).get("language", "zh")
        super().__init__(
            module_name="research",
            agent_name="research_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )
        self.researching_config = config.get("researching", {})
        self.max_iterations = self.researching_config.get("max_iterations", 5)
        self.iteration_mode = self.researching_config.get("iteration_mode", "fixed")
        self.enable_rag = self.researching_config.get("enable_rag", True)

        tools_web_search_enabled = (
            config.get("tools", {}).get("web_search", {}).get("enabled", True)
        )
        research_web_search_enabled = self.researching_config.get("enable_web_search", False)
        self.enable_web_search = tools_web_search_enabled and research_web_search_enabled
        self.enable_paper_search = self.researching_config.get("enable_paper_search", False)
        self.enable_run_code = self.researching_config.get("enable_run_code", True)
        self.enabled_tools = self.researching_config.get("enabled_tools", ["rag"])

        kb_name = config.get("rag", {}).get("kb_name")
        self._tool_registry = ToolRegistry(config=config, kb_name=kb_name)

        intent_mode = str(config.get("intent", {}).get("mode", "") or "")
        reporting_style = str(config.get("reporting", {}).get("style", "") or "")
        self._research_style = reporting_style or self._MODE_TO_STYLE.get(intent_mode, "report")

    @staticmethod
    def _convert_to_template_format(template_str: str) -> str:
        return re.sub(r"\{(\w+)\}", r"$\1", template_str)

    def _safe_format(self, template_str: str, **kwargs: Any) -> str:
        converted = self._convert_to_template_format(template_str)
        return Template(converted).safe_substitute(**kwargs)

    def _get_mode_contract(self, stage: str) -> str:
        return (
            self.get_prompt("mode_contracts", f"{self._research_style}_{stage}", "") or ""
        ).strip()

    def _get_enabled_prompt_tools(self) -> list[str]:
        tool_names: list[str] = []
        if self.enable_rag:
            tool_names.append("rag")
        if self.enable_paper_search:
            tool_names.append("paper_search")
        if self.enable_web_search:
            tool_names.append("web_search")
        if self.enable_run_code:
            tool_names.append("code_execution")
        deduped: list[str] = []
        for name in tool_names:
            if name not in deduped:
                deduped.append(name)
        return deduped

    def _is_llm_only_mode(self) -> bool:
        return not self._get_enabled_prompt_tools()

    def _generate_available_tools_text(self) -> str:
        tool_names = self._get_enabled_prompt_tools()
        if not tool_names:
            return "(no tools available)"
        return self._tool_registry.build_prompt_text(tool_names, format="aliases", language=self.language)

    def _generate_tool_phase_guidance(self) -> str:
        tool_names = self._get_enabled_prompt_tools()
        guidance = self._tool_registry.build_prompt_text(tool_names, format="phased", language=self.language)
        if guidance:
            return guidance
        if self.language == "zh":
            return "当前没有额外工具可用，请围绕现有知识继续分析。"
        return "No extra tools are currently enabled; continue reasoning with the knowledge already gathered."

    def _generate_research_depth_guidance(self, iteration: int, used_tools: list[str]) -> str:
        early_threshold = max(2, self.max_iterations // 3)
        middle_threshold = max(4, self.max_iterations * 2 // 3)

        if iteration <= early_threshold:
            phase_desc = f"Early Stage (Iteration 1-{early_threshold})"
            guidance = "Focus on building foundational knowledge using RAG/knowledge base tools."
        elif iteration <= middle_threshold:
            phase_desc = f"Middle Stage (Iteration {early_threshold + 1}-{middle_threshold})"
            if self.enable_paper_search or self.enable_web_search:
                guidance = "Consider using Paper/Web search to add academic depth."
            else:
                guidance = "Deepen knowledge coverage, explore different angles."
        else:
            phase_desc = f"Late Stage (Iteration {middle_threshold + 1}+)"
            guidance = "Fill knowledge gaps, ensure completeness before concluding."

        unique_tools = set(used_tools)
        available_unexplored = [
            t for t in ["rag", "paper_search", "web_search"]
            if getattr(self, f"enable_{t.replace('_search', '_search')}", False)
            and t not in unique_tools
        ]
        diversity_hint = ""
        if available_unexplored and iteration > early_threshold:
            diversity_hint = f"\n**Tool Diversity Suggestion**: Consider unexplored tools: {', '.join(available_unexplored)}"

        if self.iteration_mode == "flexible":
            mode_guidance = (
                "\n**Iteration Mode: FLEXIBLE (Auto)**\n"
                "You have autonomy to decide when knowledge is sufficient. Stop early if core concepts are well covered."
            )
        else:
            mode_guidance = (
                "\n**Iteration Mode: FIXED**\n"
                "Be CONSERVATIVE about declaring sufficiency. Require strong evidence of comprehensive coverage."
            )

        return f"\n**Research Phase Guidance** ({phase_desc}):\n{guidance}\n\nCurrent: {iteration}/{self.max_iterations}{diversity_hint}{mode_guidance}\n"

    def _generate_online_search_instruction(self) -> str:
        if not self.enable_web_search and not self.enable_paper_search:
            return ""
        if self.enable_web_search and self.enable_paper_search:
            return self.get_prompt("guidance", "online_search_both") or ""
        if self.enable_web_search:
            return self.get_prompt("guidance", "online_search_web_only") or ""
        return self.get_prompt("guidance", "online_search_paper_only") or ""

    def _generate_iteration_mode_criteria(self, iteration: int) -> str:
        early_threshold = max(2, self.max_iterations // 3)
        if self.iteration_mode == "flexible":
            criteria = self.get_prompt("guidance", "iteration_mode_flexible")
            return criteria or "- **FLEXIBLE mode**: You have autonomy to decide sufficiency."
        criteria = self.get_prompt("guidance", "iteration_mode_fixed")
        if criteria:
            return criteria.format(early_threshold=early_threshold)
        return f"- **FIXED mode**: Be CONSERVATIVE. Early threshold: {early_threshold}"

    async def _run_llm_self_research(
        self,
        *,
        topic: str,
        overview: str,
        query: str,
        current_knowledge: str,
        iteration: int,
        block_id: str,
    ) -> str:
        if self.language == "zh":
            system_prompt = (
                "你是一个深度研究助理。在没有任何外部工具可用时，你需要只基于模型已有知识进行研究。"
                "不要假装访问了网页、论文或知识库。"
            )
            user_prompt = self._safe_format(
                "请对以下查询进行一次纯LLM内部研究。\n\n"
                "主话题：{topic}\n话题概览：{overview}\n当前轮次：{iteration}\n"
                "本轮查询：{query}\n已有知识：\n{current_knowledge}\n\n"
                "仅输出JSON：\n{{\"content\": \"结构化研究内容\", \"confidence\": \"high/medium/low\"}}",
                topic=topic, overview=overview or "(无)", iteration=iteration,
                query=query, current_knowledge=current_knowledge[:3000] or "(无)",
            )
        else:
            system_prompt = (
                "You are a deep-research assistant. When no external tools are enabled, "
                "use only model internal knowledge. Do not claim to have searched externally."
            )
            user_prompt = self._safe_format(
                "Perform one round of LLM-only internal research.\n\n"
                "Main Topic: {topic}\nOverview: {overview}\nIteration: {iteration}\n"
                "Query: {query}\nCurrent Knowledge:\n{current_knowledge}\n\n"
                "Output JSON only:\n{{\"content\": \"Structured content\", \"confidence\": \"high/medium/low\"}}",
                topic=topic, overview=overview or "(none)", iteration=iteration,
                query=query, current_knowledge=current_knowledge[:3000] or "(none)",
            )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt, stage="llm_self_research",
            trace_meta=self._build_trace_meta(label="LLM self research", iteration=iteration, block_id=block_id),
        ):
            _chunks.append(_c)
        response = "".join(_chunks)

        try:
            data = extract_json_from_text(response)
        except Exception:
            data = None
        return json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else json.dumps({"content": response})

    async def plan_next_step(
        self,
        topic: str,
        overview: str,
        current_knowledge: str,
        iteration: int,
        existing_topics: list[str] | None = None,
        used_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("ResearchAgent missing system prompt")
        user_prompt_template = self.get_prompt("process", "plan_next_step")
        if not user_prompt_template:
            raise ValueError("ResearchAgent missing plan_next_step prompt")

        topics_text = "(No other topics)"
        if existing_topics:
            topics_text = "\n".join([f"- {t}" for t in existing_topics])

        user_prompt = self._safe_format(
            user_prompt_template,
            topic=topic, overview=overview,
            current_knowledge=current_knowledge[:3000] or "(None)",
            iteration=iteration, max_iterations=self.max_iterations,
            existing_topics=topics_text,
            available_tools=self._generate_available_tools_text(),
            tool_phase_guidance=self._generate_tool_phase_guidance(),
            research_depth_guidance=self._generate_research_depth_guidance(iteration, used_tools or []),
            online_search_instruction=self._generate_online_search_instruction(),
            iteration_mode_criteria=self._generate_iteration_mode_criteria(iteration),
            mode_instruction=self._get_mode_contract("research"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt, stage="plan_next_step",
            trace_meta=self._build_trace_meta(label="Plan next step", iteration=iteration, trace_role="plan"),
        ):
            _chunks.append(_c)
        response = "".join(_chunks)

        from ..utils.json_utils import ensure_json_dict, ensure_keys

        data = extract_json_from_text(response)
        obj = ensure_json_dict(data)
        ensure_keys(obj, ["is_sufficient", "sufficiency_reason"])
        return obj

    async def process(
        self,
        topic_block: TopicBlock,
        call_tool_callback: Callable[[str, str], Awaitable[str]],
        note_agent: Any,
        citation_manager: Any,
        queue: DynamicTopicQueue,
        manager_agent: Any,
        config: dict[str, Any],
        progress_callback: Callable[[str, Any], None] | None = None,
    ) -> dict[str, Any]:
        block_id_prefix = f"[{topic_block.block_id}]"
        print(f"\n{block_id_prefix} ResearchAgent: topic={topic_block.sub_topic} max_iter={self.max_iterations}")

        iteration = 0
        current_knowledge = ""
        tools_used: list[str] = []
        queries_used: list[dict[str, Any]] = []
        llm_only_mode = self._is_llm_only_mode()

        def send_progress(event_type: str, **data: Any) -> None:
            if progress_callback:
                try:
                    progress_callback(event_type, **data)
                except Exception:
                    pass

        while iteration < self.max_iterations:
            iteration += 1
            print(f"{block_id_prefix} Iteration {iteration}/{self.max_iterations}")

            send_progress("iteration_started", iteration=iteration, max_iterations=self.max_iterations, tools_used=tools_used.copy())
            send_progress("checking_sufficiency", iteration=iteration, max_iterations=self.max_iterations)

            plan = await self.plan_next_step(
                topic=topic_block.sub_topic, overview=topic_block.overview,
                current_knowledge=current_knowledge, iteration=iteration,
                existing_topics=queue.list_topics(), used_tools=tools_used,
            )

            if plan.get("is_sufficient", False):
                send_progress("knowledge_sufficient", iteration=iteration, max_iterations=self.max_iterations,
                              reason=plan.get("sufficiency_reason", ""))
                break

            send_progress("generating_query", iteration=iteration, max_iterations=self.max_iterations)

            # Dynamic topic splitting
            new_topic = plan.get("new_sub_topic")
            new_overview = plan.get("new_overview")
            new_topic_score = float(plan.get("new_topic_score") or 0)
            should_add_new_topic = plan.get("should_add_new_topic")
            min_score = config.get("researching", {}).get("new_topic_min_score", 0.75)

            if isinstance(new_topic, str) and new_topic.strip():
                trimmed = new_topic.strip()
                if should_add_new_topic is False:
                    pass
                elif new_topic_score < min_score:
                    pass
                else:
                    add_topic_method = getattr(manager_agent, "add_new_topic")
                    if inspect.iscoroutinefunction(add_topic_method):
                        added = await add_topic_method(trimmed, new_overview or "")
                    else:
                        added = manager_agent.add_new_topic(trimmed, new_overview or "")
                    if added:
                        send_progress("new_topic_added", iteration=iteration, max_iterations=self.max_iterations,
                                      new_topic=trimmed, new_overview=new_overview or "")

            query = plan.get("query", "").strip()
            tool_type = "llm_self_research" if llm_only_mode else plan.get("tool_type", "rag")
            rationale = plan.get("rationale", "")

            if not query:
                send_progress("query_empty", iteration=iteration, max_iterations=self.max_iterations)
                continue

            queries_used.append({"query": query, "tool_type": tool_type, "rationale": rationale, "iteration": iteration})
            send_progress("tool_calling", iteration=iteration, max_iterations=self.max_iterations,
                          tool_type=tool_type, query=query, rationale=rationale)

            if llm_only_mode:
                raw_answer = await self._run_llm_self_research(
                    topic=topic_block.sub_topic, overview=topic_block.overview,
                    query=query, current_knowledge=current_knowledge, iteration=iteration, block_id=topic_block.block_id,
                )
            else:
                raw_answer = await call_tool_callback(tool_type, query)

            send_progress("tool_completed", iteration=iteration, max_iterations=self.max_iterations,
                          tool_type=tool_type, query=query)
            send_progress("processing_notes", iteration=iteration, max_iterations=self.max_iterations)

            # Get citation_id (prefer async variant)
            if hasattr(citation_manager, "get_next_citation_id_async") and inspect.iscoroutinefunction(
                getattr(citation_manager, "get_next_citation_id_async", None)
            ):
                citation_id = await citation_manager.get_next_citation_id_async(
                    stage="research", block_id=topic_block.block_id
                )
            else:
                citation_id = citation_manager.get_next_citation_id(stage="research", block_id=topic_block.block_id)

            trace = await note_agent.process(
                tool_type=tool_type, query=query, raw_answer=raw_answer,
                citation_id=citation_id, topic=topic_block.sub_topic, context=current_knowledge,
            )
            topic_block.add_tool_trace(trace)

            # Add citation (prefer async variant)
            add_citation_fn = getattr(citation_manager, "add_citation", None)
            if add_citation_fn:
                if inspect.iscoroutinefunction(add_citation_fn):
                    await add_citation_fn(citation_id=citation_id, tool_type=tool_type, tool_trace=trace, raw_answer=raw_answer)
                else:
                    citation_manager.add_citation(citation_id=citation_id, tool_type=tool_type, tool_trace=trace, raw_answer=raw_answer)

            current_knowledge = (current_knowledge + "\n" + trace.summary).strip()
            topic_block.iteration_count = iteration
            tools_used.append(tool_type)

            send_progress("iteration_completed", iteration=iteration, max_iterations=self.max_iterations,
                          tool_type=tool_type, query=query, tools_used=tools_used.copy())

        return {
            "block_id": topic_block.block_id,
            "iterations": iteration,
            "final_knowledge": current_knowledge,
            "tools_used": tools_used,
            "queries_used": queries_used,
            "status": "completed" if iteration < self.max_iterations else "max_iterations_reached",
        }


__all__ = ["ResearchAgent"]
