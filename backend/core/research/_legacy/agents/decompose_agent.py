"""DecomposeAgent - Topic decomposition Agent (faithful port from DeepTutor).

Adaptation: ``rag_search`` calls are routed through the local ``ToolRegistry``
instead of DeepTutor's ``deeptutor.tools.rag_tool``.
"""

from __future__ import annotations

import json
from typing import Any

from ..base_agent import BaseAgent
from ..data_structures import ToolTrace
from ..trace import build_trace_metadata, new_call_id
from ..tool_registry import ToolRegistry
from ..utils.json_utils import extract_json_from_text


class DecomposeAgent(BaseAgent):
    """Topic decomposition Agent"""

    _MODE_TO_STYLE = {
        "notes": "study_notes",
        "report": "report",
        "comparison": "comparison",
        "learning_path": "learning_path",
    }

    @staticmethod
    def _build_trace_meta(mode: str) -> dict[str, Any]:
        return build_trace_metadata(
            call_id=new_call_id("research-decompose"),
            phase="decomposing",
            label="Decompose topic",
            call_kind="llm_generation",
            trace_role="plan",
            trace_kind="llm_generation",
            mode=mode,
        )

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        kb_name: str | None = None,
    ):
        language = config.get("system", {}).get("language", "zh")
        super().__init__(
            module_name="research",
            agent_name="decompose_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )
        rag_cfg = config.get("rag", {}) or {}
        self.kb_name = rag_cfg.get("kb_name") or kb_name or None

        researching_cfg = config.get("researching", {})
        self.enable_rag = researching_cfg.get("enable_rag", True)
        if not self.kb_name:
            self.enable_rag = False

        self.conversation_history: list[dict[str, Any]] = config.get("conversation_history", [])
        self.citation_manager = None

        intent_mode = str(config.get("intent", {}).get("mode", "") or "")
        reporting_style = str(config.get("reporting", {}).get("style", "") or "")
        self._research_style = reporting_style or self._MODE_TO_STYLE.get(intent_mode, "report")

        self._tool_registry = ToolRegistry(config=config, kb_name=self.kb_name)

    def set_citation_manager(self, citation_manager: Any) -> None:
        self.citation_manager = citation_manager

    def _format_conversation_context(self) -> str:
        if not self.conversation_history:
            return ""
        parts = []
        for msg in self.conversation_history:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if content:
                parts.append(f"[{role}]: {content}")
        if not parts:
            return ""
        return (
            "\n<conversation_history>\n"
            "The following is the conversation history of this session.\n\n"
            + "\n\n".join(parts)
            + "\n</conversation_history>\n"
        )

    def _get_mode_contract(self, stage: str) -> str:
        return (
            self.get_prompt("mode_contracts", f"{self._research_style}_{stage}", "") or ""
        ).strip()

    async def process(
        self,
        topic: str,
        num_subtopics: int = 5,
        mode: str = "manual",
        attachments: list[Any] | None = None,
    ) -> dict[str, Any]:
        print(f"\n{'=' * 70}")
        print("DecomposeAgent - Topic Decomposition")
        print(f"{'=' * 70}")
        print(f"Main Topic: {topic} | Mode: {mode} | RAG: {self.enable_rag}")

        if not self.enable_rag:
            return await self._process_without_rag(topic, num_subtopics, mode, attachments)

        rag_context, source_query = await self._retrieve_background_knowledge(topic)

        if mode == "auto":
            sub_topics = await self._generate_sub_topics_auto(
                topic=topic, rag_context=rag_context, max_subtopics=num_subtopics, attachments=attachments
            )
        else:
            sub_topics = await self._generate_sub_topics(
                topic=topic, rag_context=rag_context, num_subtopics=num_subtopics, attachments=attachments
            )

        print(f"Generated {len(sub_topics)} subtopics")
        return {
            "main_topic": topic,
            "sub_queries": [source_query] if source_query else [],
            "rag_context": rag_context,
            "sub_topics": sub_topics,
            "total_subtopics": len(sub_topics),
            "mode": mode,
        }

    async def _retrieve_background_knowledge(self, topic: str) -> tuple[str, str]:
        source_query = (topic or "").strip()
        if not source_query or not self.kb_name:
            return "", source_query

        try:
            tool_result = await self._tool_registry.execute(
                "rag",
                query=source_query,
                kb_name=self.kb_name or "",
            )
            raw_answer = tool_result.content
            result = json.loads(raw_answer) if isinstance(raw_answer, str) else raw_answer
            rag_context = result.get("answer", "") if isinstance(result, dict) else str(raw_answer)
            print(f"  Retrieved background ({len(rag_context)} chars)")

            if self.citation_manager:
                citation_id = self.citation_manager.get_next_citation_id(stage="planning")
                import time
                trace = ToolTrace(
                    tool_id=f"plan_tool_{int(time.time() * 1000)}",
                    citation_id=citation_id,
                    tool_type="rag",
                    query=source_query,
                    raw_answer=raw_answer,
                    summary=rag_context[:500],
                )
                self.citation_manager.add_citation(
                    citation_id=citation_id,
                    tool_type="rag",
                    tool_trace=trace,
                    raw_answer=raw_answer,
                )

            return rag_context, source_query
        except Exception as exc:
            print(f"  RAG retrieval failed: {exc}")
            return "", source_query

    async def _process_without_rag(
        self, topic: str, num_subtopics: int, mode: str = "manual", attachments: list[Any] | None = None
    ) -> dict[str, Any]:
        system_prompt = self.get_prompt(
            "system", "role",
            "You are a research planning expert. Decompose complex topics into clear subtopics.",
        )
        system_prompt += self._format_conversation_context()

        user_prompt_template = self.get_prompt("process", "decompose_without_rag")
        if not user_prompt_template:
            raise ValueError("DecomposeAgent missing decompose_without_rag prompt")

        if mode == "auto":
            decompose_requirement = (
                f"\nGenerate between 3 and {num_subtopics} subtopics based on complexity.\n"
            )
        else:
            decompose_requirement = (
                f"\nGenerate exactly {num_subtopics} subtopics, no more, no less.\n"
            )

        user_prompt = user_prompt_template.format(
            topic=topic,
            decompose_requirement=decompose_requirement,
            mode_instruction=self._get_mode_contract("decompose"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt, stage="decompose_no_rag",
            attachments=attachments, trace_meta=self._build_trace_meta(mode),
        ):
            _chunks.append(_c)
        response = "".join(_chunks)

        sub_topics = self._parse_subtopics(response, limit=num_subtopics)
        return {
            "main_topic": topic,
            "sub_queries": [],
            "rag_context": "",
            "sub_topics": sub_topics,
            "total_subtopics": len(sub_topics),
            "mode": f"{mode}_no_rag",
        }

    async def _generate_sub_topics_auto(
        self, topic: str, rag_context: str, max_subtopics: int, attachments: list[Any] | None = None
    ) -> list[dict[str, str]]:
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("DecomposeAgent missing system prompt")
        system_prompt += self._format_conversation_context()

        user_prompt_template = self.get_prompt("process", "decompose")
        if not user_prompt_template:
            raise ValueError("DecomposeAgent missing decompose prompt")

        decompose_requirement = (
            f"\nDynamically generate no more than {max_subtopics} subtopics based on the background knowledge.\n"
        )
        user_prompt = user_prompt_template.format(
            topic=topic, rag_context=rag_context,
            decompose_requirement=decompose_requirement,
            mode_instruction=self._get_mode_contract("decompose"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt, stage="decompose",
            attachments=attachments, trace_meta=self._build_trace_meta("auto"),
        ):
            _chunks.append(_c)
        return self._parse_subtopics("".join(_chunks), limit=max_subtopics)

    async def _generate_sub_topics(
        self, topic: str, rag_context: str, num_subtopics: int, attachments: list[Any] | None = None
    ) -> list[dict[str, str]]:
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("DecomposeAgent missing system prompt")
        system_prompt += self._format_conversation_context()

        user_prompt_template = self.get_prompt("process", "decompose")
        if not user_prompt_template:
            raise ValueError("DecomposeAgent missing decompose prompt")

        decompose_requirement = (
            f"\nGenerate exactly {num_subtopics} subtopics, no more, no less.\n"
        )
        user_prompt = user_prompt_template.format(
            topic=topic, rag_context=rag_context,
            decompose_requirement=decompose_requirement,
            mode_instruction=self._get_mode_contract("decompose"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt, stage="decompose",
            attachments=attachments, trace_meta=self._build_trace_meta("manual"),
        ):
            _chunks.append(_c)
        return self._parse_subtopics("".join(_chunks), limit=num_subtopics)

    @staticmethod
    def _parse_subtopics(response: str, limit: int) -> list[dict[str, str]]:
        from ..utils.json_utils import ensure_json_dict, ensure_keys

        data = extract_json_from_text(response)
        try:
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["sub_topics"])
            subs = obj.get("sub_topics", [])
            if not isinstance(subs, list):
                raise ValueError("sub_topics must be an array")
            cleaned = []
            for it in subs[:limit]:
                if isinstance(it, dict):
                    cleaned.append({"title": it.get("title", ""), "overview": it.get("overview", "")})
            return cleaned
        except Exception:
            return []


__all__ = ["DecomposeAgent"]
