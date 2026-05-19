"""NoteAgent - Recording Agent (faithful port from DeepTutor).

Adaptation: ``parse_json_response`` is imported from local ``utils.json_utils``
instead of ``deeptutor.utils.json_parser``.
"""

from __future__ import annotations

import re
import time
from string import Template
from typing import Any

from ..base_agent import BaseAgent
from ..data_structures import ToolTrace
from ..trace import build_trace_metadata, new_call_id
from ..utils.json_utils import extract_json_from_text, parse_json_response


class NoteAgent(BaseAgent):
    """Recording Agent"""

    _MODE_TO_STYLE = {
        "notes": "study_notes",
        "report": "report",
        "comparison": "comparison",
        "learning_path": "learning_path",
    }

    @staticmethod
    def _build_trace_meta(tool_type: str, query: str) -> dict[str, Any]:
        return build_trace_metadata(
            call_id=new_call_id("research-note"),
            phase="researching",
            label="Summarize evidence",
            call_kind="llm_observation",
            trace_role="observe",
            trace_kind="llm_generation",
            tool_name=tool_type,
            query=query,
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
            agent_name="note_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )
        researching_cfg = config.get("researching", {})
        self.summary_mode = researching_cfg.get("note_agent_mode", "auto")
        intent_mode = str(config.get("intent", {}).get("mode", "") or "")
        reporting_style = str(config.get("reporting", {}).get("style", "") or "")
        self._research_style = reporting_style or self._MODE_TO_STYLE.get(intent_mode, "report")

    async def process(
        self,
        tool_type: str,
        query: str,
        raw_answer: str,
        citation_id: str,
        topic: str = "",
        context: str = "",
    ) -> ToolTrace:
        print(f"\nNoteAgent - tool={tool_type} query={query[:60]} citation={citation_id}")

        summary = ""
        use_rule = self.summary_mode in ("rule", "auto")
        use_llm_fallback = self.summary_mode in ("llm", "auto")

        if use_rule:
            summary = self._extract_summary_by_rule(tool_type=tool_type, raw_answer=raw_answer)

        if (not summary or len(summary) < 50) and use_llm_fallback:
            summary = await self._generate_summary(
                tool_type=tool_type, query=query, raw_answer=raw_answer, topic=topic, context=context
            )
        elif not summary:
            summary = raw_answer[:1000]

        tool_id = f"tool_{int(time.time() * 1000)}"
        return ToolTrace(
            tool_id=tool_id,
            citation_id=citation_id,
            tool_type=tool_type,
            query=query,
            raw_answer=raw_answer,
            summary=summary,
        )

    @staticmethod
    def _convert_to_template_format(template_str: str) -> str:
        return re.sub(r"\{(\w+)\}", r"$\1", template_str)

    @staticmethod
    def _truncate_text(text: str, limit: int = 800) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    def _get_mode_contract(self, stage: str) -> str:
        return (
            self.get_prompt("mode_contracts", f"{self._research_style}_{stage}", "") or ""
        ).strip()

    def _get_mode_instruction_text(self, stage: str) -> str:
        instruction = self._get_mode_contract(stage)
        return f"Mode-specific note focus:\n{instruction}\n" if instruction else ""

    def _extract_summary_by_rule(self, tool_type: str, raw_answer: str) -> str:
        data = parse_json_response(raw_answer, fallback=None)
        if data is None:
            return ""

        tl = (tool_type or "").lower()

        if tl in {"rag_hybrid", "rag_naive", "rag"}:
            answer = data.get("answer") or data.get("content") or ""
            return self._truncate_text(answer)

        if tl == "web_search":
            answer = data.get("answer") or ""
            snippets: list[str] = []
            for item in (data.get("search_results") or data.get("results") or [])[:3]:
                if not isinstance(item, dict):
                    continue
                title = (item.get("title") or "").strip()
                snippet = (item.get("snippet") or item.get("content") or "").strip()
                piece = " - ".join(p for p in [title, snippet] if p)
                if piece:
                    snippets.append(piece)
            combined = "\n".join(p for p in [answer.strip(), *snippets] if p)
            return self._truncate_text(combined)

        if tl == "paper_search":
            papers = data.get("papers") or []
            formatted: list[str] = []
            for paper in papers[:3]:
                if not isinstance(paper, dict):
                    continue
                title = (paper.get("title") or "").strip()
                authors = paper.get("authors") or []
                authors_text = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
                year = paper.get("year")
                abstract = (paper.get("abstract") or "").strip()
                parts = [title]
                if authors_text:
                    parts.append(authors_text)
                if year:
                    parts.append(str(year))
                header = " | ".join(p for p in parts if p)
                body = "\n".join(p for p in [header, abstract] if p)
                if body:
                    formatted.append(body)
            return self._truncate_text("\n\n".join(formatted))

        if tl in {"run_code", "code_execution", "code_execute"}:
            stdout = (data.get("stdout") or "").strip()
            stderr = (data.get("stderr") or "").strip()
            artifacts = data.get("artifacts") or []
            parts = []
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if artifacts:
                parts.append(f"artifacts: {', '.join(str(a) for a in artifacts)}")
            return self._truncate_text("\n\n".join(parts))

        if isinstance(data, dict):
            for key in ("answer", "content", "summary", "message"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return self._truncate_text(value)

        return ""

    async def _generate_summary(
        self, tool_type: str, query: str, raw_answer: str, topic: str = "", context: str = ""
    ) -> str:
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("NoteAgent missing system prompt")

        user_prompt_template = self.get_prompt("process", "generate_summary")
        if not user_prompt_template:
            raise ValueError("NoteAgent missing generate_summary prompt")

        template_str = self._convert_to_template_format(user_prompt_template)
        template = Template(template_str)
        user_prompt = template.safe_substitute(
            tool_type=tool_type, query=query, raw_answer=raw_answer,
            topic=topic, context=context,
            mode_instruction=self._get_mode_instruction_text("note"),
        )

        _chunks: list[str] = []
        async for _c in self.stream_llm(
            user_prompt=user_prompt, system_prompt=system_prompt,
            stage="generate_summary",
            trace_meta=self._build_trace_meta(tool_type, query),
        ):
            _chunks.append(_c)
        response = "".join(_chunks)

        from ..utils.json_utils import ensure_json_dict, ensure_keys

        data = extract_json_from_text(response)
        try:
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["summary"])
            summary = obj.get("summary", "")
            return str(summary) if summary else ""
        except Exception:
            return (response or "")[:1000]


__all__ = ["NoteAgent"]
