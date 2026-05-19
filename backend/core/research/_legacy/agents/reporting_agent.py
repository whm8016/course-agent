"""ReportingAgent - Report generation Agent (faithful port from DeepTutor DR-in-KG 2.0).

Adaptation: ``deeptutor.utils.json_parser.parse_json_response`` → local ``utils.json_utils``.
"""

from __future__ import annotations

import json as _json
import re
from collections.abc import Callable
from string import Template
from typing import Any

from ..base_agent import BaseAgent
from ..data_structures import DynamicTopicQueue, TopicBlock
from ..trace import build_trace_metadata, new_call_id
from ..utils.json_utils import ensure_json_dict, ensure_keys, extract_json_from_text, parse_json_response


class ReportingAgent(BaseAgent):
    """Report generation Agent"""

    @staticmethod
    def _build_trace_meta(label: str, trace_kind: str = "llm_generation") -> dict[str, Any]:
        return build_trace_metadata(
            call_id=new_call_id("research-report"),
            phase="reporting",
            label=label,
            call_kind=trace_kind,
            trace_role="response",
            trace_kind=trace_kind,
        )

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _convert_to_template_format(template_str: str) -> str:
        return re.sub(r"\{(\w+)\}", r"$\1", template_str)

    def _safe_format(self, template_str: str, **kwargs: Any) -> str:
        converted = self._convert_to_template_format(template_str)
        return Template(converted).safe_substitute(**kwargs)

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
            agent_name="reporting_agent",
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            language=language,
            config=config,
        )
        self.reporting_config = config.get("reporting", {})
        self.citation_manager = None

        self.enable_citation_list = self.reporting_config.get("enable_citation_list", False)
        self.enable_inline_citations = self.reporting_config.get("enable_inline_citations", False)
        self.deduplicate_enabled = self.reporting_config.get("deduplicate_enabled", False)
        self.single_pass_threshold = int(self.reporting_config.get("report_single_pass_threshold", 0))
        self.report_style = str(self.reporting_config.get("style", "report") or "report")

    def set_citation_manager(self, citation_manager: Any) -> None:
        self.citation_manager = citation_manager

    @staticmethod
    def _append_contract(prompt: str, heading: str, contract: str) -> str:
        contract = str(contract or "").strip()
        return prompt if not contract else f"{prompt}\n\n{heading}:\n{contract}\n"

    def _get_mode_contract(self, stage: str) -> str:
        return (self.get_prompt("mode_contracts", f"{self.report_style}_{stage}", "") or "").strip()

    def _get_mode_process_prompt(self, base_key: str, default: str = "") -> str:
        if self.report_style and self.report_style != "report":
            mode_key = f"{base_key}_{self.report_style}"
            mode_prompt = self.get_prompt("process", mode_key, "")
            if mode_prompt:
                return mode_prompt
        return self.get_prompt("process", base_key, default)

    async def process(
        self,
        queue: DynamicTopicQueue,
        topic: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        print(f"\n{'=' * 70}\nReportingAgent - Report Generation\ntopic={topic}")
        self._progress_callback = progress_callback
        self._notify_progress(progress_callback, "reporting_started", topic=topic, total_blocks=len(queue.blocks))

        candidate_blocks = queue.get_all_completed_blocks() or queue.blocks

        print("Step 1: Preparing topic blocks...")
        if self.deduplicate_enabled:
            cleaned_blocks = await self._deduplicate_blocks(candidate_blocks)
        else:
            cleaned_blocks = candidate_blocks
        self._notify_progress(progress_callback, "deduplicate_completed", kept_blocks=len(cleaned_blocks))

        print("Step 2: Generating outline...")
        outline = await self._generate_outline(topic, cleaned_blocks)
        self._notify_progress(progress_callback, "outline_completed", sections=len(outline.get("sections", [])))
        self._current_outline = outline

        print("Step 3: Writing report...")
        report_markdown = await self._write_report(topic, cleaned_blocks, outline)
        self._notify_progress(progress_callback, "writing_completed")

        word_count = len(report_markdown)
        sections = len(cleaned_blocks)
        citations = sum(len(b.tool_traces) for b in cleaned_blocks)

        self._notify_progress(progress_callback, "reporting_completed",
                               word_count=word_count, sections=sections, citations=citations)

        result: dict[str, Any] = {"report": report_markdown, "word_count": word_count, "sections": sections, "citations": citations}
        if hasattr(self, "_current_outline"):
            result["outline"] = self._current_outline
            del self._current_outline

        return result

    async def _deduplicate_blocks(self, blocks: list[TopicBlock]) -> list[TopicBlock]:
        if len(blocks) <= 1:
            return blocks
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("ReportingAgent missing system prompt")
        user_prompt = self.get_prompt("process", "deduplicate")
        if not user_prompt:
            raise ValueError("ReportingAgent missing deduplicate prompt")

        topics_text = "\n".join([f"{i+1}. {b.sub_topic}: {b.overview[:200]}" for i, b in enumerate(blocks)])
        filled = self._safe_format(user_prompt, topics=topics_text, total_topics=len(blocks))
        _chunks: list[str] = []
        async for _c in self.stream_llm(filled, system_prompt, stage="deduplicate", trace_meta=self._build_trace_meta("Deduplicate topics")):
            _chunks.append(_c)
        data = extract_json_from_text("".join(_chunks))
        try:
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["keep_indices"])
            keep_indices = obj.get("keep_indices", [])
            return [blocks[i] for i in keep_indices if isinstance(i, int) and i < len(blocks)]
        except Exception:
            return blocks

    async def _generate_outline(self, topic: str, blocks: list[TopicBlock]) -> dict[str, Any]:
        system_prompt = self.get_prompt("system", "role")
        if not system_prompt:
            raise ValueError("ReportingAgent missing system prompt")
        user_prompt = self._get_mode_process_prompt("generate_outline")
        if not user_prompt:
            raise ValueError("ReportingAgent missing generate_outline prompt")

        topics_data = [
            {
                "index": i, "block_id": block.block_id, "sub_topic": block.sub_topic,
                "overview": block.overview,
                "tool_summaries": [t.summary for t in block.tool_traces] if block.tool_traces else [],
            }
            for i, block in enumerate(blocks, 1)
        ]
        topics_json = _json.dumps(topics_data, ensure_ascii=False, indent=2)
        filled = self._safe_format(user_prompt, topic=topic, topics_json=topics_json, total_topics=len(blocks))
        filled = self._append_contract(filled, "Mode-specific outline contract", self._get_mode_contract("outline"))

        _chunks: list[str] = []
        async for _c in self.stream_llm(filled, system_prompt, stage="generate_outline", trace_meta=self._build_trace_meta("Generate outline")):
            _chunks.append(_c)
        data = extract_json_from_text("".join(_chunks))
        try:
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["title", "introduction", "sections", "conclusion"])
            if not obj.get("title", "").startswith("#"):
                obj["title"] = f"# {obj.get('title', topic)}"
            for key in ("introduction", "conclusion"):
                if obj.get(key) and not obj[key].startswith("##"):
                    obj[key] = f"## {obj[key]}"
            for section in obj.get("sections", []):
                if section.get("title") and not section["title"].startswith("##"):
                    section["title"] = f"## {section['title']}"
                for sub in section.get("subsections", []):
                    if sub.get("title") and not sub["title"].startswith("###"):
                        sub["title"] = f"### {sub['title']}"
            return obj
        except Exception:
            return self._create_default_outline(topic, blocks)

    def _create_default_outline(self, topic: str, blocks: list[TopicBlock]) -> dict[str, Any]:
        style_defaults: dict[str, tuple[str, str, str, str]] = {
            "study_notes": ("## Study Overview", "Orient the learner, define the scope, and state learning goals",
                            "## Key Takeaways", "Summarize the most important concepts and memory anchors"),
            "comparison": ("## Comparison Setup", "Define the comparison target and evaluation lens",
                           "## Recommendation by Scenario", "Summarize trade-offs and recommend options"),
            "learning_path": ("## Learning Goal and Scope", "Clarify the learner profile and prerequisites",
                              "## Milestones and Next Steps", "Summarize stage milestones and how to progress"),
        }
        intro_title, intro_instr, conclusion_title, conclusion_instr = style_defaults.get(
            self.report_style,
            ("## Introduction", "Present background, motivation, objectives and report structure",
             "## Conclusion and Future Directions", "Summarize core findings and future directions"),
        )
        sections = []
        for i, b in enumerate(blocks, 1):
            section = {
                "title": f"## {i}. {b.sub_topic}",
                "instruction": f"Provide detailed introduction to {b.sub_topic}",
                "block_id": b.block_id,
                "subsections": [
                    {"title": f"### {i}.1 Core Concepts", "instruction": f"Core concepts of {b.sub_topic}"},
                    {"title": f"### {i}.2 Key Mechanisms", "instruction": f"Mechanisms and principles of {b.sub_topic}"},
                ],
            }
            sections.append(section)
        return {
            "title": f"# {topic}",
            "introduction": intro_title,
            "introduction_instruction": intro_instr,
            "sections": sections,
            "conclusion": conclusion_title,
            "conclusion_instruction": conclusion_instr,
        }

    def _ser_block(self, b: TopicBlock) -> dict[str, Any]:
        traces = []
        for t in b.tool_traces:
            cid = getattr(t, "citation_id", None) or f"CIT-{b.block_id.split('_')[-1]}-01"
            trace_data: dict[str, Any] = {
                "citation_id": cid,
                "tool_type": t.tool_type,
                "query": t.query,
                "raw_answer": t.raw_answer,
                "summary": t.summary,
            }
            if hasattr(self, "_citation_map") and self._citation_map:
                ref_num = self._citation_map.get(cid, 0)
                if ref_num > 0:
                    trace_data["ref_number"] = ref_num
            traces.append(trace_data)
        return {"block_id": b.block_id, "sub_topic": b.sub_topic, "overview": b.overview, "traces": traces}

    def _build_citation_table(self, block: TopicBlock) -> str:
        if not block.tool_traces:
            return "  (No citations available for this section)"
        lines = []
        for trace in block.tool_traces:
            cid = getattr(trace, "citation_id", None)
            if not cid:
                continue
            ref_num = self._citation_map.get(cid, 0) if hasattr(self, "_citation_map") else 0
            if ref_num <= 0:
                continue
            query_preview = trace.query[:60] + "..." if len(trace.query) > 60 else trace.query
            tool_display = {"rag": "RAG", "paper_search": "Paper", "web_search": "Web", "run_code": "Code"}.get(
                trace.tool_type.lower(), trace.tool_type
            )
            lines.append(f"  - Cite as [{ref_num}] → ({tool_display}) {query_preview}")
        return "\n".join(lines) if lines else "  (No citations available for this section)"

    async def _call_llm_json(
        self, prompt: str, system_prompt: str, stage: str, trace_label: str,
        required_keys: list[str], max_retries: int = 1,
    ) -> dict[str, Any]:
        last_error = None
        for attempt in range(max_retries + 1):
            _chunks: list[str] = []
            async for _c in self.stream_llm(prompt, system_prompt, stage=stage, trace_meta=self._build_trace_meta(trace_label)):
                _chunks.append(_c)
            resp = "".join(_chunks)
            data = extract_json_from_text(resp)
            try:
                obj = ensure_json_dict(data)
                ensure_keys(obj, required_keys)
                return obj
            except (ValueError, KeyError) as exc:
                last_error = exc
        raise ValueError(f"Failed to get valid JSON for {stage}. Required keys: {required_keys}. Last error: {last_error}")

    async def _write_introduction(self, topic: str, blocks: list[TopicBlock], outline: dict[str, Any]) -> str:
        system_prompt = self.get_prompt("system", "role", "You are an academic writing expert.")
        tmpl = self._get_mode_process_prompt("write_introduction")
        if not tmpl:
            raise ValueError("Cannot get introduction writing prompt template")
        topics_summary = [{"sub_topic": b.sub_topic, "overview": b.overview, "tool_count": len(b.tool_traces)} for b in blocks]
        intro_instruction = outline.get("introduction_instruction", "") or outline.get("introduction", "")
        topics_summary_json = _json.dumps(topics_summary, ensure_ascii=False, indent=2)
        filled = self._safe_format(tmpl, topic=topic, introduction_instruction=intro_instruction,
                                   topics_summary=topics_summary_json, total_topics=len(blocks))
        filled = self._append_contract(filled, "Mode-specific introduction contract", self._get_mode_contract("introduction"))
        data = await self._call_llm_json(filled, system_prompt, "write_introduction", "Write introduction", ["introduction"])
        return data["introduction"]

    async def _write_section_body(self, topic: str, block: TopicBlock, section_outline: dict[str, Any]) -> str:
        system_prompt = self.get_prompt("system", "role", "You are an academic writing expert.")
        tmpl = self._get_mode_process_prompt("write_section_body")
        if not tmpl:
            raise ValueError("Cannot get section writing prompt template")

        block_data = self._ser_block(block)
        citation_instruction, citation_output_hint = self._build_citation_instruction(block)

        block_data_json = _json.dumps(block_data, ensure_ascii=False, indent=2)
        filled = self._safe_format(
            tmpl, topic=topic,
            section_title=section_outline.get("title", block.sub_topic),
            section_instruction=section_outline.get("instruction", ""),
            block_data=block_data_json,
            min_section_length=self.reporting_config.get("min_section_length", 500),
            citation_instruction=citation_instruction,
            citation_output_hint=citation_output_hint,
        )
        filled = self._append_contract(filled, "Mode-specific section contract", self._get_mode_contract("section"))
        data = await self._call_llm_json(filled, system_prompt, "write_section_body", "Write section", ["section_content"])
        return data["section_content"]

    async def _write_conclusion(self, topic: str, blocks: list[TopicBlock], outline: dict[str, Any]) -> str:
        system_prompt = self.get_prompt("system", "role", "You are an academic writing expert.")
        tmpl = self._get_mode_process_prompt("write_conclusion")
        if not tmpl:
            raise ValueError("Cannot get conclusion writing prompt template")
        topics_findings = [{"sub_topic": b.sub_topic, "overview": b.overview, "key_findings": [t.summary for t in b.tool_traces[:3]]} for b in blocks]
        conclusion_instruction = outline.get("conclusion_instruction", "") or outline.get("conclusion", "")
        topics_findings_json = _json.dumps(topics_findings, ensure_ascii=False, indent=2)
        filled = self._safe_format(tmpl, topic=topic, conclusion_instruction=conclusion_instruction,
                                   topics_findings=topics_findings_json, total_topics=len(blocks))
        filled = self._append_contract(filled, "Mode-specific conclusion contract", self._get_mode_contract("conclusion"))
        data = await self._call_llm_json(filled, system_prompt, "write_conclusion", "Write conclusion", ["conclusion"])
        return data["conclusion"]

    def _build_citation_instruction(self, block: TopicBlock) -> tuple[str, str]:
        if self.enable_inline_citations:
            citation_table = self._build_citation_table(block)
            tmpl = self.get_prompt("citation", "enabled_instruction")
            if tmpl:
                return tmpl.format(citation_table=citation_table), ", citations"
            return f"**Citation Reference Table**:\n{citation_table}", ", citations"
        return self.get_prompt("citation", "disabled_instruction") or "", ""

    def _build_citation_number_map(self, blocks: list[TopicBlock]) -> dict[str, int]:
        if self.citation_manager:
            return self.citation_manager.build_ref_number_map()

        def sort_key(cit_id: str) -> tuple:
            try:
                if cit_id.startswith("PLAN-"):
                    return (0, 0, int(cit_id.replace("PLAN-", "")))
                parts = cit_id.replace("CIT-", "").split("-")
                if len(parts) == 2:
                    return (1, int(parts[0]), int(parts[1]))
            except Exception:
                pass
            return (999, 999, 999)

        all_cit_ids: list[str] = []
        seen: set[str] = set()
        for block in blocks:
            for trace in (block.tool_traces or []):
                cid = getattr(trace, "citation_id", None)
                if cid and cid not in seen:
                    all_cit_ids.append(cid)
                    seen.add(cid)
        all_cit_ids.sort(key=sort_key)
        return {cid: idx for idx, cid in enumerate(all_cit_ids, 1)}

    def _generate_references(self, blocks: list[TopicBlock]) -> str:
        if self.citation_manager:
            return self._generate_references_from_manager(blocks)
        return self._generate_references_from_blocks(blocks)

    def _generate_references_from_manager(self, blocks: list[TopicBlock]) -> str:
        parts = ["## References\n\n"]
        all_citations = self.citation_manager.get_all_citations()
        if not all_citations:
            return "## References\n\n*No citations available.*\n"

        ref_map = self.citation_manager.get_ref_number_map()
        ref_to_citations: dict[int, list[tuple[str, dict[str, Any], Any]]] = {}

        for citation_id, citation in all_citations.items():
            tool_type = citation.get("tool_type", "").lower()
            if tool_type == "paper_search":
                papers = citation.get("papers", [])
                if papers:
                    for pidx, paper in enumerate(papers):
                        ref_num = ref_map.get(f"{citation_id}-{pidx+1}") or ref_map.get(citation_id, 0)
                        if ref_num > 0:
                            ref_to_citations.setdefault(ref_num, []).append((citation_id, citation, paper))
                else:
                    ref_num = ref_map.get(citation_id, 0)
                    if ref_num > 0:
                        ref_to_citations.setdefault(ref_num, []).append((citation_id, citation, None))
            else:
                ref_num = ref_map.get(citation_id, 0)
                if ref_num > 0:
                    ref_to_citations.setdefault(ref_num, []).append((citation_id, citation, None))

        for ref_num in sorted(ref_to_citations):
            entries = ref_to_citations[ref_num]
            if not entries:
                continue
            citation_id, citation, paper = entries[0]
            tool_type = citation.get("tool_type", "").lower()
            parts.append(f'<a id="ref-{ref_num}"></a>**[{ref_num}]** ')
            if tool_type == "paper_search":
                parts.append(self._format_single_paper_apa(paper) if paper else self._format_paper_citation_apa(citation))
            elif tool_type == "web_search":
                parts.append(self._format_web_search_citation(citation))
            elif tool_type in ("rag", "rag_naive", "rag_hybrid"):
                parts.append(self._format_rag_citation(citation))
            elif tool_type == "run_code":
                parts.append(self._format_code_citation(citation))
            else:
                query = citation.get("query", "")
                summary = citation.get("summary", "")
                parts.append(f"**{tool_type}**\n\n- **Query**: {query}\n")
                if summary:
                    clean = self._strip_markdown(summary)
                    parts.append(f"- **Summary**: {clean[:300]}{'...' if len(clean) > 300 else ''}\n")
            parts.append("\n\n")
        return "".join(parts)

    def _format_single_paper_apa(self, paper: dict[str, Any]) -> str:
        authors = paper.get("authors", "Unknown Author")
        year = paper.get("year", "n.d.")
        title = paper.get("title", "Untitled")
        url = paper.get("url", "")
        arxiv_id = paper.get("arxiv_id", "")
        venue = paper.get("venue", "")
        doi = paper.get("doi", "")
        result = f"{authors} ({year}). *{title}*."
        if venue:
            result += f" {venue}."
        if arxiv_id:
            result += f" arXiv:{arxiv_id}."
        if doi:
            result += f" https://doi.org/{doi}"
        elif url:
            result += f" {url}"
        return result

    def _format_paper_citation_apa(self, citation: dict[str, Any]) -> str:
        return self._format_single_paper_apa(citation)

    def _format_web_search_citation(self, citation: dict[str, Any]) -> str:
        query = citation.get("query", "")
        summary = citation.get("summary", "")
        web_sources = citation.get("web_sources", [])
        result = "**Web Search**\n\n"
        result += f"- **Query**: {query}\n"
        if summary:
            clean = self._strip_markdown(summary)
            result += f"- **Summary**: {clean[:300]}{'...' if len(clean) > 300 else ''}\n"
        if web_sources:
            result += f"\n<details>\n<summary>Retrieved Sources ({len(web_sources)} links)</summary>\n\n"
            for i, src in enumerate(web_sources, 1):
                title = src.get("title", "Untitled")
                url = src.get("url", "")
                snippet = src.get("snippet", "")
                if url:
                    result += f"{i}. [{title}]({url})"
                    if snippet:
                        clean_snip = self._strip_markdown(snippet)
                        result += f"\n   > {clean_snip[:150]}{'...' if len(clean_snip) > 150 else ''}"
                    result += "\n\n"
            result += "</details>"
        return result

    def _format_rag_citation(self, citation: dict[str, Any]) -> str:
        query = citation.get("query", "")
        summary = citation.get("summary", "")
        kb_name = citation.get("kb_name", "")
        sources = citation.get("sources", [])
        result = "**RAG**"
        if kb_name:
            result += f" (KB: {kb_name})"
        result += "\n\n"
        result += f"- **Query**: {query}\n"
        if summary:
            clean = self._strip_markdown(summary)
            result += f"- **Summary**: {clean[:300]}{'...' if len(clean) > 300 else ''}\n"
        if sources:
            result += f"\n<details>\n<summary>Source Documents ({len(sources)} docs)</summary>\n\n"
            for i, src in enumerate(sources, 1):
                title = src.get("title", "") or src.get("source_file", f"Document {i}")
                content = src.get("content_preview", "")
                page = src.get("page", "")
                result += f"{i}. **{title}**"
                if page:
                    result += f" (Page {page})"
                if content:
                    clean_c = self._strip_markdown(content)
                    result += f"\n   > {clean_c[:150]}{'...' if len(clean_c) > 150 else ''}"
                result += "\n\n"
            result += "</details>"
        return result

    def _format_code_citation(self, citation: dict[str, Any]) -> str:
        query = citation.get("query", "")
        summary = citation.get("summary", "")
        result = "**Code Execution**\n\n"
        if query:
            code_preview = query[:300] + ("..." if len(query) > 300 else "")
            result += f"- **Code**: `{code_preview}`\n"
        if summary:
            result += f"- **Result**: {summary[:300]}{'...' if len(summary) > 300 else ''}\n"
        return result

    def _generate_references_from_blocks(self, blocks: list[TopicBlock]) -> str:
        parts = ["## References\n\n"]
        all_cit: list[dict[str, Any]] = []
        for block in blocks:
            for trace in (block.tool_traces or []):
                cid = getattr(trace, "citation_id", None) or f"CIT-{block.block_id.split('_')[-1]}-01"
                all_cit.append({"citation_id": cid, "trace": trace})
        if not all_cit:
            return "## References\n\n*No citations available.*\n"

        def sort_key(cit: dict[str, Any]) -> tuple:
            cid = cit["citation_id"]
            try:
                if cid.startswith("PLAN-"):
                    return (0, 0, int(cid.replace("PLAN-", "")))
                parts = cid.replace("CIT-", "").split("-")
                if len(parts) == 2:
                    return (1, int(parts[0]), int(parts[1]))
            except Exception:
                pass
            return (999, 999, 999)

        all_cit.sort(key=sort_key)
        for idx, cit in enumerate(all_cit, 1):
            trace = cit["trace"]
            tool_type = trace.tool_type.lower() if trace.tool_type else ""
            tool_display = {"rag": "RAG", "paper_search": "Paper Search", "web_search": "Web Search", "run_code": "Code Execution"}.get(tool_type, tool_type)
            parts.append(f'<a id="ref-{idx}"></a>**[{idx}]** **{tool_display}**\n\n')
            parts.append(f"- **Query**: {trace.query}\n")
            if trace.summary:
                summary_text = trace.summary[:500] + ("..." if len(trace.summary) > 500 else "")
                parts.append(f"- **Summary**: {summary_text}\n")
            parts.append("\n")
        return "".join(parts)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"  +", " ", text)
        return text.strip()

    def _convert_citation_format(self, text: str) -> str:
        valid_refs: set[int] = set()
        if hasattr(self, "_citation_map") and self._citation_map:
            valid_refs = set(self._citation_map.values())

        def replace_citation(match: re.Match) -> str:
            try:
                num = int(match.group(1))
                if num in valid_refs:
                    return f"[[{num}]](#ref-{num})"
            except ValueError:
                pass
            return match.group(0)

        text = re.sub(r"\[ref=(\d+)\]", replace_citation, text)
        text = re.sub(r"(?<!\[)\[(\d+)\](?!\(#ref-)", replace_citation, text)
        return text

    def _validate_and_fix_citations(self, text: str) -> tuple[str, dict[str, Any]]:
        valid_refs: set[int] = set()
        if hasattr(self, "_citation_map") and self._citation_map:
            valid_refs = set(self._citation_map.values())

        pattern = r"\[\[(\d+)\]\]\(#ref-\d+\)"
        found = re.findall(pattern, text)
        valid, invalid = [], []
        for ref in found:
            try:
                num = int(ref)
                (valid if num in valid_refs else invalid).append(num)
            except ValueError:
                invalid.append(ref)

        if invalid:
            def remove_invalid(match: re.Match) -> str:
                try:
                    return match.group(0) if int(match.group(1)) in valid_refs else ""
                except ValueError:
                    return ""
            text = re.sub(pattern, remove_invalid, text)

        return text, {"valid_citations": valid, "invalid_citations": invalid, "is_valid": not invalid, "total_found": len(found)}

    async def _write_report(self, topic: str, blocks: list[TopicBlock], outline: dict[str, Any]) -> str:
        if self.enable_inline_citations:
            self._citation_map = self._build_citation_number_map(blocks)
        else:
            self._citation_map = {}

        if self.single_pass_threshold > 0 and len(blocks) <= self.single_pass_threshold:
            report = await self._write_report_single_pass(topic, blocks, outline)
        else:
            report = await self._write_report_step_by_step(topic, blocks, outline)

        if self.enable_inline_citations:
            report = self._convert_citation_format(report)
            report, validation = self._validate_and_fix_citations(report)
            if not validation["is_valid"]:
                print(f"  Removed {len(validation['invalid_citations'])} invalid citations")

        return report

    async def _write_report_step_by_step(self, topic: str, blocks: list[TopicBlock], outline: dict[str, Any]) -> str:
        parts = []
        title = outline.get("title", f"# {topic}")
        if not title.startswith("#"):
            title = f"# {title}"
        parts.append(f"{title}\n\n")

        introduction = await self._write_introduction(topic, blocks, outline)
        intro_title = outline.get("introduction", "## Introduction")
        if not intro_title.startswith("##"):
            intro_title = f"## {intro_title}"
        parts.append(f"{intro_title}\n\n{introduction}\n\n")

        sections = outline.get("sections", [])
        for i, section in enumerate(sections, 1):
            block_id = section.get("block_id")
            block = next((b for b in blocks if b.block_id == block_id), None)
            if not block:
                continue
            subsections = section.get("subsections", [])
            if subsections:
                content = await self._write_section_with_subsections(topic, block, section, subsections)
            else:
                content = await self._write_section_body(topic, block, section)
            self._notify_progress(
                getattr(self, "_progress_callback", None), "writing_section",
                current_section=section.get("title", block.sub_topic).replace("##", "").strip(),
                section_index=i, total_sections=len(sections) + 2,
            )
            parts.append(f"{content}\n\n")

        conclusion = await self._write_conclusion(topic, blocks, outline)
        conclusion_title = outline.get("conclusion", "## Conclusion")
        if not conclusion_title.startswith("##"):
            conclusion_title = f"## {conclusion_title}"
        parts.append(f"{conclusion_title}\n\n{conclusion}\n\n")

        if self.enable_citation_list:
            parts.append(self._generate_references(blocks))

        return "".join(parts)

    async def _write_report_single_pass(self, topic: str, blocks: list[TopicBlock], outline: dict[str, Any]) -> str:
        system_prompt = self.get_prompt("system", "role", "You are an academic writing expert.")
        tmpl = self._get_mode_process_prompt("write_full_report")
        if not tmpl:
            raise ValueError("Cannot get single-pass report prompt template")

        blocks_data = [self._ser_block(block) for block in blocks]
        outline_json = _json.dumps(outline, ensure_ascii=False, indent=2)
        blocks_json = _json.dumps(blocks_data, ensure_ascii=False, indent=2)

        citation_instruction = (
            "Use inline citations in [N] format whenever a trace provides a ref_number. Do not add a References section."
            if self.enable_inline_citations
            else "Do not add inline citations and do not add a References section."
        )

        filled = self._safe_format(
            tmpl, topic=topic, outline_json=outline_json, blocks_json=blocks_json,
            total_topics=len(blocks), citation_instruction=citation_instruction,
        )
        filled = self._append_contract(filled, "Mode-specific full-report contract", self._get_mode_contract("single_pass"))

        _chunks: list[str] = []
        async for _c in self.stream_llm(filled, system_prompt, stage="write_full_report", trace_meta=self._build_trace_meta("Write full report")):
            _chunks.append(_c)
        resp = "".join(_chunks)
        data = extract_json_from_text(resp)

        try:
            obj = ensure_json_dict(data)
            ensure_keys(obj, ["report"])
            report = obj.get("report", "")
            if not isinstance(report, str) or not report.strip():
                raise ValueError("Empty report field")
        except Exception:
            if isinstance(data, dict) and ("sections" in data or "title" in data):
                report = self._assemble_markdown_from_structured(data)
            else:
                stripped = resp.strip()
                report = stripped if (stripped and stripped.startswith("#")) else self._strip_json_wrapper(resp)

        if self.enable_citation_list:
            report = report.rstrip() + "\n\n" + self._generate_references(blocks)

        return report

    @staticmethod
    def _assemble_markdown_from_structured(obj: dict[str, Any]) -> str:
        parts: list[str] = []
        if obj.get("title"):
            parts.append(str(obj["title"]))
        if obj.get("introduction"):
            parts.append(str(obj["introduction"]))
        for section in obj.get("sections", []):
            if isinstance(section, str):
                parts.append(section)
            elif isinstance(section, dict):
                if section.get("title"):
                    parts.append(str(section["title"]))
                if section.get("content"):
                    parts.append(str(section["content"]))
                for sub in section.get("subsections", []):
                    if isinstance(sub, str):
                        parts.append(sub)
                    elif isinstance(sub, dict):
                        if sub.get("title"):
                            parts.append(str(sub["title"]))
                        if sub.get("content"):
                            parts.append(str(sub["content"]))
        if obj.get("conclusion"):
            parts.append(str(obj["conclusion"]))
        return "\n\n".join(parts)

    @staticmethod
    def _strip_json_wrapper(resp: str) -> str:
        obj = parse_json_response(resp.strip(), fallback=None)
        if isinstance(obj, dict):
            for key in ("report", "content", "text", "markdown", "output"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key]
        stripped = resp.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            for line in stripped.split("\\n"):
                if line.strip().startswith("#"):
                    return stripped.replace("\\n", "\n")
        return stripped

    async def _write_section_with_subsections(
        self, topic: str, block: TopicBlock, section: dict[str, Any], subsections: list[dict[str, Any]]
    ) -> str:
        system_prompt = self.get_prompt("system", "role", "You are an academic writing expert.")
        tmpl = self._get_mode_process_prompt("write_section_body")
        if not tmpl:
            raise ValueError("Cannot get section writing prompt template")

        subsection_info = [{"title": s.get("title", ""), "instruction": s.get("instruction", "")} for s in subsections]
        block_data = self._ser_block(block)
        block_data["expected_subsections"] = subsection_info

        section_instruction = section.get("instruction", "")
        if subsection_info:
            guide = "\n\n**Expected subsection structure:**\n"
            for sub in subsection_info:
                guide += f"- {sub['title']}: {sub['instruction']}\n"
            section_instruction += guide

        citation_instruction, citation_output_hint = self._build_citation_instruction(block)
        block_data_json = _json.dumps(block_data, ensure_ascii=False, indent=2)
        filled = self._safe_format(
            tmpl, topic=topic,
            section_title=section.get("title", block.sub_topic),
            section_instruction=section_instruction,
            block_data=block_data_json,
            min_section_length=self.reporting_config.get("min_section_length", 800),
            citation_instruction=citation_instruction,
            citation_output_hint=citation_output_hint,
        )
        filled = self._append_contract(filled, "Mode-specific section contract", self._get_mode_contract("section"))

        try:
            data = await self._call_llm_json(filled, system_prompt, "write_section_with_subsections", "Write section", ["section_content"])
            content = data["section_content"]
            if isinstance(content, str) and content.strip():
                return content
            raise ValueError("Empty section_content field")
        except ValueError:
            _chunks: list[str] = []
            async for _c in self.stream_llm(filled, system_prompt, stage="write_section_with_subsections_fallback",
                                             trace_meta=self._build_trace_meta("Write section (fallback)")):
                _chunks.append(_c)
            resp = "".join(_chunks).strip()
            if not resp:
                raise ValueError(f"Unable to generate section for '{section.get('title', 'unknown')}'")
            return self._strip_json_wrapper(resp)

    def _notify_progress(self, callback: Any, status: str, **payload: Any) -> None:
        if not callback:
            return
        event: dict[str, Any] = {"status": status}
        event.update({k: v for k, v in payload.items() if v is not None})
        try:
            callback(event)
        except Exception:
            pass


__all__ = ["ReportingAgent"]
