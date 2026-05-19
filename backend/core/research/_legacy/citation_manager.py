"""Citation manager - faithful port from DeepTutor.

Only change from the original: ``get_path_service()`` replaced by a local
``_get_cache_dir`` helper that resolves to ``<BASE_DIR>/data/research/workspace``.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils.json_utils import parse_json_response


def _get_cache_dir(research_id: str) -> Path:
    try:
        from config import BASE_DIR
        base = Path(BASE_DIR)
    except ImportError:
        base = Path(__file__).resolve().parents[3]
    return base / "data" / "research" / "workspace" / research_id


class CitationManager:
    """Citation manager with global ID management"""

    def __init__(self, research_id: str, cache_dir: Path | None = None):
        self.research_id = research_id
        if cache_dir is None:
            cache_dir = _get_cache_dir(research_id)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.citations_file = self.cache_dir / "citations.json"
        self._citations: dict[str, dict[str, Any]] = {}

        self._plan_counter = 0
        self._block_counters: dict[str, int] = {}
        self._ref_number_map: dict[str, int] = {}
        self._lock = asyncio.Lock()

        self._load_citations()

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def generate_plan_citation_id(self) -> str:
        self._plan_counter += 1
        return f"PLAN-{self._plan_counter:02d}"

    def generate_research_citation_id(self, block_id: str) -> str:
        block_num = 0
        try:
            if block_id and "_" in block_id:
                block_num = int(block_id.split("_")[1])
        except (ValueError, IndexError):
            block_num = 0

        block_key = str(block_num)
        if block_key not in self._block_counters:
            self._block_counters[block_key] = 0
        self._block_counters[block_key] += 1

        return f"CIT-{block_num}-{self._block_counters[block_key]:02d}"

    def get_next_citation_id(self, stage: str = "research", block_id: str = "") -> str:
        if stage == "planning":
            return self.generate_plan_citation_id()
        return self.generate_research_citation_id(block_id)

    def citation_exists(self, citation_id: str) -> bool:
        return citation_id in self._citations

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_citations(self) -> None:
        if self.citations_file.exists():
            try:
                with open(self.citations_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self._citations = data.get("citations", {})
                    counters = data.get("counters", {})
                    if counters:
                        self._plan_counter = counters.get("plan_counter", 0)
                        self._block_counters = counters.get("block_counters", {})
                    else:
                        self._restore_counters_from_citations()
            except Exception as exc:
                print(f"Warning: failed to load citation file: {exc}")
                self._citations = {}
        else:
            self._citations = {}

    def _restore_counters_from_citations(self) -> None:
        for citation_id in self._citations:
            if citation_id.startswith("PLAN-"):
                try:
                    num = int(citation_id.replace("PLAN-", ""))
                    self._plan_counter = max(self._plan_counter, num)
                except ValueError:
                    pass
            elif citation_id.startswith("CIT-"):
                try:
                    parts = citation_id.replace("CIT-", "").split("-")
                    if len(parts) == 2:
                        block_num = parts[0]
                        seq_num = int(parts[1])
                        if block_num not in self._block_counters:
                            self._block_counters[block_num] = 0
                        self._block_counters[block_num] = max(
                            self._block_counters[block_num], seq_num
                        )
                except (ValueError, IndexError):
                    pass

    def _save_citations(self) -> None:
        try:
            data = {
                "research_id": self.research_id,
                "updated_at": datetime.now().isoformat(),
                "citations": self._citations,
                "counters": {
                    "plan_counter": self._plan_counter,
                    "block_counters": self._block_counters,
                },
            }
            with open(self.citations_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Warning: failed to save citation file: {exc}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_citation_references(self, text: str) -> dict[str, Any]:
        pattern = r"\[\[([A-Z]+-\d+-?\d*)\]\]"
        found_refs = re.findall(pattern, text)

        valid = []
        invalid = []
        for ref in found_refs:
            (valid if self.citation_exists(ref) else invalid).append(ref)

        return {
            "valid_citations": valid,
            "invalid_citations": invalid,
            "is_valid": len(invalid) == 0,
            "total_found": len(found_refs),
        }

    def fix_invalid_citations(self, text: str) -> str:
        pattern = r"\[\[([A-Z]+-\d+-?\d*)\]\]\(#ref-[a-z]+-\d+-?\d*\)"

        def replace_invalid(match: re.Match) -> str:
            return match.group(0) if self.citation_exists(match.group(1)) else ""

        return re.sub(pattern, replace_invalid, text)

    # ------------------------------------------------------------------
    # Adding citations
    # ------------------------------------------------------------------

    def add_citation(
        self,
        citation_id: str,
        tool_type: str,
        tool_trace: Any,
        raw_answer: str,
    ) -> bool:
        try:
            tl = tool_type.lower()
            if tl in ("rag", "rag_naive", "rag_hybrid"):
                info = self._extract_rag_citation(citation_id, "rag", raw_answer, tool_trace)
            elif tl == "web_search":
                info = self._extract_web_citation(citation_id, tool_type, raw_answer, tool_trace)
            elif tl == "paper_search":
                info = self._extract_paper_citation(citation_id, tool_type, raw_answer, tool_trace)
            elif tl == "run_code":
                info = self._extract_code_citation(citation_id, tool_type, tool_trace)
            else:
                info = self._extract_generic_citation(citation_id, tool_type, tool_trace)

            if info:
                self._citations[citation_id] = info
                self._save_citations()
                return True
            return False
        except Exception as exc:
            print(f"Warning: failed to add citation {citation_id}: {exc}")
            return False

    def _extract_rag_citation(
        self, citation_id: str, tool_type: str, raw_answer: str, tool_trace: Any
    ) -> dict[str, Any]:
        citation_info: dict[str, Any] = {
            "citation_id": citation_id,
            "tool_type": tool_type,
            "query": tool_trace.query,
            "summary": tool_trace.summary,
            "timestamp": tool_trace.timestamp,
            "sources": [],
        }
        try:
            answer_data = parse_json_response(raw_answer) or {}
            sources = []
            for field_name in ["chunks", "documents", "sources", "context", "retrieved_docs"]:
                if field_name in answer_data:
                    source_list = answer_data[field_name]
                    if isinstance(source_list, list):
                        for i, doc in enumerate(source_list[:5]):
                            src: dict[str, Any] = {}
                            if isinstance(doc, dict):
                                src["title"] = doc.get("title", doc.get("doc_title", ""))
                                src["content_preview"] = doc.get("content", doc.get("text", ""))[:200]
                                src["source_file"] = doc.get(
                                    "source", doc.get("file_path", doc.get("filename", ""))
                                )
                                src["page"] = doc.get("page", doc.get("page_number", ""))
                                src["chunk_id"] = doc.get("chunk_id", doc.get("id", i))
                                src["score"] = doc.get("score", doc.get("similarity", ""))
                            elif isinstance(doc, str):
                                src["content_preview"] = doc[:200]
                            if src:
                                sources.append(src)
                    break
            citation_info["kb_name"] = answer_data.get("kb_name", "")
            citation_info["sources"] = sources
            citation_info["total_sources"] = len(sources)
        except Exception as exc:
            print(f"Warning: failed to parse RAG source info: {exc}")
        return citation_info

    def _extract_web_citation(
        self, citation_id: str, tool_type: str, raw_answer: str, tool_trace: Any
    ) -> dict[str, Any]:
        citation_info: dict[str, Any] = {
            "citation_id": citation_id,
            "tool_type": tool_type,
            "query": tool_trace.query,
            "summary": tool_trace.summary,
            "timestamp": tool_trace.timestamp,
            "web_sources": [],
        }
        try:
            answer_data = parse_json_response(raw_answer) or {}
            web_sources = []
            for field_name in ["results", "web_results", "search_results", "urls"]:
                if field_name in answer_data:
                    result_list = answer_data[field_name]
                    if isinstance(result_list, list):
                        for result in result_list[:5]:
                            if isinstance(result, dict):
                                ws = {
                                    "title": result.get("title", ""),
                                    "url": result.get("url", result.get("link", "")),
                                    "snippet": result.get(
                                        "snippet", result.get("description", "")
                                    )[:200],
                                    "domain": result.get("domain", ""),
                                }
                                if ws["url"]:
                                    web_sources.append(ws)
                    break
            citation_info["web_sources"] = web_sources
            citation_info["total_sources"] = len(web_sources)
        except Exception as exc:
            print(f"Warning: failed to parse web source info: {exc}")
        return citation_info

    def _extract_paper_citation(
        self, citation_id: str, tool_type: str, raw_answer: str, tool_trace: Any
    ) -> dict[str, Any]:
        citation_info: dict[str, Any] = {
            "citation_id": citation_id,
            "tool_type": tool_type,
            "query": tool_trace.query,
            "summary": tool_trace.summary,
            "timestamp": tool_trace.timestamp,
            "papers": [],
        }
        try:
            answer_data = parse_json_response(raw_answer) or {}
            papers = answer_data.get("papers", [])
            if not papers:
                return citation_info

            processed_papers = []
            for paper in papers[:5]:
                authors = paper.get("authors", [])
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                processed_papers.append(
                    {
                        "title": paper.get("title", ""),
                        "authors": author_str,
                        "authors_list": authors,
                        "year": paper.get("year", ""),
                        "url": paper.get("url", ""),
                        "arxiv_id": paper.get("arxiv_id", ""),
                        "abstract": paper.get("abstract", "")[:300],
                        "doi": paper.get("doi", ""),
                        "venue": paper.get("venue", paper.get("journal", "")),
                    }
                )

            citation_info["papers"] = processed_papers
            citation_info["total_papers"] = len(processed_papers)
            if processed_papers:
                primary = processed_papers[0]
                for key in ("title", "authors", "authors_list", "year", "url", "arxiv_id"):
                    citation_info[key] = primary[key]
        except Exception as exc:
            print(f"Warning: failed to parse paper citation: {exc}")
        return citation_info

    def _extract_code_citation(
        self, citation_id: str, tool_type: str, tool_trace: Any
    ) -> dict[str, Any]:
        return {
            "citation_id": citation_id,
            "tool_type": tool_type,
            "query": tool_trace.query,
            "summary": tool_trace.summary,
            "timestamp": tool_trace.timestamp,
        }

    def _extract_generic_citation(
        self, citation_id: str, tool_type: str, tool_trace: Any
    ) -> dict[str, Any]:
        return {
            "citation_id": citation_id,
            "tool_type": tool_type,
            "query": tool_trace.query,
            "summary": tool_trace.summary,
            "timestamp": tool_trace.timestamp,
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_citation(self, citation_id: str) -> dict[str, Any] | None:
        return self._citations.get(citation_id)

    def get_all_citations(self) -> dict[str, dict[str, Any]]:
        return self._citations.copy()

    def get_citations_file_path(self) -> Path:
        return self.citations_file

    def format_citation_for_report(self, citation_id: str) -> str | None:
        citation = self.get_citation(citation_id)
        if not citation:
            return None

        tool_type = citation.get("tool_type", "").lower()

        if tool_type == "paper_search":
            title = citation.get("title", "")
            authors = citation.get("authors", "")
            year = citation.get("year", "")
            url = citation.get("url", "")
            arxiv_id = citation.get("arxiv_id", "")
            parts = []
            if authors:
                parts.append(authors)
            if year:
                parts.append(f"({year})")
            if title:
                parts.append(f'"{title}"')
            if arxiv_id:
                parts.append(f"arXiv:{arxiv_id}")
            if url:
                parts.append(f"<{url}>")
            total = citation.get("total_papers", 1)
            if total > 1:
                parts.append(f"[+{total - 1} more papers]")
            return " ".join(parts) if parts else None

        if tool_type in ("rag", "rag_naive", "rag_hybrid"):
            query = citation.get("query", "")
            kb_name = citation.get("kb_name", "")
            sources = citation.get("sources", [])
            parts = [f"RAG: {query}"]
            if kb_name:
                parts.append(f"[KB: {kb_name}]")
            if sources:
                titles = [s.get("title", s.get("source_file", "")) for s in sources[:3] if s]
                titles = [t for t in titles if t]
                if titles:
                    parts.append(f"[Sources: {', '.join(titles)}]")
            return " ".join(parts)

        if tool_type == "web_search":
            query = citation.get("query", "")
            web_sources = citation.get("web_sources", [])
            parts = [f"Web Search: {query}"]
            if web_sources:
                urls = [s.get("url", "") for s in web_sources[:3] if s.get("url")]
                if urls:
                    parts.append(f"[URLs: {', '.join(urls)}]")
            return " ".join(parts)

        display = {"run_code": "Code Execution"}.get(tool_type, tool_type)
        query = citation.get("query", "")
        return f"{display}: {query}"

    # ------------------------------------------------------------------
    # Reference number map
    # ------------------------------------------------------------------

    def _get_citation_dedup_key(
        self, citation: dict[str, Any], paper: dict[str, Any] | None = None
    ) -> str:
        tool_type = citation.get("tool_type", "").lower()
        citation_id = citation.get("citation_id", "")

        if tool_type == "paper_search":
            src = paper or citation
            title = src.get("title", "").lower().strip()
            authors = src.get("authors", "").lower().strip()
            first_author = authors.split(",")[0].strip() if authors else ""
            if title:
                return f"paper:{title}|{first_author}"
            return f"unique:{citation_id}"
        return f"unique:{citation_id}"

    def _extract_citation_sort_key(self, citation_id: str) -> tuple:
        try:
            if citation_id.startswith("PLAN-"):
                return (0, 0, int(citation_id.replace("PLAN-", "")))
            parts = citation_id.replace("CIT-", "").split("-")
            if len(parts) == 2:
                return (1, int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            pass
        return (999, 999, 999)

    def build_ref_number_map(self) -> dict[str, int]:
        if not self._citations:
            self._ref_number_map = {}
            return self._ref_number_map

        sorted_ids = sorted(self._citations, key=self._extract_citation_sort_key)
        seen_keys: dict[str, int] = {}
        ref_idx = 0
        ref_map: dict[str, int] = {}

        for citation_id in sorted_ids:
            citation = self._citations.get(citation_id)
            if not citation:
                continue
            tool_type = citation.get("tool_type", "").lower()

            if tool_type == "paper_search":
                papers = citation.get("papers", [])
                if papers:
                    for pidx, paper in enumerate(papers):
                        dedup = self._get_citation_dedup_key(citation, paper)
                        if dedup in seen_keys:
                            existing = seen_keys[dedup]
                            if pidx == 0:
                                ref_map[citation_id] = existing
                            ref_map[f"{citation_id}-{pidx + 1}"] = existing
                        else:
                            ref_idx += 1
                            seen_keys[dedup] = ref_idx
                            if pidx == 0:
                                ref_map[citation_id] = ref_idx
                            ref_map[f"{citation_id}-{pidx + 1}"] = ref_idx
                else:
                    dedup = self._get_citation_dedup_key(citation)
                    if dedup in seen_keys:
                        ref_map[citation_id] = seen_keys[dedup]
                    else:
                        ref_idx += 1
                        seen_keys[dedup] = ref_idx
                        ref_map[citation_id] = ref_idx
            else:
                dedup = self._get_citation_dedup_key(citation)
                if dedup in seen_keys:
                    ref_map[citation_id] = seen_keys[dedup]
                else:
                    ref_idx += 1
                    seen_keys[dedup] = ref_idx
                    ref_map[citation_id] = ref_idx

        self._ref_number_map = ref_map
        return ref_map

    def get_ref_number(self, citation_id: str) -> int:
        if not self._ref_number_map:
            self.build_ref_number_map()
        return self._ref_number_map.get(citation_id, 0)

    def get_ref_number_map(self) -> dict[str, int]:
        if not self._ref_number_map:
            self.build_ref_number_map()
        return self._ref_number_map.copy()

    # ------------------------------------------------------------------
    # Async thread-safe wrappers (for parallel research mode)
    # ------------------------------------------------------------------

    async def generate_plan_citation_id_async(self) -> str:
        async with self._lock:
            return self.generate_plan_citation_id()

    async def generate_research_citation_id_async(self, block_id: str) -> str:
        async with self._lock:
            return self.generate_research_citation_id(block_id)

    async def get_next_citation_id_async(self, stage: str = "research", block_id: str = "") -> str:
        async with self._lock:
            return self.get_next_citation_id(stage, block_id)

    async def add_citation_async(
        self,
        citation_id: str,
        tool_type: str,
        tool_trace: Any,
        raw_answer: str,
    ) -> bool:
        async with self._lock:
            return self.add_citation(citation_id, tool_type, tool_trace, raw_answer)


__all__ = ["CitationManager"]
