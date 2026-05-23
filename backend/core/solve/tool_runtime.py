"""Solve tool runtime: RAG only (course LlamaIndex), aligned with DeepTutor SolveToolRuntime API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_CONTROL_ACTIONS = {
    "en": [
        {
            "name": "done",
            "when_to_use": "Use when the current step already has enough reliable evidence to move on.",
            "input_format": "Empty string.",
        },
        {
            "name": "replan",
            "when_to_use": "Use when the current plan is no longer appropriate and the planner should revise it.",
            "input_format": "A short reason describing why replanning is needed.",
        },
    ],
    "zh": [
        {
            "name": "done",
            "when_to_use": "当当前步骤已经获得足够且可靠的证据，可以进入下一步时使用。",
            "input_format": "空字符串。",
        },
        {
            "name": "replan",
            "when_to_use": "当现有计划已不合适，需要重新规划时使用。",
            "input_format": "简要说明为何需要重规划。",
        },
    ],
}


@dataclass
class ToolResult:
    content: str
    success: bool = True
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SolveToolRuntime:
    """Minimal wrapper: enabled_tools filters which tools appear in prompts; execution supports rag."""

    _TOOL_NAMES_DEFAULT = ["rag"]

    def __init__(
        self,
        enabled_tools: list[str] | None,
        language: str = "zh",
    ) -> None:
        self.language = language
        names = [str(x).strip().lower() for x in (enabled_tools or self._TOOL_NAMES_DEFAULT)]
        self._tool_names: list[str] = []
        for n in names:
            if n == "rag" and "rag" not in self._tool_names:
                self._tool_names.append("rag")
        if not self._tool_names:
            self._tool_names = ["rag"]

        self._valid_actions: set[str] = {"done", "replan"}
        self._valid_actions.update(self._tool_names)
        self._valid_actions.update({"rag_hybrid", "rag_naive"})

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    @property
    def valid_actions(self) -> set[str]:
        return set(self._valid_actions)

    def has_tool(self, name: str) -> bool:
        return str(name).lower() == "rag" and "rag" in self._tool_names

    def resolve_tool_name(self, name: str) -> str | None:
        n = str(name).lower()
        if n in ("rag", "rag_hybrid", "rag_naive") and self.has_tool("rag"):
            return "rag"
        return None

    def build_planner_description(self, kb_name: str = "") -> str:
        lines = [
            "- **rag**: Retrieve relevant passages from the course knowledge base.",
        ]
        if kb_name:
            lines.append(f"  Knowledge base (course): `{kb_name}`")
        return "\n".join(lines)

    def build_solver_description(self) -> str:
        ctrl = _CONTROL_ACTIONS.get(self.language, _CONTROL_ACTIONS["en"])
        rows = ["| Action | When to use | Input |", "|--------|-------------|-------|"]
        for c in ctrl:
            rows.append(
                f"| `{c['name']}` | {c['when_to_use']} | {c['input_format']} |",
            )
        if self.has_tool("rag"):
            rows.append(
                "| `rag` | Fetch course materials for a focused query. | Natural language query string. |",
            )
        return "\n".join(rows)

    async def execute(
        self,
        action: str,
        action_input: str,
        *,
        kb_name: str | None = None,
        output_dir: str | None = None,
        reason_context: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        event_sink: Any | None = None,
    ) -> ToolResult:
        del output_dir, reason_context, api_key, base_url, model, max_tokens, temperature

        resolved = self.resolve_tool_name(action)
        if resolved != "rag":
            raise KeyError(f"Unknown tool action: {action}")
        action = resolved

        if "rag" not in self._tool_names:
            raise PermissionError(f"Tool action '{action}' is not enabled for solve.")

        if not kb_name:
            return ToolResult(
                content=(
                    "RAG retrieval was requested, but no knowledge base is "
                    "configured for this turn. Continue without retrieved "
                    "knowledge or ask the user to select a course/knowledge base."
                ),
                success=False,
                sources=[],
                metadata={"skipped": True, "reason": "no_kb_selected"},
            )

        async def _sink(msg: str) -> None:
            if event_sink is None:
                return
            try:
                result = event_sink("tool_log", msg, {})
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass

        await _sink(f"Query: {action_input}" if action_input else "Starting retrieval")

        try:
            from core.rag.rag_llama import retrieve_context_llamaindex

            result = await retrieve_context_llamaindex(course_id=str(kb_name).strip(), query=action_input)
            answer = (result.get("answer") or "").strip()
            payload = {
                "answer": answer,
                "kb_name": kb_name,
                "query": action_input,
                "provider": result.get("provider", "llamaindex"),
            }
            sources: list[dict[str, Any]] = [
                {
                    "type": "rag",
                    "file": str(kb_name),
                    "chunk_id": action_input,
                }
            ]
            await _sink(f"Retrieve complete ({len(answer)} chars)")
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                success=True,
                sources=sources,
                metadata=payload,
            )
        except Exception as exc:
            logger.warning("RAG execute failed: %s", exc)
            await _sink(f"Retrieve failed: {exc}")
            payload = {"status": "failed", "error": str(exc), "query": action_input}
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                success=False,
                metadata=payload,
            )


__all__ = ["SolveToolRuntime", "ToolResult"]
