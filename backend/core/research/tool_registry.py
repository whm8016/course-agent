"""ToolRegistry (Step 0): RAG only."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    content: str
    success: bool = True
    sources: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        kb_name: str | None = None,
    ) -> None:
        self._config = dict(config or {})
        if kb_name:
            self._default_kb_name = str(kb_name).strip()
        else:
            rag = self._config.get("rag") or {}
            self._default_kb_name = str(rag.get("kb_name") or "").strip()

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool_name = (tool_name or "").lower()
        if tool_name in ("rag", "rag_hybrid", "rag_naive"):
            return await self._execute_rag(**kwargs)
        payload = {"status": "unknown_tool", "tool": tool_name}
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False),
            success=False,
            metadata=payload,
        )

    async def _execute_rag(self, query: str = "", kb_name: str = "", **_: Any) -> ToolResult:
        from core.rag.rag_llama import retrieve_context_llamaindex

        effective_kb = (kb_name or "").strip() or self._default_kb_name
        if not effective_kb:
            payload = {
                "status": "skipped",
                "reason": "no_kb_selected",
                "message": "RAG requires kb_name on the research request.",
                "tool": "rag",
                "query": query,
            }
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                success=False,
                metadata=payload,
            )
        try:
            result = await retrieve_context_llamaindex(course_id=effective_kb, query=query)
            answer = result.get("answer", "")
            payload = {
                "answer": answer,
                "content": answer,
                "kb_name": effective_kb,
                "query": query,
                "success": True,
            }
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                success=True,
                metadata=payload,
            )
        except Exception as exc:
            logger.warning("RAG execute failed: %s", exc)
            payload = {"status": "failed", "error": str(exc), "query": query}
            return ToolResult(
                content=json.dumps(payload, ensure_ascii=False),
                success=False,
                metadata=payload,
            )


__all__ = ["ToolRegistry", "ToolResult"]
