"""
Thin mimic entrypoint — delegates to AgentCoordinator (aligned with DeepTutor tools/question/exam_mimic.py).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from core.question.coordinator import AgentCoordinator

WsCallback = Callable[[dict[str, Any]], Awaitable[None]]


async def mimic_exam_questions(
    pdf_path: str | None = None,
    paper_dir: str | None = None,
    kb_name: str | None = None,
    output_dir: str | None = None,
    max_questions: int | None = None,
    ws_callback: WsCallback | None = None,
    language: str = "zh",
) -> dict[str, Any]:
    """Backward-compatible wrapper around the coordinator exam pipeline."""
    if not pdf_path and not paper_dir:
        return {"success": False, "error": "Either pdf_path or paper_dir must be provided."}
    if pdf_path and paper_dir:
        return {"success": False, "error": "pdf_path and paper_dir cannot be used together."}

    coordinator = AgentCoordinator(
        kb_name=kb_name,
        output_dir=output_dir,
        language=language,
        enable_idea_rag=True,
    )

    if ws_callback:
        coordinator.set_ws_callback(ws_callback)

    if pdf_path:
        summary = await coordinator.generate_from_exam(
            exam_paper_path=pdf_path,
            max_questions=max_questions or 10,
            paper_mode="upload",
        )
    else:
        summary = await coordinator.generate_from_exam(
            exam_paper_path=paper_dir or "",
            max_questions=max_questions or 10,
            paper_mode="parsed",
        )

    return {
        "success": bool(summary.get("success", False)),
        "summary": summary,
        "generated_questions": [r.get("qa_pair", {}) for r in summary.get("results", [])],
        "failed_questions": [r for r in summary.get("results", []) if not r.get("success")],
        "total_reference_questions": summary.get("template_count", 0),
    }
