"""Minimal Deep Research pipeline (Step 0): topic -> single RAG -> LLM report."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import TEXT_MODEL
from core.llm.llm import client as _openai_client

from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Single-shot research: planning stub -> one RAG call -> one LLM report."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        api_key: str = "",
        base_url: str = "",
        api_version: str | None = None,
        research_id: str | None = None,
        kb_name: str | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        trace_callback: Callable[[dict[str, Any]], Any] | None = None,
        pre_confirmed_outline: list[dict[str, str]] | None = None,
        attachments: list[Any] | None = None,
    ) -> None:
        del api_key, base_url, api_version, trace_callback, pre_confirmed_outline, attachments

        self.config = config or {}
        self.progress_callback = progress_callback
        rag_cfg = self.config.get("rag") or {}
        self.kb_name = (kb_name or rag_cfg.get("kb_name") or "").strip() or None

        if research_id is None:
            research_id = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.research_id = research_id

        system_cfg = self.config.get("system") or {}
        self.reports_dir = Path(system_cfg.get("reports_dir", "./data/research/reports"))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._tool_registry = ToolRegistry(config=self.config, kb_name=self.kb_name)

    def _emit_progress(self, stage: str, status: str, **payload: Any) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(
                {
                    "type": "progress",
                    "stage": stage,
                    "status": status,
                    "research_id": self.research_id,
                    **{k: v for k, v in payload.items() if v is not None},
                }
            )
        except Exception as exc:
            logger.warning("Progress callback failed: %s", exc)

    async def run(self, topic: str) -> dict[str, Any]:
        started = datetime.now()
        self._emit_progress("planning", "planning_started", user_topic=topic)

        optimized_topic = topic.strip()
        self._emit_progress("planning", "planning_completed", optimized_topic=optimized_topic)

        self._emit_progress("researching", "researching_started")
        rag_result = await self._tool_registry.execute(
            "rag", query=optimized_topic, kb_name=self.kb_name or ""
        )
        self._emit_progress(
            "researching",
            "researching_completed",
            rag_success=rag_result.success,
        )

        self._emit_progress("reporting", "reporting_started")
        report = await self._generate_report(
            topic=topic,
            optimized_topic=optimized_topic,
            rag_context=rag_result.content,
        )
        self._emit_progress("reporting", "reporting_completed")

        report_file = self.reports_dir / f"{self.research_id}.md"
        report_file.write_text(report, encoding="utf-8")

        elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        metadata = {
            "research_id": self.research_id,
            "topic": topic,
            "optimized_topic": optimized_topic,
            "report_word_count": len(report.split()),
            "elapsed_ms": elapsed_ms,
            "rag_success": rag_result.success,
            "kb_name": self.kb_name,
            "completed_at": datetime.now().isoformat(),
        }
        meta_path = self.reports_dir / f"{self.research_id}_metadata.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "research_id": self.research_id,
            "topic": topic,
            "report": report,
            "final_report_path": str(report_file),
            "metadata": metadata,
        }

    async def _generate_report(self, topic: str, optimized_topic: str, rag_context: str) -> str:
        system_prompt = (
            "你是课程深度研究助手。根据提供的知识库检索结果，用 Markdown 写一份简洁研究报告。"
            "只使用检索内容中的事实；若检索为空或失败，明确说明资料不足，不要编造。"
        )
        user_prompt = (
            f"研究主题：{topic}\n"
            f"优化主题：{optimized_topic}\n\n"
            f"知识库检索结果：\n{rag_context[:12000]}\n\n"
            "请输出 Markdown 报告，包含：标题、要点摘要、正文、结论。"
        )
        model = (self.config.get("llm") or {}).get("model") or TEXT_MODEL
        resp = await _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        choice = resp.choices[0] if resp.choices else None
        return ((choice.message.content if choice and choice.message else None) or "").strip()


__all__ = ["ResearchPipeline"]
