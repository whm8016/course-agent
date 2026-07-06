"""ResearchPipeline — 式 4 阶段深度研究（tool_calls 版）。

agents/research/pipeline.py 的 rephrase → decompose → research →
reporting 四阶段，去除 label 协议层：每阶段用 run_agent_loop（tool_calls），
按 Phase 2/3 范式做 dataclasses.replace context 隔离 + stream.stage() + prompt YAML。

阶段：
  rephrase    → run_agent_loop mini（max 2，可用 rag/web_search）改写主题 → refined_topic
  decompose   → run_agent_loop 单轮（prompt 要求 JSON）出 {sub_topics:[{title,overview}]}
                → 解析入 DynamicTopicQueue
  research    → DynamicTopicQueue 代码管理 + asyncio.gather 并行跑多个 run_agent_loop
                （每子主题一个，rag/web_search 检索），FINISH 摘要落入 TopicBlock.knowledge
  reporting   → 多段 run_agent_loop（outline→intro→sections×N→conclusion），CitationManager
                从 research 阶段的 [来源: ...] 标记去重编号，section 内联 [n] + 末尾附录

简化项（见返回说明）：
- Note Agent 省略：run_agent_loop 无 per-tool hook，不为它改 loop；每块 loop 的 FINISH 摘要
  本身就是该块的知识整合，sources 从摘要里的 ``[来源: ...]`` 标记提取。
- Citation 用基础 ``[n]`` 编号（按 url/title/source 去重），非 的 CIT-x-yy。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agentic.loop import _get_tool_schemas, run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.pipeline_common import (
    assemble_common_context,
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
)
from core.prompt_loader import load_prompt_dict
from core.research.citation_manager import CitationManager
from core.research.data_structures import (
    DEFAULT_QUEUE_MAX_LENGTH,
    DynamicTopicQueue,
    TopicBlock,
    ToolTrace,
)
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

_SOURCE = "research"
_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "pipeline.yaml"

# research 阶段每块 loop 允许的工具（出证据，不需要 ask_user）
_RESEARCH_TOOLS = ("rag", "web_search")

DEFAULT_REPHRASE_MAX_ITERATIONS = 2
DEFAULT_DECOMPOSE_NUM_SUBTOPICS = 5
DEFAULT_BLOCK_MAX_ITERATIONS = 5
DEFAULT_MAX_PARALLEL_TOPICS = 3


# ── JSON 解析（参照 question/pipeline.py 的 _extract_json）─────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """从 LLM 输出提取 JSON 对象，容忍 markdown fence / 前后噪声。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start == -1:
        return {}
    try:
        data, _end = json.JSONDecoder().raw_decode(cleaned, start)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_sub_topics(text: str, topic: str, num_subtopics: int) -> list[dict[str, str]]:
    """解析 decompose 的 {sub_topics:[{title,overview}]}。

    解析失败或为空时回退为单个以原主题命名的子主题，保证后续阶段可继续。
    """
    data = _extract_json(text)
    raw = data.get("sub_topics") or data.get("subtopics") or []
    if not isinstance(raw, list):
        raw = []
    items: list[dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, dict):
            title = str(entry.get("title") or entry.get("topic") or "").strip()
            overview = str(entry.get("overview") or entry.get("description") or "").strip()
        elif isinstance(entry, str):
            title = entry.strip()
            overview = ""
        else:
            continue
        if title:
            items.append({"title": title, "overview": overview})
        if len(items) >= num_subtopics:
            break
    if not items:
        items = [{"title": topic, "overview": ""}]
    return items


# ── 来源标记提取（从 research loop 的 FINISH 摘要里抽 ``[来源: ...]``）─────────

# 形如 [来源: https://x.com] 或 [来源: 知识库: 导数定义] 或 [来源: 文件.pdf]
_SOURCE_MARKER_RE = re.compile(r"\[\s*来源\s*[:：]\s*(?P<src>[^\]]+?)\s*\]")
_URL_RE = re.compile(r"https?://\S+")


def _extract_traces_from_knowledge(
    knowledge: str, block: TopicBlock
) -> list[ToolTrace]:
    """从一块 research 摘要里提取所有 ``[来源: ...]`` 标记为 ToolTrace 列表。

    每条标记的 src 可能是 url（→ web_search）或纯文本（→ rag / 通用）。
    """
    traces: list[ToolTrace] = []
    for m in _SOURCE_MARKER_RE.finditer(knowledge or ""):
        src = m.group("src").strip()
        if not src:
            continue
        is_url = bool(_URL_RE.match(src))
        query = ""
        url = ""
        title = ""
        if is_url:
            url = src
            tool_type = "web_search"
            # 域名作为 title 提示
            title = _URL_RE.findall(src)[0].split("/")[2] if "//" in src else src
        else:
            tool_type = "rag"
            title = src
            query = src
        traces.append(
            ToolTrace(
                tool_type=tool_type,
                query=query,
                summary=src,
                source=url or title,
            )
        )
    return traces


def _strip_source_markers(text: str) -> str:
    """把摘要里的 ``[来源: ...]`` 标记去掉，留干净正文（用于报告 evidence 注入）。"""
    return _SOURCE_MARKER_RE.sub("", text or "").strip()


# ── Pipeline ───────────────────────────────────────────────────────────────────


class ResearchPipeline:
    """4 阶段深度研究 pipeline：rephrase → decompose → research → reporting。"""

    def __init__(
        self,
        *,
        reports_dir: str = "./data/research/reports",
        num_subtopics: int = DEFAULT_DECOMPOSE_NUM_SUBTOPICS,
        max_parallel: int = DEFAULT_MAX_PARALLEL_TOPICS,
        block_max_iterations: int = DEFAULT_BLOCK_MAX_ITERATIONS,
        queue_max_length: int = DEFAULT_QUEUE_MAX_LENGTH,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.num_subtopics = max(1, int(num_subtopics))
        self.max_parallel = max(1, int(max_parallel))
        self.block_max_iterations = max(2, int(block_max_iterations))
        self.queue_max_length = max(1, int(queue_max_length))

    def _with_common(self, task_system: str, layers=None) -> str:
        """叠加通用上下文层到 task system（solve/research/quiz 共享语义）。

        layers 默认取 self._layers（run() 开头算一次，多阶段 + asyncio.gather 并行块
        只读复用同一份不可变 dataclass，无竞态）。通用层为空时原样返回。
        """
        common = assemble_common_context(layers if layers is not None else self._layers)
        return f"{task_system}\n\n{common}" if common else task_system

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    async def run(
        self,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
    ) -> dict[str, Any]:
        started = datetime.now()
        research_id = f"research_{started.strftime('%Y%m%d_%H%M%S')}"
        cfg = load_prompt_dict(_PROMPT_PATH)
        topic = (topic or "").strip() or context.user_message.strip()
        # 解析对话供应商 runtime + 通用上下文层（四条 pipeline 共享步骤，pipeline_common）
        # 不可变 dataclass：5 个阶段方法 + asyncio.gather 并行块只读共享，无竞态
        self._rt = await resolve_profile_runtime(context.llm_profile_id, context.user_id)
        self._layers = await build_common_context_layers(context)
        log_flow("research.pipeline.start", research_id=research_id, topic=topic[:120])

        # ── Phase 1: rephrase ────────────────────────────────────────────
        _t_phase = datetime.now()
        async with stream.stage("rephrase", source=_SOURCE):
            refined_topic = await self._rephrase(
                topic=topic, context=context, stream=stream, cfg=cfg
            )
        log_flow("research.stage.rephrase",
                 elapsed_ms=int((datetime.now() - _t_phase).total_seconds() * 1000),
                 refined_topic=refined_topic[:80])

        # ── Phase 2: decompose ───────────────────────────────────────────
        _t_phase = datetime.now()
        async with stream.stage("decompose", source=_SOURCE):
            sub_topics = await self._decompose(
                topic=refined_topic, context=context, stream=stream, cfg=cfg
            )
        log_flow("research.stage.decompose",
                 elapsed_ms=int((datetime.now() - _t_phase).total_seconds() * 1000),
                 sub_topics=len(sub_topics))

        # ── Phase 3: research（动态队列 + 并行）──────────────────────────
        queue = DynamicTopicQueue(research_id, max_length=self.queue_max_length)
        for sub in sub_topics:
            if queue.is_full():
                break
            # 简单去重：与已有块过于相似的跳过
            if queue.find_similar(sub["title"]) is not None:
                continue
            queue.add_block(sub["title"], sub["overview"])

        _t_phase = datetime.now()
        async with stream.stage("researching", source=_SOURCE):
            await self._drive_queue(
                queue=queue, topic=refined_topic, context=context, stream=stream, cfg=cfg
            )
        log_flow("research.stage.researching",
                 elapsed_ms=int((datetime.now() - _t_phase).total_seconds() * 1000),
                 blocks=queue.statistics().get("total", 0))

        # ── Phase 4: reporting ───────────────────────────────────────────
        _t_phase = datetime.now()
        async with stream.stage("reporting", source=_SOURCE):
            report = await self._write_report(
                topic=refined_topic,
                queue=queue,
                context=context,
                stream=stream,
                cfg=cfg,
            )
        log_flow("research.stage.reporting",
                 elapsed_ms=int((datetime.now() - _t_phase).total_seconds() * 1000),
                 report_chars=len(report))

        report_file = self.reports_dir / f"{research_id}.md"
        try:
            report_file.write_text(report, encoding="utf-8")
        except OSError:
            logger.warning("research report 落盘失败：%s", report_file)

        elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        log_flow("research.pipeline.complete", research_id=research_id,
                 elapsed_ms=elapsed_ms, report_chars=len(report))
        stats = queue.statistics()
        return {
            "research_id": research_id,
            "topic": refined_topic,
            "report": report,
            "metadata": {
                "research_id": research_id,
                "topic": refined_topic,
                "elapsed_ms": elapsed_ms,
                "sub_topics": queue.list_titles(),
                "block_count": stats["total"],
                "sources": stats["sources"],
                "kb_name": context.course_id or None,
                "stages": ["rephrase", "decompose", "researching", "reporting"],
            },
        }

    # ------------------------------------------------------------------
    # Phase 1: rephrase
    # ------------------------------------------------------------------
    async def _rephrase(
        self,
        *,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> str:
        rephrase_cfg = cfg.get("rephrase") or {}
        system_prompt = self._with_common(rephrase_cfg.get("system", ""))
        tools = [t for t in context.enabled_tools if t in _RESEARCH_TOOLS]
        ctx = replace(
            context,
            user_message=await describe_images(
                context, f"用户的研究主题：\n\n{topic}", self._rt
            ),
            conversation_history=[],
            enabled_tools=tools,
            mode="research",
        )
        outcome = await run_agent_loop(
            context=ctx,
            stream=stream,
            system_prompt=system_prompt,
            tool_schemas=_get_tool_schemas(ctx) if tools else None,
            max_iterations=DEFAULT_REPHRASE_MAX_ITERATIONS,
            client=self._rt.client,
            model=self._rt.text_model,
            binding=self._rt.binding,
            emit_terminal_events=False,
        )
        refined = (outcome.final_text or "").strip()
        return refined or topic

    # ------------------------------------------------------------------
    # Phase 2: decompose
    # ------------------------------------------------------------------
    async def _decompose(
        self,
        *,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> list[dict[str, str]]:
        decompose_cfg = cfg.get("decompose") or {}
        system_prompt = self._with_common(
            (decompose_cfg.get("system", "")).format(num_subtopics=self.num_subtopics)
        )
        ctx = replace(
            context,
            user_message=(
                f"精炼后的研究主题：\n\n{topic}\n\n"
                f"现在产出约 {self.num_subtopics} 个子主题的 JSON 大纲。"
            ),
            conversation_history=[],
            enabled_tools=[],
            mode="research",
        )
        outcome = await run_agent_loop(
            context=ctx,
            stream=stream,
            system_prompt=system_prompt,
            tool_schemas=None,
            max_iterations=1,
            client=self._rt.client,
            model=self._rt.text_model,
            binding=self._rt.binding,
            emit_terminal_events=False,
        )
        return _parse_sub_topics(outcome.final_text, topic, self.num_subtopics)

    # ------------------------------------------------------------------
    # Phase 3: research — 调度 + 单块 loop
    # ------------------------------------------------------------------
    async def _drive_queue(
        self,
        *,
        queue: DynamicTopicQueue,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> None:
        """并行 drained 队列：每轮取 max_parallel 个 pending 块并行研究。

        每块 loop 内部多轮检索；loop 结束后从 FINISH 摘要里抽 [来源: ...] 入 sources。
        """
        rounds = 0
        safety_cap = max(20, len(queue.blocks) * 4)
        while not queue.all_done():
            pending = queue.get_pending()
            if not pending:
                break
            batch = pending[: self.max_parallel]
            await asyncio.gather(
                *[
                    self._research_block(
                        block=b,
                        queue=queue,
                        topic=topic,
                        context=context,
                        stream=stream,
                        cfg=cfg,
                    )
                    for b in batch
                ],
                return_exceptions=True,
            )
            rounds += 1
            if rounds > safety_cap:
                logger.warning("research 调度超过安全上限 %d 轮，中止", safety_cap)
                break

    async def _research_block(
        self,
        *,
        block: TopicBlock,
        queue: DynamicTopicQueue,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> None:
        """研究单个子主题：run_agent_loop 检索 → FINISH 摘要 → 抽来源入 sources。"""
        queue.mark_researching(block.block_id)
        step_cfg = cfg.get("research_step") or {}
        tools = [t for t in context.enabled_tools if t in _RESEARCH_TOOLS]
        # 只有用户选了知识库（enabled_tools 含 rag）且课程存在时，才提示已挂载知识库；
        # 否则本轮工具里没有 rag，却提示"调用 rag 检索"会让 LLM 困惑（有提示无工具可调）
        kb_note = ""
        if "rag" in tools and context.course_id:
            kb_note = f"\n    已挂载知识库：{context.course_id}，调用 rag 时优先检索该库。"
        system_prompt = self._with_common(
            (step_cfg.get("system", "")).format(
                topic=topic,
                block_title=block.sub_topic,
                block_overview=block.overview or "(无额外说明)",
                kb_note=kb_note,
            )
        )
        siblings = "\n".join(
            f"  - {b.sub_topic}" for b in queue.blocks if b.block_id != block.block_id
        ) or "  (无)"
        user_prompt = (step_cfg.get("user_template", "")).format(sibling_topics=siblings)

        ctx = replace(
            context,
            user_message=user_prompt,
            conversation_history=[],
            enabled_tools=tools,
            mode="research",
        )
        try:
            outcome = await run_agent_loop(
                context=ctx,
                stream=stream,
                system_prompt=system_prompt,
                tool_schemas=_get_tool_schemas(ctx) if tools else None,
                max_iterations=self.block_max_iterations,
                client=self._rt.client,
                model=self._rt.text_model,
                binding=self._rt.binding,
                emit_terminal_events=False,
            )
        except Exception:
            logger.exception("research 块 %s 失败", block.block_id)
            queue.mark_failed(block.block_id)
            return

        block.knowledge = (outcome.final_text or "").strip()
        for trace in _extract_traces_from_knowledge(block.knowledge, block):
            block.add_source(trace)
        if outcome.completed and block.knowledge:
            queue.mark_completed(block.block_id)
        else:
            queue.mark_failed(block.block_id)

    # ------------------------------------------------------------------
    # Phase 4: reporting — outline → intro → sections → conclusion
    # ------------------------------------------------------------------
    async def _write_report(
        self,
        *,
        topic: str,
        queue: DynamicTopicQueue,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> str:
        blocks = [b for b in queue.blocks if b.knowledge]

        # 收集所有来源到 CitationManager（去重 + 编号）
        citations = CitationManager()
        for b in blocks:
            for trace in b.sources:
                citations.add_source(
                    url=trace.source if trace.tool_type == "web_search" else "",
                    title=trace.source if trace.tool_type != "web_search" else "",
                    tool_type=trace.tool_type,
                    query=trace.query,
                    snippet=trace.summary,
                )

        outline = await self._gen_report_outline(
            topic=topic, blocks=blocks, context=context, stream=stream, cfg=cfg
        )
        report_title = outline.get("title") or topic
        sections = outline.get("sections") or []
        # 回退：每块一节
        if not sections:
            sections = [
                {
                    "id": b.block_id,
                    "title": b.sub_topic,
                    "intent": b.overview,
                    "block_ids": [b.block_id],
                }
                for b in blocks
            ]

        parts: list[str] = [f"# {report_title}"]

        intro = await self._write_intro(
            topic=topic,
            title=report_title,
            sections=sections,
            context=context,
            stream=stream,
            cfg=cfg,
        )
        if intro:
            parts.append(intro)

        section_bodies: list[str] = []
        for idx, section in enumerate(sections, start=1):
            section_number = idx + 1  # 引言为 1，章节从 2 起
            body = await self._write_section(
                section=section,
                section_number=section_number,
                topic=topic,
                report_title=report_title,
                blocks=blocks,
                citations=citations,
                context=context,
                stream=stream,
                cfg=cfg,
            )
            if body:
                parts.append(body)
                section_bodies.append(body)

        conclusion_number = len(sections) + 2
        conclusion = await self._write_conclusion(
            topic=topic,
            title=report_title,
            sections=sections,
            section_bodies=section_bodies,
            section_number=conclusion_number,
            context=context,
            stream=stream,
            cfg=cfg,
        )
        if conclusion:
            parts.append(conclusion)

        references = citations.render_references()
        if references:
            parts.append(references)

        return "\n\n".join(p for p in parts if p and p.strip())

    async def _gen_report_outline(
        self,
        *,
        topic: str,
        blocks: list[TopicBlock],
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        report_cfg = cfg.get("report_outline") or {}
        summaries = []
        for b in blocks:
            preview = _strip_source_markers(b.knowledge).split("\n\n")[0][:400]
            summaries.append(f"- [{b.block_id}] {b.sub_topic}\n  {preview}")
        system_prompt = self._with_common(report_cfg.get("system", ""))
        user_prompt = (report_cfg.get("user_template", "")).format(
            topic=topic,
            block_summaries="\n".join(summaries) or "(无研究块)",
        )
        ctx = replace(
            context,
            user_message=user_prompt,
            conversation_history=[],
            enabled_tools=[],
            mode="research",
        )
        outcome = await run_agent_loop(
            context=ctx,
            stream=stream,
            system_prompt=system_prompt,
            tool_schemas=None,
            max_iterations=1,
            client=self._rt.client,
            model=self._rt.text_model,
            binding=self._rt.binding,
            emit_terminal_events=False,
        )
        data = _extract_json(outcome.final_text)
        if isinstance(data, dict) and (data.get("title") or data.get("sections")):
            return data
        return {"title": topic, "sections": []}

    async def _write_intro(
        self,
        *,
        topic: str,
        title: str,
        sections: list[dict[str, Any]],
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> str:
        intro_cfg = (cfg.get("report_intro") or {})
        system_prompt = intro_cfg.get("system", "")
        overview = "\n".join(
            f"- {s.get('title', '')}: {s.get('intent', '')}".rstrip(": ")
            for s in sections
        )
        user_prompt = (intro_cfg.get("user_template", "")).format(
            topic=topic, title=title, sections_overview=overview or "(无)"
        )
        return await self._one_shot_report(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            stream=stream,
        )

    async def _write_section(
        self,
        *,
        section: dict[str, Any],
        section_number: int,
        topic: str,
        report_title: str,
        blocks: list[TopicBlock],
        citations: CitationManager,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> str:
        section_cfg = cfg.get("report_section") or {}
        valid_ids = {b.block_id for b in blocks}
        by_id = {b.block_id: b for b in blocks}
        block_ids = [
            bid for bid in (section.get("block_ids") or []) if bid in valid_ids
        ] or [b.block_id for b in blocks]

        # 渲染本节可引用的来源 → 编号映射 + evidence
        citation_map_lines: list[str] = []
        evidence_chunks: list[str] = []
        for bid in block_ids:
            b = by_id.get(bid)
            if b is None:
                continue
            chunk_lines = [f"### 块 [{b.block_id}] {b.sub_topic}"]
            for trace in b.sources:
                num = citations.add_source(
                    url=trace.source if trace.tool_type == "web_search" else "",
                    title=trace.source if trace.tool_type != "web_search" else "",
                    tool_type=trace.tool_type,
                    query=trace.query,
                    snippet=trace.summary,
                )
                citation_map_lines.append(f"- [{num}] {trace.source}")
                chunk_lines.append(
                    f"- 来源 [{num}]（{trace.tool_type}，query: {trace.query or 'N/A'}）"
                )
            knowledge_clean = _strip_source_markers(b.knowledge)
            if knowledge_clean:
                chunk_lines.append(knowledge_clean[:1500])
            evidence_chunks.append("\n".join(chunk_lines))

        system_prompt = (section_cfg.get("system", "")).format(section_number=section_number)
        user_prompt = (section_cfg.get("user_template", "")).format(
            topic=topic,
            report_title=report_title,
            section_id=section.get("id", ""),
            section_title=section.get("title", ""),
            section_number=section_number,
            section_intent=section.get("intent", "") or "(无额外指引)",
            citation_map="\n".join(citation_map_lines) or "(本节无外部来源)",
            evidence="\n\n".join(evidence_chunks) or "(无证据)",
        )
        return await self._one_shot_report(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            stream=stream,
        )

    async def _write_conclusion(
        self,
        *,
        topic: str,
        title: str,
        sections: list[dict[str, Any]],
        section_bodies: list[str],
        section_number: int,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> str:
        concl_cfg = cfg.get("report_conclusion") or {}
        recap_chunks: list[str] = []
        for sec, body in zip(sections, section_bodies, strict=False):
            snippet = _strip_source_markers(body).split("\n\n", 1)[0]
            recap_chunks.append(f"### {sec.get('title', '')}\n{snippet[:300]}")
        system_prompt = (concl_cfg.get("system", "")).format(section_number=section_number)
        user_prompt = (concl_cfg.get("user_template", "")).format(
            topic=topic,
            title=title,
            section_number=section_number,
            sections_recap="\n\n".join(recap_chunks) or "(无章节正文)",
        )
        return await self._one_shot_report(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            stream=stream,
        )

    async def _one_shot_report(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: UnifiedContext,
        stream: StreamBus,
    ) -> str:
        """报告子段通用 runner：单轮 run_agent_loop，禁用工具，取 final_text。"""
        ctx = replace(
            context,
            user_message=user_prompt,
            conversation_history=[],
            enabled_tools=[],
            mode="research",
        )
        outcome = await run_agent_loop(
            context=ctx,
            stream=stream,
            system_prompt=self._with_common(system_prompt),
            tool_schemas=None,
            max_iterations=1,
            client=self._rt.client,
            model=self._rt.text_model,
            binding=self._rt.binding,
            emit_terminal_events=False,
        )
        return (outcome.final_text or "").strip()


__all__ = ["ResearchPipeline"]
