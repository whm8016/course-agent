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
import copy
import logging
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from core.agent.registry import get_tool_schemas
from core.agentic.loop import run_agent_loop
from core.context import UnifiedContext
from core.llm.json_extract import extract_json_from_llm
from core.observability import log_flow
from core.pipeline_common import (
    build_common_context_layers,
    describe_images,
    resolve_profile_runtime,
    with_common_prompt,
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
from settings import get_settings

logger = logging.getLogger(__name__)

_SOURCE = "research"
_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "pipeline.yaml"

# research 阶段每块 loop 允许的工具（出证据，不需要 ask_user）
_RESEARCH_TOOLS = ("rag", "web_search")
# rephrase 阶段允许的工具：rag/web_search 快速确认主题边界 + ask_user（仅 WS 入口挂载）做前置澄清。
# 与 _RESEARCH_TOOLS 分开：研究块/报告不该问用户（跑到中后期再问收益为负，见 plan 调研依据）。
_REPHRASE_TOOLS = ("rag", "web_search", "ask_user")

# rephrase 挂 ask_user 后，暂停吃掉一轮、恢复后还要留一轮出 refined_topic，末轮 loop 强制 tools=None。
# 2→3：旧值 2 时若第 1 轮调 ask_user，恢复后只剩 1 轮且末轮强制无工具，refined_topic 易退化。
DEFAULT_REPHRASE_MAX_ITERATIONS = 3
DEFAULT_DECOMPOSE_NUM_SUBTOPICS = 5
DEFAULT_BLOCK_MAX_ITERATIONS = 5
DEFAULT_MAX_PARALLEL_TOPICS = 3
# 单块研究自检重试上限（含首次）：与 solve/session.py DEFAULT_MAX_REPLANS 同语义的「有界重试」。
# plan 第三批：失败重跑一次、仍失败才 mark_failed，故总尝试 = 2。
DEFAULT_BLOCK_MAX_TRIES = 2
# 块自检：有效研究摘要至少这么长（明显截断 / 空判失败）。research FINISH 摘要通常是一段话，
# 50 字（中文约 25 字）是非常宽松的下限，只挡真正的退化输出。
_BLOCK_MIN_KNOWLEDGE_CHARS = 50

# ── Observer 汇总质量门（PEOS，消融开关默认关）──────────────────────────────
# _drive_queue 后、_write_report 前，对「有内容但来源不足」的块补检索一轮。
# 与块级 _block_self_check 互补：块级只保证「非空且至少 1 个来源」，Observer 看来源是否够多。
_OBSERVER_MIN_SOURCES = 2          # 块来源少于此数视为证据不足，补检索
_OBSERVER_MAX_REFILL_BLOCKS = 8    # safety：一次最多补这么多块，防队列异常膨胀
_OBSERVER_DEFAULT = False          # 默认关，行为零变化；经 context.metadata["research_observer"] 开


def _parse_sub_topics(text: str, topic: str, num_subtopics: int) -> list[dict[str, str]]:
    """解析 decompose 的 {sub_topics:[{title,overview}]}。

    解析失败或为空时回退为单个以原主题命名的子主题，保证后续阶段可继续。
    """
    data = extract_json_from_llm(text)
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


def _block_self_check(knowledge: str, has_retrieval: bool) -> bool:
    """research 块轻量自检（plan 第三批-1）。返回 True=可用，False=判失败需重试。

    三类失败信号（plan：块内容为空 / 无引用 / 长度异常）：
    - 空 / 过短（< _BLOCK_MIN_KNOWLEDGE_CHARS）→ 明显没产出有效研究，判失败；
    - 有检索工具（rag/web_search）却无任何 ``[来源: ...]`` 标记 → 应检索却未引证，判失败
      （CitationManager 靠这些标记建参考文献，无引证的块产出的章节没有来源，违背研究
      pipeline 的引证设计）。
    纯推理块（has_retrieval=False，如禁用了知识库与联网）只校验长度。
    """
    k = (knowledge or "").strip()
    if len(k) < _BLOCK_MIN_KNOWLEDGE_CHARS:
        return False
    if has_retrieval and not _SOURCE_MARKER_RE.search(k):
        return False
    return True


def _fork_for_block(context: UnifiedContext, **overrides: Any) -> UnifiedContext:
    """为并行 research block 派生隔离 context（M-5）。

    dataclasses.replace 是浅拷贝：attachments / metadata 等可变嵌套字段仍指向原 context 的
    同一对象。run_agent_loop → _build_messages 会原地改 Attachment（doc 文件清空 base64）、
    并行 block 共享同一份时，先跑的 block 清空 base64 会让后跑的 block 丢掉附件内容。

    这里对 attachments 做深拷贝（每个 block 拿到独立的 Attachment 对象副本，互不影响），
    metadata 也深拷贝（防 block 内写 metadata 互相覆盖）。其余字段由 replace 正常处理。
    仅在 _research_block（唯一并行入口）用；串行阶段（rephrase/decompose/reporting）浅拷贝
    无并发风险，保持裸 replace 不变。
    """
    forked = replace(
        context,
        attachments=copy.deepcopy(context.attachments),
        metadata=copy.deepcopy(context.metadata),
    )
    if overrides:
        forked = replace(forked, **overrides)
    return forked


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
        # Observer 汇总质量门开关（默认关）：开则 _drive_queue 后对来源不足的块补检索一轮。
        # 经 context.metadata["research_observer"] 控制（消融实验 on/off 对比用）。
        self._observer_gate = bool(context.metadata.get("research_observer", _OBSERVER_DEFAULT))
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

        # ── Observer 汇总质量门（可选，默认关）：来源不足的块补检索一轮 ──
        if self._observer_gate:
            _t_observer = datetime.now()
            async with stream.stage("observing", source=_SOURCE):
                await self._observe_and_refill(
                    queue=queue, topic=refined_topic, context=context, stream=stream, cfg=cfg
                )
            log_flow("research.stage.observing",
                     elapsed_ms=int((datetime.now() - _t_observer).total_seconds() * 1000),
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
        system_prompt = with_common_prompt(rephrase_cfg.get("system", ""), self._layers)
        # rephrase 是主题澄清阶段：rag/web_search 快速确认边界；ask_user 仅在「澄清开关开 + WS 入口
        # （注入了 wait_for_user_reply callable）」时挂载。HTTP/IM 无 waiter 时不挂——否则 LLM 一旦
        # 调用 ask_user，loop 因无 waiter 直接结束（loop.py ask_user 分支），research 提前夭折。
        # 每块 research（出证据）/ reporting 不挂 ask_user——不打断检索与写报告。
        tools = [t for t in context.enabled_tools if t in _REPHRASE_TOOLS]
        if get_settings().research.clarify_enabled and callable(
            context.metadata.get("wait_for_user_reply")
        ):
            tools.append("ask_user")
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
            tool_schemas=get_tool_schemas(ctx) if tools else None,
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
        system_prompt = with_common_prompt(
            (decompose_cfg.get("system", "")).format(num_subtopics=self.num_subtopics),
            self._layers,
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

    async def _observe_and_refill(
        self,
        *,
        queue: DynamicTopicQueue,
        topic: str,
        context: UnifiedContext,
        stream: StreamBus,
        cfg: dict[str, Any],
    ) -> None:
        """队列级汇总质量门（PEOS Observer）：_drive_queue 后补检索来源不足的块。

        块级 _block_self_check 只保证「非空且有至少 1 个 [来源] 标记」；Observer 看的是来源
        是否够多支撑可信结论——对 knowledge 非空但 sources < _OBSERVER_MIN_SOURCES 的块补检索
        一次（串行，复用 _research_block 的 self_check+重试+抽来源）。补检索后 CitationManager
        去重兜底，故 _research_block 的 add_source 累加语义（旧来源+新来源）不会造成重复引用。

        Safety：只补一轮（每候选至多补一次）、候选数有 _OBSERVER_MAX_REFILL_BLOCKS 上限；
        开关 self._observer_gate 默认关，默认行为零变化。串行而非 gather：补检索是已有块的增强，
        候选通常少，串行简单可测、无并发 fork 顾虑。
        """
        candidates = [
            b for b in queue.blocks
            if b.knowledge and len(b.sources) < _OBSERVER_MIN_SOURCES
        ][:_OBSERVER_MAX_REFILL_BLOCKS]
        if not candidates:
            return
        log_flow("research.observer.refill",
                 block_ids=[b.block_id for b in candidates],
                 min_sources=_OBSERVER_MIN_SOURCES)
        for b in candidates:
            try:
                await self._research_block(
                    block=b, queue=queue, topic=topic,
                    context=context, stream=stream, cfg=cfg,
                )
            except Exception:
                logger.exception("research observer 补检索块 %s 异常", b.block_id)

    async def _flush_child_stream(
        self, stream: StreamBus, child: StreamBus, block: TopicBlock
    ) -> None:
        """把子 bus 缓冲的事件整体 flush 到主 stream（M-10 块级事件隔离的收尾）。

        发 progress 边界事件标记本块归属（block_id / sub_topic），再按原顺序把子 bus 的
        全部事件逐条 emit 到主 stream。因 flush 在 _research_block 末尾串行执行（gather 内
        各 block 各自走到自己的 flush），同一 block 的事件在主 stream 上是连续的一段，不会
        与其它 block 交错。
        """
        await stream.emit({
            "type": "progress", "stage": "researching",
            "status": "block_start",
            "block_id": block.block_id, "sub_topic": block.sub_topic,
        })
        for event in child._history:
            await stream.emit(event)
        await stream.emit({
            "type": "progress", "stage": "researching",
            "status": "block_end",
            "block_id": block.block_id, "sub_topic": block.sub_topic,
        })

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
        """研究单个子主题：run_agent_loop 检索 → FINISH 摘要 → 自检 → 抽来源入 sources。

        自检重试（plan 第三批-1）：单块 loop 后做一次轻量校验（空 / 无引用 / 过短即判失败），
        失败重跑，最多 DEFAULT_BLOCK_MAX_TRIES 次（含首次），仍失败才 mark_failed。重试预算与
        solve/session.py 的 bounded replan 同语义。不引入独立 Reviewer agent——按 Anthropic
        隔离原则，子块本身即隔离边界，再加一层 agent 只增成本。

        重试在单协程内串行（attempt 1 → 2），不与其它并行 block 共享可变态（block/ctx 各自独立）。
        每次尝试用独立子 StreamBus；最终只 flush「被采纳的那次」（成功 or 最后一次）的事件，
        避免把中间失败的检索过程暴露给前端造成一段块内多次起止的噪音。
        """
        queue.mark_researching(block.block_id)
        step_cfg = cfg.get("research_step") or {}
        tools = [t for t in context.enabled_tools if t in _RESEARCH_TOOLS]
        has_retrieval = bool(tools)  # 自检：有检索工具却无 [来源: ...] 标记 → 判失败需重试
        # 只有用户选了知识库（enabled_tools 含 rag）且课程存在时，才提示已挂载知识库；
        # 否则本轮工具里没有 rag，却提示"调用 rag 检索"会让 LLM 困惑（有提示无工具可调）
        kb_note = ""
        if "rag" in tools and context.course_id:
            kb_note = f"\n    已挂载知识库：{context.course_id}，调用 rag 时优先检索该库。"
        system_prompt = with_common_prompt(
            (step_cfg.get("system", "")).format(
                topic=topic,
                block_title=block.sub_topic,
                block_overview=block.overview or "(无额外说明)",
                kb_note=kb_note,
            ),
            self._layers,
        )
        siblings = "\n".join(
            f"  - {b.sub_topic}" for b in queue.blocks if b.block_id != block.block_id
        ) or "  (无)"
        user_prompt = (step_cfg.get("user_template", "")).format(sibling_topics=siblings)

        # M-10：并行 block 事件隔离——给本块一个独立子 StreamBus，run_agent_loop 的所有事件
        # （thinking/tool_call/tool_result/token）先缓冲在子 bus，块跑完后整体（按原顺序、
        # 不与其它块交错）flush 回主 stream。否则多块 asyncio.gather 并行时，各块事件按
        # await 调度交错塞进主 bus._history，前端拿到的 token 序列会跨子主题混在一起，无法
        # 归属。牺牲块内逐字真流式（research 块本就是后台检索子任务，emit_terminal_events
        # =False，非交互），换取块级事件不交错。
        final_outcome = None       # 被采纳那次（成功 or 最后一次）的 outcome
        final_stream: StreamBus | None = None  # 与之配对的子 bus（供末尾 flush）
        success = False
        for attempt in range(1, DEFAULT_BLOCK_MAX_TRIES + 1):
            # 每次尝试重新 fork：run_agent_loop→_build_messages 会原地清掉 fork 的 doc base64，
            # 复用同一 fork 会让重试丢失附件内容（_fork_for_block deepcopy 的意义所在）。
            # 重试是失败路径（罕见），多一次 deepcopy 可接受。
            ctx = _fork_for_block(
                context,
                user_message=user_prompt,
                conversation_history=[],
                enabled_tools=tools,
                mode="research",
            )
            child_stream = StreamBus()
            try:
                outcome = await run_agent_loop(
                    context=ctx,
                    stream=child_stream,
                    system_prompt=system_prompt,
                    tool_schemas=get_tool_schemas(ctx) if tools else None,
                    max_iterations=self.block_max_iterations,
                    client=self._rt.client,
                    model=self._rt.text_model,
                    binding=self._rt.binding,
                    emit_terminal_events=False,
                )
            except Exception:
                logger.exception("research 块 %s 第 %d/%d 次执行异常",
                                 block.block_id, attempt, DEFAULT_BLOCK_MAX_TRIES)
                final_stream = child_stream  # 留作兜底 flush（让前端能看到最后一次的轨迹）
                continue
            final_outcome = outcome
            final_stream = child_stream
            if outcome.completed and _block_self_check(outcome.final_text or "", has_retrieval):
                success = True
                break
            log_flow("research.block.self_check_failed",
                     block_id=block.block_id, attempt=attempt,
                     max_tries=DEFAULT_BLOCK_MAX_TRIES,
                     knowledge_chars=len((outcome.final_text or "").strip()),
                     completed=bool(outcome.completed))
            logger.warning("research 块 %s 第 %d/%d 次自检不过（空/无引用/过短），重试",
                           block.block_id, attempt, DEFAULT_BLOCK_MAX_TRIES)

        # 采纳最终 outcome 的内容（无论成败：reporting 阶段只取 knowledge 非空的块；
        # mark_failed 的块若有内容仍可作为降级证据）。sources 仅从采纳的这一份抽，重试不累加。
        if final_outcome is not None:
            block.knowledge = (final_outcome.final_text or "").strip()
            for trace in _extract_traces_from_knowledge(block.knowledge, block):
                block.add_source(trace)

        # flush 被采纳那次的子 bus（成功 or 最后一次失败/异常），保持块级事件不交错
        if final_stream is not None:
            await self._flush_child_stream(stream, final_stream, block)

        if success:
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
        system_prompt = with_common_prompt(report_cfg.get("system", ""), self._layers)
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
        data = extract_json_from_llm(outcome.final_text)
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
            system_prompt=with_common_prompt(system_prompt, self._layers),
            tool_schemas=None,
            max_iterations=1,
            client=self._rt.client,
            model=self._rt.text_model,
            binding=self._rt.binding,
            emit_terminal_events=False,
        )
        return (outcome.final_text or "").strip()


__all__ = ["ResearchPipeline"]
