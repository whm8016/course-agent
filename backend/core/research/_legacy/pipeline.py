"""ResearchPipeline - DR-in-KG 2.0 research workflow (faithful port from DeepTutor).

Adaptations from the original:
- ``get_tool_registry()`` → local ``ToolRegistry`` instance (already embedded in each agent)
- ``PROJECT_ROOT`` / ``get_llm_config()`` → omitted (callers supply config/api_key directly)
- DeepTutor token_tracker import is wrapped in a try/except (optional)
- The ``main()`` CLI helper is removed; only ``ResearchPipeline`` is exported.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import inspect
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable

from .agents import (
    DecomposeAgent,
    ManagerAgent,
    NoteAgent,
    RephraseAgent,
    ReportingAgent,
    ResearchAgent,
)
from .citation_manager import CitationManager
from .data_structures import DynamicTopicQueue, TopicStatus
from .tool_registry import ToolRegistry
from .trace import new_call_id


class ResearchPipeline:
    """DR-in-KG 2.0 Research workflow"""

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str = "",
        base_url: str = "",
        api_version: str | None = None,
        research_id: str | None = None,
        kb_name: str | None = None,
        progress_callback: Callable[[dict[str, Any]], Any] | None = None,
        trace_callback: Callable[[dict[str, Any]], Any] | None = None,
        pre_confirmed_outline: list[dict[str, str]] | None = None,
        attachments: list[Any] | None = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.trace_callback = trace_callback
        self.pre_confirmed_outline = pre_confirmed_outline
        self.attachments = list(attachments or [])

        if kb_name is not None:
            self.config.setdefault("rag", {})["kb_name"] = kb_name

        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version or config.get("llm", {}).get("api_version")
        self.input_topic: str | None = None
        self.optimized_topic: str | None = None

        if research_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.research_id = f"research_{timestamp}"
        else:
            self.research_id = research_id

        system_config = config.get("system", {})
        self.cache_dir = (
            Path(system_config.get("output_base_dir", "./data/research/workspace"))
            / self.research_id
        )
        self.reports_dir = Path(
            system_config.get("reports_dir", "./data/research/reports")
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.plan_progress_file = self.cache_dir / "planning_progress.json"
        self.report_progress_file = self.cache_dir / "reporting_progress.json"
        self.queue_progress_file = self.cache_dir / "queue_progress.json"
        self._stage_events: dict[str, list[dict[str, Any]]] = {
            "planning": [],
            "reporting": [],
        }

        queue_cfg = config.get("queue", {})
        self.queue = DynamicTopicQueue(
            self.research_id,
            max_length=queue_cfg.get("max_length"),
            state_file=str(self.queue_progress_file),
        )

        self._init_logger()
        self.agents: dict[str, Any] = {}
        self._init_agents()

        self._tool_registry = ToolRegistry(
            config=config, kb_name=config.get("rag", {}).get("kb_name")
        )
        self.citation_manager = CitationManager(self.research_id, self.cache_dir)
        self._progress_file_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_logger(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _init_agents(self) -> None:
        self.agents = {
            "rephrase": RephraseAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
            "decompose": DecomposeAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
            "manager": ManagerAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
            "research": ResearchAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
            "note": NoteAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
            "reporting": ReportingAgent(self.config, self.api_key, self.base_url, api_version=self.api_version),
        }
        if self.trace_callback is not None:
            for agent in self.agents.values():
                if hasattr(agent, "set_trace_callback"):
                    agent.set_trace_callback(self.trace_callback)
        self.agents["manager"].set_queue(self.queue)

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    async def _emit_trace_event(self, payload: dict[str, Any]) -> None:
        cb = self.trace_callback
        if cb is None:
            return
        result = cb(payload)
        if inspect.isawaitable(result):
            await result

    async def _call_tool_with_timeout(self, coro: Any, timeout: float = 60.0, tool_name: str = "tool") -> Any:
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Tool {tool_name} timed out after {timeout}s")
            raise

    async def _call_tool_with_retry(
        self,
        tool_func: Any,
        *args: Any,
        max_retries: int = 2,
        timeout: float = 60.0,
        tool_name: str = "tool",
        **kwargs: Any,
    ) -> Any:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(tool_func):
                    return await self._call_tool_with_timeout(tool_func(*args, **kwargs), timeout=timeout, tool_name=tool_name)
                import functools
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(loop.run_in_executor(None, functools.partial(tool_func, *args, **kwargs)), timeout=timeout)
            except (asyncio.TimeoutError, Exception) as exc:
                last_error = exc
                if attempt < max_retries:
                    self.logger.warning(f"Tool {tool_name} attempt {attempt + 1} failed: {exc}, retrying...")
                    await asyncio.sleep(1)
        self.logger.error(f"Tool {tool_name} failed after {max_retries + 1} attempts: {last_error}")
        raise last_error if last_error else RuntimeError(f"{tool_name} failed")

    async def _call_tool(self, tool_type: str, query: str) -> str:
        """Call tool and return raw JSON string answer."""
        tool_type = (tool_type or "").lower()
        call_id = new_call_id("research-tool")
        await self._emit_trace_event({
            "event": "tool_call", "phase": "researching", "call_id": call_id,
            "label": f"Use {tool_type or 'tool'}", "call_kind": "tool_execution",
            "tool_name": tool_type or "tool", "tool_args": {"query": query}, "query": query,
        })

        tool_config = self.config.get("researching", {})
        default_timeout = tool_config.get("tool_timeout", 60)
        max_retries = tool_config.get("tool_max_retries", 2)

        try:
            if tool_type in ("rag_hybrid", "rag_naive", "rag"):
                rag_cfg = self.config.get("rag", {}) or {}
                kb_name = rag_cfg.get("kb_name")
                if not kb_name:
                    return json.dumps({"status": "skipped", "reason": "no_kb_selected", "tool": "rag", "query": query}, ensure_ascii=False)
                result = await self._call_tool_with_retry(
                    self._tool_registry.execute, "rag",
                    query=query, kb_name=kb_name,
                    max_retries=max_retries, timeout=default_timeout, tool_name="rag",
                )
            elif tool_type == "web_search":
                result = await self._call_tool_with_retry(
                    self._tool_registry.execute, tool_type,
                    query=query, output_dir=str(self.cache_dir),
                    max_retries=max_retries, timeout=default_timeout, tool_name="web_search",
                )
            elif tool_type == "paper_search":
                years_limit = tool_config.get("paper_search_years_limit", 3)
                result = await self._call_tool_with_retry(
                    self._tool_registry.execute, tool_type,
                    query=query, max_results=3, years_limit=years_limit,
                    max_retries=max_retries, timeout=default_timeout, tool_name="paper_search",
                )
            elif tool_type in {"run_code", "code_execution", "code_execute"}:
                result = await self._call_tool_with_retry(
                    self._tool_registry.execute, "run_code",
                    intent=query, task_id=self.research_id,
                    max_retries=1, timeout=30, tool_name="run_code",
                )
            else:
                return json.dumps({"status": "failed", "reason": "unknown_tool", "tool": tool_type, "query": query}, ensure_ascii=False)
        except Exception as exc:
            failure = json.dumps({"status": "failed", "error": str(exc), "tool": tool_type, "query": query}, ensure_ascii=False)
            await self._emit_trace_event({"event": "tool_result", "phase": "researching", "call_id": call_id,
                                          "tool_name": tool_type, "result": failure, "state": "error", "query": query})
            return failure

        # Serialize result - handle both ToolResult objects and plain strings
        if hasattr(result, "content"):
            payload: dict[str, Any] = dict(getattr(result, "metadata", None) or {})
            payload.setdefault("content", result.content)
            if getattr(result, "sources", None):
                payload.setdefault("sources", result.sources)
            payload.setdefault("success", getattr(result, "success", True))
            serialized = json.dumps(payload, ensure_ascii=False)
        elif isinstance(result, str):
            serialized = result
        else:
            serialized = json.dumps(result, ensure_ascii=False, default=str)

        await self._emit_trace_event({"event": "tool_result", "phase": "researching", "call_id": call_id,
                                      "tool_name": tool_type, "result": serialized, "state": "complete", "query": query})
        return serialized

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, topic: str) -> dict[str, Any]:
        self.input_topic = topic
        self.logger.info(f"DR-in-KG 2.0 | topic={topic} | id={self.research_id}")

        try:
            optimized_topic = await self._phase1_planning(topic)
            await self._phase2_researching()
            report_result = await self._phase3_reporting(optimized_topic)

            report_file = self.reports_dir / f"{self.research_id}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_result["report"])

            queue_file = self.cache_dir / "queue.json"
            self.queue.save_to_json(str(queue_file))

            if "outline" in report_result:
                outline_file = self.cache_dir / "outline.json"
                with open(outline_file, "w", encoding="utf-8") as f:
                    json.dump(report_result["outline"], f, ensure_ascii=False, indent=2)

            metadata = {
                "research_id": self.research_id,
                "topic": topic,
                "optimized_topic": optimized_topic,
                "statistics": self.queue.get_statistics(),
                "report_word_count": report_result["word_count"],
                "completed_at": datetime.now().isoformat(),
            }
            metadata_file = self.reports_dir / f"{self.research_id}_metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return {
                "research_id": self.research_id,
                "topic": topic,
                "report": report_result["report"],
                "final_report_path": str(report_file),
                "metadata": metadata,
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.logger.exception(f"Research failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------

    async def _phase1_planning(self, topic: str) -> str:
        self._log_progress("planning", "planning_started", user_topic=topic)

        if self.pre_confirmed_outline:
            optimized_topic = topic
            self.optimized_topic = optimized_topic
            self._log_progress("planning", "rephrase_skipped", optimized_topic=optimized_topic, reason="pre-confirmed outline provided")
            for item in self.pre_confirmed_outline:
                title = (item.get("title") or "").strip()
                if not title:
                    continue
                try:
                    block = self.queue.add_block(sub_topic=title, overview=item.get("overview", ""))
                    self._log_progress("planning", "queue_seeded", block_id=block.block_id, sub_topic=block.sub_topic)
                except RuntimeError as err:
                    self.logger.warning(f"Queue capacity reached: {err}")
                    break
            self.agents["manager"].set_primary_topic(optimized_topic)
            self._log_progress("planning", "planning_completed", total_blocks=self.queue.get_statistics()["total_blocks"])
            return optimized_topic

        rephrase_config = self.config.get("planning", {}).get("rephrase", {})
        rephrase_enabled = rephrase_config.get("enabled", True)

        planning_attachments_used = False
        if rephrase_enabled:
            max_iterations = rephrase_config.get("max_iterations", 3)
            rephrase_result: dict[str, Any] = {"topic": topic}
            current_topic = topic
            for iteration in range(max_iterations):
                rephrase_result = await self.agents["rephrase"].process(
                    current_topic, iteration=iteration, previous_result=rephrase_result,
                    attachments=self.attachments if iteration == 0 else None,
                )
                if iteration == 0:
                    planning_attachments_used = True
                next_topic = str(rephrase_result.get("topic", "") or "").strip()
                if not next_topic or next_topic == current_topic.strip():
                    current_topic = next_topic or current_topic
                    break
                current_topic = next_topic
            optimized_topic = current_topic or topic
            self._log_progress("planning", "rephrase_completed", optimized_topic=optimized_topic)
        else:
            optimized_topic = topic
            self._log_progress("planning", "rephrase_skipped", optimized_topic=optimized_topic, reason="disabled")

        self.optimized_topic = optimized_topic

        decompose_config = self.config.get("planning", {}).get("decompose", {})
        mode = decompose_config.get("mode", "manual")
        if mode == "auto":
            num_subtopics = decompose_config.get("auto_max_subtopics", decompose_config.get("initial_subtopics", 5))
        else:
            num_subtopics = decompose_config.get("initial_subtopics", 5)

        self._log_progress("planning", "decompose_started", requested_subtopics=num_subtopics, mode=mode)
        self.agents["decompose"].set_citation_manager(self.citation_manager)

        decompose_result = await self.agents["decompose"].process(
            topic=optimized_topic, num_subtopics=num_subtopics, mode=mode,
            attachments=self.attachments if not planning_attachments_used else None,
        )
        self._log_progress("planning", "decompose_completed", generated_subtopics=decompose_result.get("total_subtopics", 0))

        try:
            step1_path = self.cache_dir / "step1_planning.json"
            with open(step1_path, "w", encoding="utf-8") as f:
                json.dump({
                    "main_topic": optimized_topic,
                    "sub_queries": decompose_result.get("sub_queries", []),
                    "sub_topics": decompose_result.get("sub_topics", []),
                    "total_subtopics": decompose_result.get("total_subtopics", 0),
                    "timestamp": datetime.now().isoformat(),
                }, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.warning(f"Failed to save planning data: {exc}")

        for sub in decompose_result.get("sub_topics", []):
            title = (sub.get("title") or "").strip()
            if not title:
                continue
            try:
                block = self.queue.add_block(sub_topic=title, overview=sub.get("overview", ""))
                self._log_progress("planning", "queue_seeded", block_id=block.block_id, sub_topic=block.sub_topic)
            except RuntimeError as err:
                self.logger.warning(f"Queue capacity reached: {err}")
                break

        self.agents["manager"].set_primary_topic(optimized_topic)
        self._log_progress("planning", "planning_completed", total_blocks=self.queue.get_statistics()["total_blocks"])
        return optimized_topic

    # ------------------------------------------------------------------
    # Phase 2: Researching
    # ------------------------------------------------------------------

    async def _phase2_researching(self) -> None:
        execution_mode = self.config.get("researching", {}).get("execution_mode", "series")
        if execution_mode == "parallel":
            await self._phase2_researching_parallel()
        else:
            await self._phase2_researching_series()

    async def _phase2_researching_series(self) -> None:
        self._stage_events.setdefault("researching", [])
        manager = self.agents["manager"]
        research = self.agents["research"]
        total_blocks = len(self.queue.blocks)
        completed_blocks = 0

        self._log_researching_progress("researching_started", total_blocks=total_blocks, execution_mode="series")

        while not manager.is_research_complete():
            block = manager.get_next_task()
            if not block:
                break

            self._log_researching_progress("block_started", block_id=block.block_id, sub_topic=block.sub_topic,
                                           current_block=completed_blocks + 1, total_blocks=total_blocks)

            iteration_callback = self._create_iteration_progress_callback(
                block_id=block.block_id, sub_topic=block.sub_topic, execution_mode="series",
                current_block=completed_blocks + 1, total_blocks=total_blocks,
            )
            result = await research.process(
                topic_block=block, call_tool_callback=self._call_tool,
                note_agent=self.agents["note"], citation_manager=self.citation_manager,
                queue=self.queue, manager_agent=manager, config=self.config,
                progress_callback=iteration_callback,
            )

            manager.complete_task(block.block_id)
            completed_blocks += 1
            total_blocks = len(self.queue.blocks)

            self._log_researching_progress("block_completed", block_id=block.block_id, sub_topic=block.sub_topic,
                                           iterations=result.get("iterations", 0), tools_used=result.get("tools_used", []),
                                           current_block=completed_blocks, total_blocks=total_blocks)
            manager.get_queue_status()

        stats = self.queue.get_statistics()
        self._log_researching_progress("researching_completed", completed_blocks=stats["completed"],
                                       total_tool_calls=stats["total_tool_calls"])

    async def _phase2_researching_parallel(self) -> None:
        self._stage_events.setdefault("researching", [])
        manager = self.agents["manager"]
        research = self.agents["research"]
        max_parallel = self.config.get("researching", {}).get("max_parallel_topics", 5)
        semaphore = asyncio.Semaphore(max_parallel)
        pending_blocks = [b for b in self.queue.blocks if b.status == TopicStatus.PENDING]
        total_blocks = len(self.queue.blocks)

        self._log_researching_progress("researching_started", total_blocks=total_blocks,
                                       execution_mode="parallel", max_parallel=max_parallel)

        completed_count = {"value": 0}
        active_tasks: dict[str, dict[str, Any]] = {}
        active_tasks_lock = asyncio.Lock()

        # Async wrappers for thread-safe parallel access
        class AsyncCitationManagerWrapper:
            def __init__(self, cm: Any) -> None:
                self._cm = cm

            async def add_citation(self, citation_id: str, tool_type: str, tool_trace: Any, raw_answer: str) -> bool:
                return await self._cm.add_citation_async(citation_id, tool_type, tool_trace, raw_answer)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._cm, name)

        class AsyncManagerAgentWrapper:
            def __init__(self, ma: Any) -> None:
                self._ma = ma

            async def add_new_topic(self, sub_topic: str, overview: str) -> Any:
                return await self._ma.add_new_topic_async(sub_topic, overview)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._ma, name)

        async_citation_manager = AsyncCitationManagerWrapper(self.citation_manager)
        async_manager_agent = AsyncManagerAgentWrapper(manager)

        async def update_active_task(block_id: str, info: dict[str, Any] | None) -> None:
            async with active_tasks_lock:
                if info is None:
                    active_tasks.pop(block_id, None)
                else:
                    active_tasks[block_id] = info
                self._log_researching_progress("parallel_status_update",
                                               active_tasks=list(active_tasks.values()),
                                               active_count=len(active_tasks),
                                               completed_count=completed_count["value"],
                                               total_blocks=total_blocks)

        async def research_single_block(block: Any) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    async with manager._lock:
                        current_block = self.queue.get_block_by_id(block.block_id)
                        if current_block and current_block.status == TopicStatus.PENDING:
                            self.queue.mark_researching(block.block_id)

                    await update_active_task(block.block_id, {"block_id": block.block_id, "sub_topic": block.sub_topic, "status": "starting"})
                    self._log_researching_progress("block_started", block_id=block.block_id, sub_topic=block.sub_topic, execution_mode="parallel")

                    config_max_iterations = self.config.get("researching", {}).get("max_iterations", 5)

                    def parallel_callback(event_type: str, **data: Any) -> None:
                        task_info = {"block_id": block.block_id, "sub_topic": block.sub_topic,
                                     "status": event_type, "iteration": data.get("iteration", 0),
                                     "max_iterations": data.get("max_iterations", config_max_iterations),
                                     "current_tool": data.get("tool_type"), "current_query": data.get("query")}
                        asyncio.create_task(update_active_task(block.block_id, task_info))
                        self._log_researching_progress(event_type, block_id=block.block_id, execution_mode="parallel", **data)

                    result = await research.process(
                        topic_block=block, call_tool_callback=self._call_tool,
                        note_agent=self.agents["note"], citation_manager=async_citation_manager,
                        queue=self.queue, manager_agent=async_manager_agent, config=self.config,
                        progress_callback=parallel_callback,
                    )
                    await manager.complete_task_async(block.block_id)
                    completed_count["value"] += 1
                    await update_active_task(block.block_id, None)
                    self._log_researching_progress("block_completed", block_id=block.block_id, sub_topic=block.sub_topic,
                                                   iterations=result.get("iterations", 0), execution_mode="parallel")
                    return result
                except Exception as exc:
                    await manager.fail_task_async(block.block_id, str(exc))
                    completed_count["value"] += 1
                    await update_active_task(block.block_id, None)
                    self._log_researching_progress("block_failed", block_id=block.block_id, error=str(exc), execution_mode="parallel")
                    return None

        tasks = [research_single_block(b) for b in pending_blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                await manager.fail_task_async(pending_blocks[i].block_id, str(result))

        max_wait = 100
        wait_count = 0
        while True:
            stats = self.queue.get_statistics()
            if stats.get("pending", 0) == 0 and stats.get("researching", 0) == 0:
                break
            new_pending = [b for b in self.queue.blocks if b.status == TopicStatus.PENDING]
            if not new_pending:
                wait_count += 1
                if wait_count > max_wait:
                    break
                await asyncio.sleep(0.1)
                continue
            wait_count = 0
            new_results = await asyncio.gather(*[research_single_block(b) for b in new_pending], return_exceptions=True)
            for i, result in enumerate(new_results):
                if isinstance(result, Exception):
                    await manager.fail_task_async(new_pending[i].block_id, str(result))

        stats = self.queue.get_statistics()
        self._log_researching_progress("researching_completed", completed_blocks=stats["completed"],
                                       total_tool_calls=stats["total_tool_calls"], execution_mode="parallel")

    # ------------------------------------------------------------------
    # Phase 3: Reporting
    # ------------------------------------------------------------------

    async def _phase3_reporting(self, topic: str) -> dict[str, Any]:
        reporting = self.agents["reporting"]
        reporting.set_citation_manager(self.citation_manager)
        return await reporting.process(self.queue, topic, progress_callback=self._report_progress_callback)

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _log_progress(self, stage: str, status: str, **payload: Any) -> None:
        if stage not in self._stage_events:
            return
        event: dict[str, Any] = {"status": status, "timestamp": datetime.now().isoformat()}
        event.update({k: v for k, v in payload.items() if v is not None})
        self._stage_events[stage].append(event)

        file_path = self.plan_progress_file if stage == "planning" else self.report_progress_file
        context = {
            "research_id": self.research_id, "stage": stage,
            "input_topic": self.input_topic, "optimized_topic": self.optimized_topic,
            "events": self._stage_events[stage],
        }
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(context, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.logger.warning(f"Failed to write progress file: {exc}")

        if self.progress_callback:
            try:
                self.progress_callback({
                    "type": "progress", "stage": stage, "status": status,
                    "research_id": self.research_id,
                    **{k: v for k, v in payload.items() if v is not None},
                })
            except Exception as exc:
                self.logger.warning(f"Progress callback failed: {exc}")

    def _log_researching_progress(self, status: str, **payload: Any) -> None:
        event: dict[str, Any] = {"status": status, "timestamp": datetime.now().isoformat()}
        event.update({k: v for k, v in payload.items() if v is not None})
        with self._progress_file_lock:
            self._stage_events.setdefault("researching", []).append(event)
            progress_file = self.cache_dir / "researching_progress.json"
            context = {
                "research_id": self.research_id, "stage": "researching",
                "input_topic": self.input_topic, "optimized_topic": self.optimized_topic,
                "events": self._stage_events["researching"],
            }
            try:
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(context, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                self.logger.warning(f"Failed to write researching progress: {exc}")

        if self.progress_callback:
            try:
                self.progress_callback({
                    "type": "progress", "stage": "researching", "status": status,
                    "research_id": self.research_id,
                    **{k: v for k, v in payload.items() if v is not None},
                })
            except Exception as exc:
                self.logger.warning(f"Progress callback failed: {exc}")

    def _create_iteration_progress_callback(
        self,
        block_id: str,
        sub_topic: str,
        execution_mode: str,
        current_block: int | None = None,
        total_blocks: int | None = None,
    ) -> Callable[..., None]:
        def callback(event_type: str, **data: Any) -> None:
            payload: dict[str, Any] = {"block_id": block_id, "sub_topic": sub_topic, "execution_mode": execution_mode}
            if current_block is not None:
                payload["current_block"] = current_block
            if total_blocks is not None:
                payload["total_blocks"] = total_blocks
            payload.update(data)
            self._log_researching_progress(event_type, **payload)
        return callback

    def _report_progress_callback(self, event: dict[str, Any]) -> None:
        status = event.pop("status", "unknown")
        self._log_progress("reporting", status, **event)


__all__ = ["ResearchPipeline"]
