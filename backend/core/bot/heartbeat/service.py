"""Heartbeat service — periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

from core.bot.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Summary of active tasks (required for run)",
                    },
                },
                "required": ["action"],
            },
        },
    }
]


class HeartbeatService:
    """Periodic heartbeat service that wakes the agent to check for tasks.

    Phase 1 (decision): reads HEARTBEAT.md and asks the LLM whether there are active tasks.
    Phase 2 (execution): only triggered when Phase 1 returns 'run'.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.interval_s = interval_s
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Heartbeat service started (interval=%ds)", self.interval_s)

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("Heartbeat service stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if not self._running:
                    break
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat tick error")

    async def _tick(self) -> None:
        heartbeat_file = self.workspace / "HEARTBEAT.md"
        if not heartbeat_file.exists():
            return

        content = heartbeat_file.read_text(encoding="utf-8").strip()
        if not content:
            return

        messages = [
            {"role": "system", "content": "You are a task scheduler. Review the task list and decide if any tasks need to be executed now."},
            {"role": "user", "content": content},
        ]

        response = await self.provider.chat_with_retry(
            messages=messages,
            tools=_HEARTBEAT_TOOL,
            model=self.model,
            tool_choice={"type": "function", "function": {"name": "heartbeat"}},
        )

        if not response.has_tool_calls:
            return

        tc = response.tool_calls[0]
        if tc.arguments.get("action") != "run":
            return

        tasks_summary = tc.arguments.get("tasks", "")
        if not tasks_summary:
            return

        if self.on_execute:
            result = await self.on_execute(tasks_summary)
            if result and self.on_notify:
                await self.on_notify(result)
