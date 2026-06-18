"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from core.bot.bus.events import InboundMessage, OutboundMessage
from core.bot.bus.queue import MessageBus
from core.bot.providers.base import LLMProvider
from core.bot.session.manager import Session, SessionManager

from .context import ContextBuilder
from .tools.message import MessageTool
from .tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        session_manager: SessionManager | None = None,
        shared_memory_dir: Path | None = None,
        default_session_key: str | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self._default_session_key = default_session_key

        self.context = ContextBuilder(workspace, shared_memory_dir=shared_memory_dir)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()

        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._processing_lock = asyncio.Lock()

        self._register_default_tools()

    def update_llm(self, *, provider: LLMProvider, model: str | None = None, context_window_tokens: int | None = None) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()
        if context_window_tokens:
            self.context_window_tokens = context_window_tokens

    def _register_default_tools(self) -> None:
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        tool = self.tools.get("message")
        if tool and hasattr(tool, "set_context"):
            tool.set_context(channel, chat_id, message_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1
            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls))

                tool_call_dicts = [tc.to_openai_tool_call() for tc in response.tool_calls]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: %s(%s)", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                if response.finish_reason == "error":
                    logger.error("LLM returned error: %s", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean,
                    reasoning_content=response.reasoning_content,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations (%d) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks."""
        self._running = True
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: %s", e)
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)

                def _cleanup_task(done_task: asyncio.Task[None], session_key: str = msg.session_key) -> None:
                    session_tasks = self._active_tasks.get(session_key, [])
                    if done_task in session_tasks:
                        session_tasks.remove(done_task)
                task.add_done_callback(_cleanup_task)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        content = f"Stopped {cancelled} task(s)." if cancelled else "No active task to stop."
        await self.bus.publish_outbound(
            OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=content)
        )

    async def _dispatch(self, msg: InboundMessage) -> None:
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
            except asyncio.CancelledError:
                logger.info("Task cancelled for session %s", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session %s", msg.session_key)
                await self.bus.publish_outbound(
                    OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="Sorry, I encountered an error.")
                )

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))

        session_key = msg.session_key or self._default_session_key or f"{msg.channel}:{msg.chat_id}"
        session = self.sessions.get_or_create(session_key)
        session.add_message("user", msg.content)

        history = session.get_history()
        messages = self.context.build_messages(history)

        async def _progress(text: str, **kw: Any) -> None:
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=text,
                    metadata={"_progress": True},
                )
            )

        final_content, tools_used, final_messages = await self._run_agent_loop(messages, _progress)

        if final_content:
            session.add_message("assistant", final_content)
        self.sessions.save(session)

        if final_content:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content,
                metadata={"message_id": msg.metadata.get("message_id")},
            )
        return None

    async def process_direct(
        self,
        content: str,
        session_key: str | None = None,
        channel: str = "web",
        chat_id: str = "web",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for web API calls)."""
        self._set_tool_context(channel, chat_id)

        key = session_key or self._default_session_key or f"{channel}:{chat_id}"
        session = self.sessions.get_or_create(key)
        session.add_message("user", content)

        history = session.get_history()
        messages = self.context.build_messages(history)

        async def _prog(text: str, **kw: Any) -> None:
            if on_progress:
                await on_progress(text)

        final_content, _, _ = await self._run_agent_loop(messages, _prog)

        if final_content:
            session.add_message("assistant", final_content)
        self.sessions.save(session)

        return final_content or ""

    async def stop(self) -> None:
        self._running = False
        logger.info("Agent loop stopped")
