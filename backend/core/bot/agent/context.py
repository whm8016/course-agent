"""Context builder for assembling agent prompts."""

import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_RESULT_MAX_CHARS = 16_000


def _build_assistant_message(
    content: str | None,
    tool_calls: list[dict] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    if content:
        msg["content"] = content
    if tool_calls:
        msg["tool_calls"] = tool_calls
    for k, v in kwargs.items():
        if v is not None:
            msg[k] = v
    return msg


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]

    def __init__(self, workspace: Path, *, shared_memory_dir: Path | None = None):
        self.workspace = workspace
        self.shared_memory_dir = shared_memory_dir

    def build_system_prompt(self) -> str:
        parts = [self._get_identity()]
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        memory = self._load_memory()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")

        return f"""# TutorBot

You are TutorBot, a personal learning companion and AI assistant.

## Environment
- Runtime: {runtime}
- Workspace: {workspace_path}
- Current time: {now}

## Guidelines
- Be helpful, clear, and patient
- Use available tools to accomplish tasks
- Always respond in the user's language
"""

    def _load_bootstrap_files(self) -> str:
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(content)
                except Exception:
                    pass
        return "\n\n---\n\n".join(parts) if parts else ""

    def _load_memory(self) -> str:
        memory_file = self.workspace / "memory" / "MEMORY.md"
        if memory_file.exists():
            try:
                return memory_file.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        if self.shared_memory_dir:
            shared_memory = self.shared_memory_dir / "MEMORY.md"
            if shared_memory.exists():
                try:
                    return shared_memory.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return ""

    def build_messages(
        self,
        session_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        system_prompt = self.build_system_prompt()
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(session_history)
        return messages

    @staticmethod
    def add_assistant_message(
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        msg = _build_assistant_message(content, tool_calls, **kwargs)
        return messages + [msg]

    @staticmethod
    def add_tool_result(
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        if len(result) > _TOOL_RESULT_MAX_CHARS:
            result = result[:_TOOL_RESULT_MAX_CHARS] + "\n...[truncated]"
        return messages + [{
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }]
