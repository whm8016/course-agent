"""Message tool — allows the agent to send messages to channels."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from core.bot.bus.events import OutboundMessage

from .base import Tool


class MessageTool(Tool):
    """Tool for sending messages to chat channels."""

    name = "message"
    description = "Send a message to the current chat channel."
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The message content to send.",
            },
        },
        "required": ["content"],
    }

    def __init__(self, send_callback: Callable[[OutboundMessage], Awaitable[None]]):
        self._send = send_callback
        self._channel: str = "web"
        self._chat_id: str = "web"
        self._message_id: str | None = None

    def set_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._message_id = message_id

    async def execute(self, content: str = "", **kwargs: Any) -> str:
        if not content:
            return "Error: content is required"
        msg = OutboundMessage(
            channel=self._channel,
            chat_id=self._chat_id,
            content=content,
            metadata={"message_id": self._message_id} if self._message_id else {},
        )
        await self._send(msg)
        return "Message sent."
