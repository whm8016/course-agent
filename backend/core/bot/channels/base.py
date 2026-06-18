"""Base channel interface for chat platforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

from core.bot.bus.events import InboundMessage, OutboundMessage
from core.bot.bus.queue import MessageBus

logger = logging.getLogger(__name__)


class BaseChannel(ABC):
    """
    Abstract base class for chat channel implementations.

    Each channel (QQ, Feishu, etc.) should implement this interface
    to integrate with the TutorBot message bus.
    """

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the channel and begin listening for messages."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel and clean up resources."""
        pass

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through this channel."""
        pass

    def is_allowed(self, sender_id: str) -> bool:
        """Check if sender_id is permitted. Empty list = deny all; '*' = allow all."""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("%s: allow_from is empty — all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """Handle an incoming message from the chat platform."""
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender %s on channel %s.",
                sender_id,
                self.name,
            )
            return

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=metadata or {},
            session_key_override=session_key,
        )

        await self.bus.publish_inbound(msg)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        return self._running
