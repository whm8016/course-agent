"""Channel manager for coordinating chat channels."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.bot.bus.queue import MessageBus

from .base import BaseChannel

logger = logging.getLogger(__name__)


class ChannelManager:
    """
    Manages chat channels and coordinates message routing.

    Responsibilities:
    - Initialize enabled channels
    - Start/stop channels
    - Route outbound messages
    """

    def __init__(self, channels_config: dict[str, Any], bus: MessageBus):
        self.channels_config = channels_config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None

        self._init_channels()

    def _init_channels(self) -> None:
        """Initialize channels from config."""
        from .registry import discover_all

        for name, cls in discover_all().items():
            section = self.channels_config.get(name)
            if section is None:
                continue
            enabled = (
                section.get("enabled", False)
                if isinstance(section, dict)
                else getattr(section, "enabled", False)
            )
            if not enabled:
                continue
            try:
                channel = cls(section, self.bus)
                self.channels[name] = channel
                logger.info("%s channel enabled", cls.display_name)
            except Exception as e:
                logger.warning("%s channel not available: %s", name, e)

    async def start_all(self) -> None:
        """Start all channels and the outbound dispatcher."""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting %s channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel %s: %s", name, e)

    async def stop_all(self) -> None:
        """Stop all channels and the dispatcher."""
        logger.info("Stopping all channels...")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped %s channel", name)
            except Exception as e:
                logger.error("Error stopping %s: %s", name, e)

    async def _dispatch_outbound(self) -> None:
        """Dispatch outbound messages to the appropriate channel."""
        while True:
            try:
                msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)

                if msg.metadata.get("_progress"):
                    continue

                channel = self.channels.get(msg.channel)
                if channel:
                    try:
                        await channel.send(msg)
                    except Exception as e:
                        logger.error("Error sending to %s: %s", msg.channel, e)
                else:
                    logger.warning("Unknown channel: %s", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_channel(self, name: str) -> BaseChannel | None:
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())
