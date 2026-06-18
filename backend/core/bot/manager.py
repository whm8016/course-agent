"""TutorBot Manager — spawn / stop / manage in-process TutorBot instances."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.bot.bus.events import OutboundMessage
from core.bot.bus.queue import MessageBus
from core.bot.config.paths import get_bot_workspace_root

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for a single TutorBot instance."""

    name: str
    description: str = ""
    persona: str = ""
    channels: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


@dataclass
class TutorBotInstance:
    """A running TutorBot and its metadata."""

    bot_id: str
    config: BotConfig
    started_at: datetime = field(default_factory=datetime.now)
    tasks: list[asyncio.Task] = field(default_factory=list, repr=False)
    agent_loop: Any = None
    channel_manager: Any = None
    heartbeat: Any = None
    notify_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    @property
    def running(self) -> bool:
        return any(not t.done() for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "name": self.config.name,
            "description": self.config.description,
            "channels": list(self.config.channels.keys()),
            "model": self.config.model,
            "running": self.running,
            "started_at": self.started_at.isoformat(),
        }


class TutorBotManager:
    """Manage TutorBot instances running in-process."""

    def __init__(self) -> None:
        self._bots: dict[str, TutorBotInstance] = {}

    @property
    def _tutorbot_dir(self) -> Path:
        return get_bot_workspace_root()

    def _bot_dir(self, bot_id: str) -> Path:
        return self._tutorbot_dir / bot_id

    def _bot_workspace(self, bot_id: str) -> Path:
        return self._bot_dir(bot_id) / "workspace"

    def _ensure_bot_dirs(self, bot_id: str) -> None:
        for sub in ("workspace/sessions", "workspace/memory", "cron", "logs", "media"):
            (self._bot_dir(bot_id) / sub).mkdir(parents=True, exist_ok=True)

    # --- Config persistence ---

    def load_bot_config(self, bot_id: str) -> BotConfig | None:
        path = self._bot_dir(bot_id) / "config.yaml"
        if not path.exists():
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return BotConfig(
                name=data.get("name", bot_id),
                description=data.get("description", ""),
                persona=data.get("persona", ""),
                channels=data.get("channels", {}),
                model=data.get("model"),
            )
        except Exception:
            logger.exception("Failed to load bot config %s", bot_id)
            return None

    def save_bot_config(self, bot_id: str, config: BotConfig, *, auto_start: bool = True) -> None:
        bot_dir = self._bot_dir(bot_id)
        bot_dir.mkdir(parents=True, exist_ok=True)
        path = bot_dir / "config.yaml"
        data: dict[str, Any] = {
            "name": config.name,
            "description": config.description,
            "persona": config.persona,
            "channels": config.channels,
            "auto_start": auto_start,
        }
        if config.model:
            data["model"] = config.model
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        tmp_path.replace(path)

    # --- Bot lifecycle ---

    async def start_bot(self, bot_id: str, config: BotConfig | None = None) -> TutorBotInstance:
        if bot_id in self._bots and self._bots[bot_id].running:
            return self._bots[bot_id]

        self._ensure_bot_dirs(bot_id)

        if config is None:
            config = self.load_bot_config(bot_id)
        if config is None:
            config = BotConfig(name=bot_id)
            self.save_bot_config(bot_id, config)

        from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, TEXT_MODEL
        from core.bot.agent.loop import AgentLoop
        from core.bot.heartbeat.service import HeartbeatService
        from core.bot.providers.openai_compat import OpenAICompatProvider
        from core.bot.session.manager import SessionManager

        model = config.model or TEXT_MODEL
        provider = OpenAICompatProvider(
            api_key=DASHSCOPE_API_KEY,
            api_base=DASHSCOPE_BASE_URL,
            default_model=model,
        )
        bus = MessageBus()
        workspace = self._bot_workspace(bot_id)
        session_manager = SessionManager(workspace)

        if config.persona:
            soul_path = workspace / "SOUL.md"
            soul_path.write_text(config.persona, encoding="utf-8")

        canonical_key = f"bot:{bot_id}"

        agent_loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            model=model,
            session_manager=session_manager,
            default_session_key=canonical_key,
        )

        # Channel setup
        channel_manager = None
        if config.channels:
            from core.bot.channels.manager import ChannelManager
            try:
                channel_manager = ChannelManager(config.channels, bus)
            except Exception:
                logger.exception("Failed to init channels for bot '%s'", bot_id)

        instance = TutorBotInstance(
            bot_id=bot_id,
            config=config,
            agent_loop=agent_loop,
            channel_manager=channel_manager,
        )

        # Core tasks
        loop_task = asyncio.create_task(agent_loop.run(), name=f"tutorbot:{bot_id}:loop")
        router_task = asyncio.create_task(
            self._outbound_router(bot_id, bus, instance), name=f"tutorbot:{bot_id}:router"
        )
        instance.tasks.extend([loop_task, router_task])

        # Start channel listeners
        if channel_manager:
            for ch_name, ch in channel_manager.channels.items():
                ch_task = asyncio.create_task(ch.start(), name=f"tutorbot:{bot_id}:ch:{ch_name}")
                instance.tasks.append(ch_task)

        # Heartbeat
        async def _hb_execute(tasks_summary: str) -> str:
            return await agent_loop.process_direct(tasks_summary, session_key=canonical_key, channel="web", chat_id="web")

        async def _hb_notify(response: str) -> None:
            await instance.notify_queue.put(response)

        heartbeat_enabled = os.getenv("TUTORBOT_HEARTBEAT_ENABLED", "true").lower() in ("1", "true", "yes")
        if heartbeat_enabled:
            interval = int(os.getenv("TUTORBOT_HEARTBEAT_INTERVAL_SEC", "1800"))
            heartbeat = HeartbeatService(
                workspace=workspace,
                provider=provider,
                model=model,
                on_execute=_hb_execute,
                on_notify=_hb_notify,
                interval_s=interval,
            )
            instance.heartbeat = heartbeat
            await heartbeat.start()

        self._bots[bot_id] = instance
        self.save_bot_config(bot_id, config)
        logger.info("TutorBot '%s' started (workspace=%s)", bot_id, workspace)
        return instance

    async def _outbound_router(self, bot_id: str, bus: MessageBus, instance: TutorBotInstance) -> None:
        """Route outbound messages to channels and web notify queue."""
        try:
            while True:
                msg: OutboundMessage = await bus.consume_outbound()
                is_progress = bool(msg.metadata and msg.metadata.get("_progress"))

                if instance.channel_manager:
                    channel = instance.channel_manager.get_channel(msg.channel)
                    if channel:
                        try:
                            await channel.send(msg)
                        except Exception:
                            logger.exception("Failed to send to channel %s for bot %s", msg.channel, bot_id)

                if not is_progress:
                    await instance.notify_queue.put(msg.content or "")
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Outbound router failed for bot %s", bot_id)

    async def stop_bot(self, bot_id: str) -> bool:
        instance = self._bots.get(bot_id)
        if not instance:
            return False

        for task in instance.tasks:
            if not task.done():
                task.cancel()
        for task in instance.tasks:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if instance.channel_manager:
            try:
                await instance.channel_manager.stop_all()
            except Exception:
                logger.exception("Error stopping channels for bot '%s'", bot_id)

        if instance.heartbeat:
            instance.heartbeat.stop()

        if instance.agent_loop:
            try:
                await instance.agent_loop.stop()
            except Exception:
                pass

        self.save_bot_config(bot_id, instance.config, auto_start=False)
        del self._bots[bot_id]
        logger.info("TutorBot '%s' stopped", bot_id)
        return True

    # --- Listing ---

    def _discover_bot_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self._tutorbot_dir.exists():
            return ids
        for entry in self._tutorbot_dir.iterdir():
            if entry.is_dir() and (entry / "config.yaml").exists():
                ids.add(entry.name)
        return ids

    def list_bots(self) -> list[dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for inst in self._bots.values():
            result[inst.bot_id] = inst.to_dict()
        for bid in self._discover_bot_ids():
            if bid in result:
                continue
            cfg = self.load_bot_config(bid)
            result[bid] = {
                "bot_id": bid,
                "name": cfg.name if cfg else bid,
                "description": cfg.description if cfg else "",
                "channels": list(cfg.channels.keys()) if cfg else [],
                "model": cfg.model if cfg else None,
                "running": False,
                "started_at": None,
            }
        return list(result.values())

    def get_bot(self, bot_id: str) -> TutorBotInstance | None:
        return self._bots.get(bot_id)

    def get_bot_history(self, bot_id: str, limit: int = 100) -> list[dict[str, Any]]:
        sessions_dir = self._bot_workspace(bot_id) / "sessions"
        if not sessions_dir.exists():
            return []
        messages: list[dict[str, Any]] = []
        for path in sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("_type") == "metadata":
                            continue
                        if data.get("role") in ("user", "assistant") and data.get("content"):
                            messages.append(data)
            except Exception:
                continue
        return messages[-limit:]

    async def send_message(self, bot_id: str, content: str, chat_id: str = "web") -> str:
        instance = self._bots.get(bot_id)
        if not instance or not instance.running:
            raise RuntimeError(f"Bot '{bot_id}' is not running")
        return await instance.agent_loop.process_direct(content, channel="web", chat_id=chat_id)

    async def auto_start_bots(self) -> None:
        """Start bots marked with auto_start: true."""
        for bid in self._discover_bot_ids():
            if bid in self._bots and self._bots[bid].running:
                continue
            try:
                path = self._bot_dir(bid) / "config.yaml"
                if not path.exists():
                    continue
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if not data.get("auto_start", False):
                    continue
                config = BotConfig(
                    name=data.get("name", bid),
                    description=data.get("description", ""),
                    persona=data.get("persona", ""),
                    channels=data.get("channels", {}),
                    model=data.get("model"),
                )
                await self.start_bot(bid, config)
                logger.info("Auto-started bot '%s'", bid)
            except Exception:
                logger.exception("Failed to auto-start bot '%s'", bid)

    async def stop_all(self) -> None:
        for bot_id in list(self._bots.keys()):
            await self.stop_bot(bot_id)


_manager: TutorBotManager | None = None


def get_bot_manager() -> TutorBotManager:
    global _manager
    if _manager is None:
        _manager = TutorBotManager()
    return _manager
