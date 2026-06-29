"""TutorBot Manager — spawn / stop / manage in-process TutorBot instances.

隔离模型（多租户）：每个 bot 归属一个 owner（user_id）。
- 目录：data/tutorbot/<owner_id>/<bot_id>/（owner_id 为空 → 扁平 legacy 目录）
- 内存 key：uid = f"{owner_id}/{bot_id}"（owner_id 空 → bot_id）
- list/get/start/stop/delete/send_message 全部按 owner 过滤；api 层注入 owner
  并校验归属，杜绝跨用户越权。
"""

from __future__ import annotations

import asyncio
import json
import logging
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
    course_id: str = ""  # 有值时 bot 可调用 rag 工具检索课程知识库
    owner_id: str = ""  # 归属用户 id（多租户隔离）；空 = legacy/系统级


@dataclass
class TutorBotInstance:
    """A running TutorBot and its metadata."""

    bot_id: str
    owner_id: str = ""
    config: BotConfig = None  # type: ignore[assignment]
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
            "owner_id": self.owner_id,
            "name": self.config.name,
            "description": self.config.description,
            "channels": list(self.config.channels.keys()),
            "model": self.config.model,
            "course_id": self.config.course_id,
            "running": self.running,
            "started_at": self.started_at.isoformat(),
        }


def _uid(owner_id: str, bot_id: str) -> str:
    """内存/逻辑唯一键。owner_id 为空（legacy）时退化为 bot_id。"""
    return f"{owner_id}/{bot_id}" if owner_id else bot_id


class TutorBotManager:
    """Manage TutorBot instances running in-process."""

    def __init__(self) -> None:
        self._bots: dict[str, TutorBotInstance] = {}

    @property
    def _tutorbot_dir(self) -> Path:
        return get_bot_workspace_root()

    def _bot_dir(self, owner_id: str, bot_id: str) -> Path:
        # owner_id 非空 → 二级目录；空 → 扁平 legacy 目录（向后兼容）
        if owner_id:
            return self._tutorbot_dir / owner_id / bot_id
        return self._tutorbot_dir / bot_id

    def _bot_workspace(self, owner_id: str, bot_id: str) -> Path:
        return self._bot_dir(owner_id, bot_id) / "workspace"

    def _ensure_bot_dirs(self, owner_id: str, bot_id: str) -> None:
        for sub in ("workspace/sessions", "workspace/memory", "cron", "logs", "media"):
            (self._bot_dir(owner_id, bot_id) / sub).mkdir(parents=True, exist_ok=True)

    # --- Config persistence ---

    def load_bot_config(self, owner_id: str, bot_id: str) -> BotConfig | None:
        path = self._bot_dir(owner_id, bot_id) / "config.yaml"
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
                course_id=data.get("course_id", ""),
                owner_id=data.get("owner_id", owner_id) or owner_id,
            )
        except Exception:
            logger.exception("Failed to load bot config %s", bot_id)
            return None

    def save_bot_config(
        self, owner_id: str, bot_id: str, config: BotConfig, *, auto_start: bool = True
    ) -> None:
        bot_dir = self._bot_dir(owner_id, bot_id)
        bot_dir.mkdir(parents=True, exist_ok=True)
        path = bot_dir / "config.yaml"
        data: dict[str, Any] = {
            "name": config.name,
            "description": config.description,
            "persona": config.persona,
            "channels": config.channels,
            "auto_start": auto_start,
            "owner_id": config.owner_id or owner_id,
        }
        if config.model:
            data["model"] = config.model
        if config.course_id:
            data["course_id"] = config.course_id
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
        tmp_path.replace(path)

    # --- Bot lifecycle ---

    async def start_bot(
        self, owner_id: str, bot_id: str, config: BotConfig | None = None
    ) -> TutorBotInstance:
        uid = _uid(owner_id, bot_id)
        if uid in self._bots and self._bots[uid].running:
            return self._bots[uid]

        self._ensure_bot_dirs(owner_id, bot_id)

        if config is None:
            config = self.load_bot_config(owner_id, bot_id)
        if config is None:
            config = BotConfig(name=bot_id, owner_id=owner_id)
            self.save_bot_config(owner_id, bot_id, config)
        else:
            # 确保归属正确（防止外部传入的 config 缺 owner_id）
            config.owner_id = config.owner_id or owner_id

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
        workspace = self._bot_workspace(owner_id, bot_id)
        session_manager = SessionManager(workspace)

        if config.persona:
            soul_path = workspace / "SOUL.md"
            soul_path.write_text(config.persona, encoding="utf-8")

        canonical_key = f"bot:{bot_id}"

        # AgentLoop 走主编排链路（TRM + Orchestrator），不再需要独立 provider/model
        # —— 复用 core/llm/llm.py（含熔断/Fallback），persona 注入 ChatPipeline。
        agent_loop = AgentLoop(
            bus=bus,
            workspace=workspace,
            course_id=config.course_id,
            persona=config.persona,
            session_manager=session_manager,
            default_session_key=canonical_key,
            owner_id=config.owner_id,
            bot_id=bot_id,
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
            owner_id=config.owner_id,
            config=config,
            agent_loop=agent_loop,
            channel_manager=channel_manager,
        )

        # Core tasks
        loop_task = asyncio.create_task(agent_loop.run(), name=f"tutorbot:{uid}:loop")
        router_task = asyncio.create_task(
            self._outbound_router(owner_id, bot_id, bus, instance), name=f"tutorbot:{uid}:router"
        )
        instance.tasks.extend([loop_task, router_task])

        # Start channel listeners
        if channel_manager:
            for ch_name, ch in channel_manager.channels.items():
                ch_task = asyncio.create_task(ch.start(), name=f"tutorbot:{uid}:ch:{ch_name}")
                instance.tasks.append(ch_task)

        # Heartbeat
        async def _hb_execute(tasks_summary: str) -> str:
            return await agent_loop.process_direct(tasks_summary, session_key=canonical_key, channel="web", chat_id="web")

        async def _hb_notify(response: str) -> None:
            await instance.notify_queue.put(response)

        from config import TUTORBOT_HEARTBEAT_ENABLED, TUTORBOT_HEARTBEAT_INTERVAL_SEC
        heartbeat_enabled = TUTORBOT_HEARTBEAT_ENABLED
        if heartbeat_enabled:
            interval = TUTORBOT_HEARTBEAT_INTERVAL_SEC
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

        self._bots[uid] = instance
        self.save_bot_config(owner_id, bot_id, config)
        logger.info("TutorBot '%s' started (owner=%s workspace=%s)", bot_id, owner_id or "(legacy)", workspace)
        return instance

    async def _outbound_router(
        self, owner_id: str, bot_id: str, bus: MessageBus, instance: TutorBotInstance
    ) -> None:
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

    async def stop_bot(self, owner_id: str, bot_id: str) -> bool:
        uid = _uid(owner_id, bot_id)
        instance = self._bots.get(uid)
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

        self.save_bot_config(owner_id, bot_id, instance.config, auto_start=False)
        del self._bots[uid]
        logger.info("TutorBot '%s' stopped (owner=%s)", bot_id, owner_id or "(legacy)")
        return True

    async def delete_bot(self, owner_id: str, bot_id: str) -> bool:
        """删除 bot：先停（若在跑）再删持久化配置目录。"""
        uid = _uid(owner_id, bot_id)
        if uid in self._bots:
            await self.stop_bot(owner_id, bot_id)
        bot_dir = self._bot_dir(owner_id, bot_id)
        if not bot_dir.exists():
            return False
        import shutil
        shutil.rmtree(bot_dir)
        logger.info("TutorBot '%s' deleted (owner=%s)", bot_id, owner_id or "(legacy)")
        return True

    async def update_bot(
        self,
        owner_id: str,
        bot_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        persona: str | None = None,
        course_id: str | None = None,
    ) -> BotConfig:
        """更新已存在 bot 的配置字段（None = 不改）；若正在运行则重启以应用新配置。"""
        config = self.load_bot_config(owner_id, bot_id)
        if config is None:
            raise KeyError(bot_id)
        if name is not None and name.strip():
            config.name = name.strip()
        if description is not None:
            config.description = description
        if persona is not None:
            config.persona = persona
        if course_id is not None:
            config.course_id = course_id
        uid = _uid(owner_id, bot_id)
        was_running = uid in self._bots and self._bots[uid].running
        self.save_bot_config(owner_id, bot_id, config, auto_start=was_running)
        if was_running:
            await self.stop_bot(owner_id, bot_id)
            await self.start_bot(owner_id, bot_id, config)
        return config

    # --- Listing ---

    def _discover_bot_ids(self, owner_id: str) -> set[str]:
        """扫描某 owner 目录下的 bot_id 集合（二级目录 owner/<bot_id>/）。"""
        ids: set[str] = set()
        if not owner_id:
            return ids
        owner_root = self._tutorbot_dir / owner_id
        if not owner_root.exists():
            return ids
        for entry in owner_root.iterdir():
            if entry.is_dir() and (entry / "config.yaml").exists():
                ids.add(entry.name)
        return ids

    def _discover_legacy_bot_ids(self) -> set[str]:
        """扫描扁平 legacy bot（直接在 data/tutorbot/<bot_id>/，owner_id 为空）。

        判据：一级 entry 是目录且自身含 config.yaml（owner 目录不含 config.yaml）。
        """
        ids: set[str] = set()
        if not self._tutorbot_dir.exists():
            return ids
        for entry in self._tutorbot_dir.iterdir():
            if entry.is_dir() and (entry / "config.yaml").exists():
                ids.add(entry.name)
        return ids

    def _discover_owners(self) -> set[str]:
        """扫描所有 owner 目录（一级 entry 是目录但自身不含 config.yaml）。"""
        owners: set[str] = set()
        if not self._tutorbot_dir.exists():
            return owners
        for entry in self._tutorbot_dir.iterdir():
            if entry.is_dir() and not (entry / "config.yaml").exists():
                owners.add(entry.name)
        return owners

    def list_bots(self, owner_id: str, *, include_legacy: bool = False) -> list[dict[str, Any]]:
        """列出某 owner 的 bot；include_legacy（admin 用）额外含扁平 legacy bot。"""
        result: dict[str, dict[str, Any]] = {}

        # 内存中匹配该 owner 的运行实例
        for uid, inst in self._bots.items():
            if inst.owner_id == owner_id or (include_legacy and not inst.owner_id):
                result[uid] = inst.to_dict()

        # 该 owner 目录下的已配置（未必运行）bot
        for bid in self._discover_bot_ids(owner_id):
            uid = _uid(owner_id, bid)
            if uid in result:
                continue
            cfg = self.load_bot_config(owner_id, bid)
            if cfg is None:
                continue
            result[uid] = {
                "bot_id": bid,
                "owner_id": owner_id,
                "name": cfg.name,
                "description": cfg.description,
                "channels": list(cfg.channels.keys()),
                "model": cfg.model,
                "running": False,
                "started_at": None,
            }

        # legacy bot（仅 admin）
        if include_legacy:
            for bid in self._discover_legacy_bot_ids():
                if bid in result:
                    continue
                cfg = self.load_bot_config("", bid)
                result[bid] = {
                    "bot_id": bid,
                    "owner_id": "",
                    "name": cfg.name if cfg else bid,
                    "description": cfg.description if cfg else "",
                    "channels": list(cfg.channels.keys()) if cfg else [],
                    "model": cfg.model if cfg else None,
                    "running": False,
                    "started_at": None,
                }

        return list(result.values())

    def get_bot(self, owner_id: str, bot_id: str) -> TutorBotInstance | None:
        return self._bots.get(_uid(owner_id, bot_id))

    def all_running_instances(self) -> list[TutorBotInstance]:
        """所有内存中的运行实例（供 notification 等跨 owner 场景定位 IM 发送载体）。"""
        return list(self._bots.values())

    def get_bot_history(self, owner_id: str, bot_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """读取该 bot owner 的 web 对话历史（per-user session，对齐 DeepTutor session 级）。

        web 对话 session_key = bot:{bot_id}:{user_id}（owner 即登录用户）。
        仅返回 web 对话线，不含 QQ/飞书各群会话（渠道隔离，对齐 session≠memory 分层）。
        """
        from core.bot.session.manager import _safe_filename

        sessions_dir = self._bot_workspace(owner_id, bot_id) / "sessions"
        key = f"bot:{bot_id}:{owner_id}"
        path = sessions_dir / f"{_safe_filename(key.replace(':', '_'))}.jsonl"
        if not path.exists():
            return []
        messages: list[dict[str, Any]] = []
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
            return []
        return messages[-limit:]

    async def send_message(
        self, owner_id: str, bot_id: str, content: str, chat_id: str = "web", user_id: str = ""
    ) -> str:
        uid = _uid(owner_id, bot_id)
        instance = self._bots.get(uid)
        if not instance or not instance.running:
            # 多 worker 场景：本 worker 没有该 bot，尝试从磁盘 config 自动启动
            if self._bot_dir(owner_id, bot_id).exists():
                instance = await self.start_bot(owner_id, bot_id)
            else:
                raise RuntimeError(f"Bot '{bot_id}' not found. Create it first via POST /api/bot/create")
        # web 直发：per-user session_key（每个用户独立 web 对话线，避免多用户串；
        # 对齐 DeepTutor session 级历史）+ user_id 让 bot 对话写学生记忆（不再匿名）
        session_key = f"bot:{bot_id}:{user_id}" if user_id else None
        return await instance.agent_loop.process_direct(
            content, channel="web", chat_id=chat_id, user_id=user_id, session_key=session_key
        )

    async def auto_start_bots(self) -> None:
        """Start bots marked with auto_start: true（所有 owner + legacy）。"""
        for bid in self._discover_legacy_bot_ids():
            await self._maybe_auto_start("", bid)
        for owner_id in self._discover_owners():
            for bid in self._discover_bot_ids(owner_id):
                await self._maybe_auto_start(owner_id, bid)

    async def _maybe_auto_start(self, owner_id: str, bot_id: str) -> None:
        uid = _uid(owner_id, bot_id)
        if uid in self._bots and self._bots[uid].running:
            return
        try:
            path = self._bot_dir(owner_id, bot_id) / "config.yaml"
            if not path.exists():
                return
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not data.get("auto_start", False):
                return
            config = BotConfig(
                name=data.get("name", bot_id),
                description=data.get("description", ""),
                persona=data.get("persona", ""),
                channels=data.get("channels", {}),
                model=data.get("model"),
                course_id=data.get("course_id", ""),
                owner_id=data.get("owner_id", owner_id) or owner_id,
            )
            await self.start_bot(owner_id, bot_id, config)
            logger.info("Auto-started bot '%s' (owner=%s)", bot_id, owner_id or "(legacy)")
        except Exception:
            logger.exception("Failed to auto-start bot '%s'", bot_id)

    async def stop_all(self) -> None:
        for uid in list(self._bots.keys()):
            instance = self._bots[uid]
            owner_id = instance.owner_id
            await self.stop_bot(owner_id, instance.bot_id)


_manager: TutorBotManager | None = None


def get_bot_manager() -> TutorBotManager:
    global _manager
    if _manager is None:
        _manager = TutorBotManager()
    return _manager
