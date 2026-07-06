"""Agent loop — 统一走主编排链路（CourseOrchestrator + TurnRuntimeManager）。

【改造说明（Partner 架构）】
旧版 bot 维护一套独立薄壳：自己造 system_prompt + 静默 StreamBus + 直调 run_agent_loop，
绕过了主链路（Orchestrator → ChatCapability → ChatPipeline），因此丢失了
课程 prompt、DB 记忆更新、LLM 熔断/Fallback。

新版 bot 与 Web（/api/chat、/api/run）完全共享同一 Agent 引擎：
  InboundMessage
    → _resolve_user_id()        sender_id ──UserSocialBinding──► DB user_id
    → _build_memory_context()   user_id ──get_user_by_id──► build_memory_context
    → 构造 UnifiedContext（与 Web 一致）
    → TurnRuntimeManager.start_turn(ctx)
    → subscribe_turn(turn_id)   消费 ANSWER 事件，拼最终文本

由此 bot 自动获得：
  - 课程 prompt（get_course_prompt）+ bot persona 注入（metadata["bot_persona"]）
  - DB 记忆更新（TRM 自动发布 CAPABILITY_COMPLETE → v3 memory / graph_memory）
  - LLM 熔断 + 指数退避 + 多供应商 Fallback（core/llm/llm.py）

约束：
  - IM bot 不支持 ask_user 暂停（无双向交互），故 enabled_tools 不含 ask_user
  - course_id 有值 → rag + web_search；无值 → 仅 web_search
  - 会话历史仍用 JSONL（SessionManager），经 ContextBuilder token 裁剪
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from core.bot.binding import parse_bind_command
from core.bot.bus.events import InboundMessage, OutboundMessage
from core.bot.bus.queue import MessageBus
from core.bot.session.manager import SessionManager
from core.context import UnifiedContext
from core.stream import StreamEventType

logger = logging.getLogger(__name__)


class AgentLoop:
    """Bot agent loop — 走主编排链路（TRM + Orchestrator），与 Web 共享 Agent 引擎。"""

    def __init__(
        self,
        bus: MessageBus,
        workspace: Path,
        course_id: str = "",
        persona: str = "",
        max_iterations: int = 10,
        session_manager: SessionManager | None = None,
        default_session_key: str | None = None,
        owner_id: str = "",
        bot_id: str = "",
        **_kwargs: Any,  # 兼容旧调用（provider/model/context_window_tokens 等）
    ):
        self.bus = bus
        self.workspace = workspace
        self.course_id = course_id
        self.persona = persona
        self.max_iterations = max_iterations
        self._default_session_key = default_session_key
        self.owner_id = owner_id
        self.bot_id = bot_id

        self.sessions = session_manager or SessionManager(workspace)

        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    @property
    def _enabled_tools(self) -> list[str]:
        """有 course_id → rag + web_search + cron；无 → web_search + cron（ask_user 不向 IM 开放）。"""
        if self.course_id:
            return ["rag", "web_search", "cron"]
        return ["web_search", "cron"]

    # ------------------------------------------------------------------ #
    #  Main bus loop                                                       #
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks."""
        self._running = True
        logger.info("Agent loop started (unified orchestrator, course_id=%r)", self.course_id)

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: %s", e)
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)

                def _cleanup(done_task: asyncio.Task[None], sk: str = msg.session_key) -> None:
                    tasks = self._active_tasks.get(sk, [])
                    if done_task in tasks:
                        tasks.remove(done_task)
                task.add_done_callback(_cleanup)

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
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        async with lock:
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
                    OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="抱歉，处理消息时出现错误，请稍后重试。",
                    )
                )
        # 当前 session 无等待中任务 且 锁未被占用 → 清理，防字典无限增长
        pending = [t for t in self._active_tasks.get(msg.session_key, []) if not t.done()]
        if not pending and not lock.locked():
            self._session_locks.pop(msg.session_key, None)

    # ------------------------------------------------------------------ #
    #  Core turn logic (unified with Web via TurnRuntimeManager)           #
    # ------------------------------------------------------------------ #

    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        # 绑定指令拦截（不进对话历史、不消耗 LLM）：「绑定 <码>」→ 绑定 IM 账号到 web User
        bind_code = parse_bind_command(msg.content)
        if bind_code is not None:
            reply = await self._handle_bind(bind_code, msg.channel, msg.sender_id)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=reply, metadata={}
            )
        session_key = msg.session_key or self._default_session_key or f"{msg.channel}:{msg.chat_id}"
        final_text = await self._run_turn(
            msg.content,
            session_key=session_key,
            channel=msg.channel,
            sender_id=msg.sender_id,
            chat_id=msg.chat_id,
        )
        if not final_text:
            return None
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_text,
            metadata={"message_id": msg.metadata.get("message_id")},
        )

    async def _handle_bind(self, code: str, channel: str, sender_id: str) -> str:
        """处理「绑定 <码>」：校验码 → 写 UserSocialBinding（IM openid ↔ web user_id）。

        绑定后该 IM openid 的消息经 _resolve_user_id 命中同一 user_id，长期记忆跨渠道。
        """
        from core.bot.binding import consume_bind_code

        if not channel or not sender_id:
            return "绑定失败：无法识别你的 IM 账号。"
        user_id = consume_bind_code(code)
        if not user_id:
            return "绑定失败：绑定码无效或已过期，请在网页端重新生成。"
        try:
            import time as _time

            from sqlalchemy import select

            from core.db.database import AsyncSessionLocal, UserSocialBinding

            platform = channel.strip().lower()
            async with AsyncSessionLocal() as db:
                existing = await db.execute(
                    select(UserSocialBinding).where(
                        UserSocialBinding.platform == platform,
                        UserSocialBinding.platform_user_id == sender_id,
                    )
                )
                if existing.scalar_one_or_none():
                    return "该 IM 账号已被绑定，无法重复绑定。"
                db.add(UserSocialBinding(
                    user_id=user_id,
                    platform=platform,
                    platform_user_id=sender_id,
                    created_at=_time.time(),
                ))
                await db.commit()
            return "✅ 绑定成功！你的 IM 账号已关联网站账号，之后我会记得你跨渠道的学习记录。"
        except Exception:
            logger.exception("handle_bind failed code=%s channel=%s", code, channel)
            return "绑定失败：服务异常，请稍后重试。"

    async def _resolve_user_id(self, channel: str, sender_id: str) -> str:
        """sender_id ──UserSocialBinding──► DB user_id。未绑定/匿名返回 ''。"""
        if not channel or not sender_id:
            return ""
        try:
            from sqlalchemy import select

            from core.db.database import AsyncSessionLocal, UserSocialBinding

            platform = channel.strip().lower()
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(UserSocialBinding.user_id).where(
                        UserSocialBinding.platform == platform,
                        UserSocialBinding.platform_user_id == sender_id,
                    )
                )
                row = result.first()
                return str(row[0]) if row else ""
        except Exception:
            logger.exception("resolve user_id failed channel=%s sender=%s", channel, sender_id)
            return ""

    async def _build_memory_context(self, user_id: str, query_text: str = "") -> str:
        """读 Mem0 记忆：用 query_text 语义检索相关记忆注入。匿名返回 ''。"""
        if not user_id:
            return ""
        try:
            from core.memory.mem0_client import build_memory_context
            return await build_memory_context(user_id, query_text)
        except Exception:
            logger.exception("build_memory_context failed user_id=%s", user_id)
            return ""

    async def _run_turn(
        self,
        user_message: str,
        session_key: str,
        channel: str = "",
        sender_id: str = "",
        user_id: str = "",
        chat_id: str = "",
    ) -> str:
        """执行一个用户回合：走主编排链路（与 Web 共享同一 Agent 引擎）。

        user_id 优先（web 直发已由 API 层从登录态解析）；为空时回退到
        sender_id → UserSocialBinding 绑定表解析（IM 渠道）。
        """
        from services.session.turn_runtime import get_turn_runtime_manager

        # 1. 会话历史（JSONL，排除刚追加的当前 user 消息）
        session = self.sessions.get_or_create(session_key)
        session.add_message("user", user_message)
        conversation_history = session.get_history()[:-1]

        # 2. 打通记忆：优先用传入的 user_id（web），否则 sender_id → 绑定表 → user_id
        resolved_user_id = user_id or await self._resolve_user_id(channel, sender_id)
        memory_context = await self._build_memory_context(resolved_user_id, user_message)

        # 3. 构造 UnifiedContext（与 Web 主链路完全一致）
        ctx = UnifiedContext(
            user_message=user_message,
            conversation_history=conversation_history,
            course_id=self.course_id,
            user_id=resolved_user_id,
            mode="chat",
            enabled_tools=self._enabled_tools,  # 不含 ask_user
            memory_context=memory_context,
        )
        if self.persona:
            ctx.metadata["bot_persona"] = self.persona
        # cron owner：让 bot 对话里 agent 能调 cron 工具设定时（到点发回本会话 channel）
        if self.bot_id:
            ctx.metadata["cron_owner"] = {
                "partner_id": f"{self.owner_id}:{self.bot_id}",
                "channel": channel,
                "chat_id": chat_id,
                "session_key": session_key,
                "user_id": resolved_user_id,
            }

        # 4. 走 TRM：start_turn → subscribe_turn，与 Web 同一引擎
        #    （TRM 自动驱动 Orchestrator → ChatCapability → ChatPipeline → run_agent_loop，
        #     turn 结束自动发布 CAPABILITY_COMPLETE → 异步更新 v3 memory / graph_memory）
        trm = get_turn_runtime_manager()
        turn_id = await trm.start_turn(ctx)

        final_text = ""
        try:
            async for event in trm.subscribe_turn(turn_id):
                if event.type == StreamEventType.ANSWER:
                    final_text += str(event.payload.get("content") or "")
        except asyncio.CancelledError:
            await trm.cancel_turn(turn_id)
            raise
        # 正常结束：bus 由 TRM._run_turn 的 finally 关闭，subscribe_turn 自然退出；
        # turn 状态由 TRM TTL 自动清理，无需手动 cancel。

        if final_text:
            session.add_message("assistant", final_text)
        self.sessions.save(session)

        logger.info(
            "Turn complete: session=%s user=%s chars=%d",
            session_key, resolved_user_id or "(anon)", len(final_text),
        )
        return final_text

    # ------------------------------------------------------------------ #
    #  Public REST / cron interface                                        #
    # ------------------------------------------------------------------ #

    async def process_direct(
        self,
        content: str,
        session_key: str | None = None,
        channel: str = "web",
        chat_id: str = "web",
        user_id: str = "",
        on_progress: Any = None,
        **_kwargs: Any,
    ) -> str:
        """Process a message directly (for REST API / cron calls).

        user_id 由 API 层从登录态注入（web 直发），使 bot 对话也写学生记忆；
        IM/cron 调用未传时按匿名处理（user_id=''，记忆更新跳过），仍享安全护栏 /
        课程 RAG / LLM 熔断。
        """
        key = session_key or self._default_session_key or f"{channel}:{chat_id}"
        return await self._run_turn(
            content, session_key=key, channel=channel, sender_id="", user_id=user_id, chat_id=chat_id
        )

    async def stop(self) -> None:
        self._running = False
        logger.info("Agent loop stopped")
