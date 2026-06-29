"""
Turn Runtime Manager
====================

管理单回合（turn）的完整生命周期：创建、后台执行、事件订阅、取消。

【在架构中的位置】
  API 层（POST /api/chat、WS /api/run/{cap}）
    → TurnRuntimeManager.start_turn(context)         返回 turn_id
    → TurnRuntimeManager.subscribe_turn(turn_id)     订阅事件流（支持 after_seq 重连）
    → TurnRuntimeManager.cancel_turn(turn_id)        取消（客户端断开时）

  内部执行链：
    _run_turn()
      → ContextBuilder.build(history)                裁剪历史到 token 预算内
      → async for event in orchestrator.handle(ctx): 消费 orchestrator 事件流
          await execution.bus.emit(event)            fan-out 给所有 subscribe_turn 订阅者
      → 收集 ANSWER 内容 → 发布 CAPABILITY_COMPLETE 到全局 EventBus

【与 DeepTutor TurnRuntimeManager 的对应关系】
  start_turn()      ← start_turn()
  subscribe_turn()  ← subscribe_turn(turn_id, after_seq)（in-memory 回放）
  cancel_turn()     ← cancel_turn()
  _TurnExecution    ← _TurnExecution（turn_id / bus / task / seq 事件列表）
  EventBus 发布     ← _publish_live_event + CAPABILITY_COMPLETE

  本版本暂不含：
  - SQLite 事件持久化（after_seq 基于内存事件列表实现）
  - regenerate_last_turn
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from core.context import UnifiedContext
from core.observability import bind_context, log_flow
from core.observability.metrics import observe_turn
from core.stream import StreamEvent, StreamEventType
from core.stream_bus import StreamBus
from services.session.context_builder import ContextBuilder, resolve_budget
from config import TEXT_MODEL

logger = logging.getLogger(__name__)

_EXECUTION_TTL = 300.0  # turn 结束后保留状态的时间（秒），供迟到订阅者回放


@dataclass
class _TurnExecution:
    """单个 turn 的运行时状态。"""

    turn_id: str
    context: UnifiedContext
    bus: StreamBus = field(default_factory=StreamBus)
    task: asyncio.Task | None = None
    # (seq, event) 列表，供 subscribe_turn(after_seq) 做内存回放
    events: list[tuple[int, StreamEvent]] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


class TurnRuntimeManager:
    """管理 turn 生命周期：创建 → ContextBuilder → orchestrator → fan-out → EventBus。"""

    def __init__(self) -> None:
        self._executions: dict[str, _TurnExecution] = {}
        self._lock = asyncio.Lock()
        # ask_user 暂停/恢复：turn_id → asyncio.Queue，loop 挂起时 await queue.get()
        self._reply_queues: dict[str, asyncio.Queue] = {}

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    async def start_turn(self, context: UnifiedContext) -> str:
        """启动一个新 turn，返回 turn_id。

        后台任务：调用 ContextBuilder 裁剪历史 → 驱动 orchestrator 生成事件流。
        """
        turn_id = str(uuid.uuid4())
        if not context.session_id:
            context.session_id = turn_id
        context.metadata["turn_id"] = turn_id  # orchestrator 用于注册全局 bus

        # ContextBuilder：裁剪历史到 token 预算
        builder = ContextBuilder(max_history_tokens=resolve_budget(TEXT_MODEL))
        context.conversation_history = builder.build(context.conversation_history)

        # 创建 reply queue 并注入 waiter 到 context.metadata
        # loop 调用 ask_user 工具时会 await context.metadata["wait_for_user_reply"]()
        reply_queue: asyncio.Queue = asyncio.Queue()
        self._reply_queues[turn_id] = reply_queue

        async def _wait_for_user_reply() -> dict | None:
            return await reply_queue.get()

        context.metadata["wait_for_user_reply"] = _wait_for_user_reply

        # bind_context BEFORE create_task：asyncio 会把当前 context 复制给子任务
        bind_context(
            turn_id=turn_id,
            user_id=str(context.user_id or ""),
            course_id=str(context.course_id or ""),
            mode=str(context.mode or ""),
        )
        log_flow("turn.start", turn_id=turn_id, mode=context.mode,
                 user_id=context.user_id, course_id=context.course_id)

        execution = _TurnExecution(turn_id=turn_id, context=context)

        async with self._lock:
            self._executions[turn_id] = execution

        execution.task = asyncio.create_task(
            self._run_turn(execution),
            name=f"turn-{turn_id[:8]}",
        )
        return turn_id

    async def subscribe_turn(
        self,
        turn_id: str,
        after_seq: int = 0,
    ) -> AsyncIterator[StreamEvent]:
        """订阅指定 turn 的事件流，支持断线后按 seq 回放。

        Args:
            turn_id:   由 start_turn() 返回的 ID。
            after_seq: 从第几个事件开始（0 = 全部）。已完成的事件从内存列表回放，
                       新事件从 fan-out bus 实时接收。

        Raises:
            KeyError: turn_id 不存在。
        """
        execution = self._executions.get(turn_id)
        if execution is None:
            raise KeyError(f"未知 turn_id: {turn_id}")

        seq = 0
        async for event in execution.bus.subscribe():
            if seq >= after_seq:
                yield event
            seq += 1

    async def cancel_turn(self, turn_id: str) -> None:
        """取消正在运行的 turn（客户端断开时调用）。"""
        execution = self._executions.get(turn_id)
        if execution is None:
            return
        if execution.task and not execution.task.done():
            execution.task.cancel()
            log_flow("turn.cancel", turn_id=turn_id)

    async def submit_user_reply(
        self,
        turn_id: str,
        text: str | None = None,
        answers: list[dict] | None = None,
    ) -> bool:
        """向正在等待 ask_user 回复的 turn 投递用户回答。

        Returns:
            True  — 成功投递（turn 存在且处于 pause 状态）
            False — turn 不存在或已结束
        """
        queue = self._reply_queues.get(turn_id)
        if queue is None:
            logger.warning("TurnRuntime: submit_user_reply turn_id=%s 不存在或已结束", turn_id)
            return False
        payload: dict = {"text": text or "", "answers": answers}
        await queue.put(payload)
        logger.info("TurnRuntime: submitted user reply for turn_id=%s", turn_id)
        return True

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    async def _run_turn(self, execution: _TurnExecution) -> None:
        """后台任务：orchestrator → fan-out → EventBus。"""
        from core.orchestrator import get_orchestrator
        from events.event_bus import CapabilityCompleteEvent, get_event_bus
        from core.observability.langsmith_trace import trace_context

        orchestrator = get_orchestrator()
        answer_parts: list[str] = []
        seq = 0
        first_event_logged = False
        t0 = time.monotonic()

        # 顶层 turn trace：下游所有 @traceable 工具/RAG 与 wrap_openai 的 LLM run
        # 通过 langsmith contextvars run tree 自动成为本 turn 的子 run。
        async with trace_context(
            name="turn",
            metadata={
                "turn_id": execution.turn_id,
                "user_id": str(execution.context.user_id or ""),
                "course_id": str(execution.context.course_id or ""),
                "mode": str(execution.context.mode or ""),
            },
            tags=[f"mode:{execution.context.mode or 'chat'}"],
        ):
            try:
                async for event in orchestrator.handle(execution.context):
                    # 记录 seq（供断线重连回放）
                    execution.events.append((seq, event))
                    seq += 1

                    if not first_event_logged:
                        log_flow("turn.first_event", turn_id=execution.turn_id,
                                 elapsed_ms=int((time.monotonic() - t0) * 1000),
                                 event_type=event.type.value if hasattr(event.type, "value") else str(event.type))
                        first_event_logged = True

                    # 收集最终答案文本
                    if event.type == StreamEventType.ANSWER:
                        answer_parts.append(str(event.payload.get("content") or ""))

                    # fan-out 给所有 subscribe_turn 订阅者
                    await execution.bus.emit(event)

            except asyncio.CancelledError:
                log_flow("turn.cancel", turn_id=execution.turn_id,
                         elapsed_ms=int((time.monotonic() - t0) * 1000), events=seq)
                await execution.bus.emit({
                    "type": "error",
                    "message": "turn 被取消",
                    "source": "turn_runtime",
                })
            except Exception as exc:
                logger.exception(
                    "TurnRuntime: turn_id=%s unhandled error: %s",
                    execution.turn_id, exc,
                )
                log_flow("turn.error", turn_id=execution.turn_id,
                         elapsed_ms=int((time.monotonic() - t0) * 1000), error=str(exc))
                await execution.bus.emit({
                    "type": "error",
                    "message": str(exc),
                    "source": "turn_runtime",
                })
            finally:
                execution.finished_at = time.monotonic()
                elapsed_total = int((execution.finished_at - t0) * 1000)
                agent_output = "".join(answer_parts)
                status = "ok" if agent_output else "empty"
                log_flow("turn.complete", turn_id=execution.turn_id,
                         elapsed_ms=elapsed_total, events=seq,
                         answer_chars=len(agent_output))
                observe_turn(
                    mode=execution.context.mode or "chat",
                    status=status,
                    elapsed_ms=elapsed_total,
                )

                # 清理 reply queue：若 loop 仍在 await waiter()，put(None) 让它正常返回
                q = self._reply_queues.pop(execution.turn_id, None)
                if q is not None:
                    try:
                        q.put_nowait(None)
                    except Exception:
                        pass
                if not execution.bus._closed:
                    await execution.bus.close()

                # 发布 CAPABILITY_COMPLETE 事件（记忆更新等后台任务订阅）
                if agent_output:
                    try:
                        await get_event_bus().publish(CapabilityCompleteEvent(
                            turn_id=execution.turn_id,
                            user_id=execution.context.user_id,
                            course_id=execution.context.course_id,
                            mode=execution.context.mode or "chat",
                            user_message=execution.context.user_message,
                            agent_output=agent_output,
                        ))
                    except Exception:
                        logger.warning("TurnRuntime: EventBus publish failed", exc_info=True)

                # TTL 后清理内存
                asyncio.create_task(
                    self._schedule_cleanup(execution.turn_id),
                    name=f"cleanup-{execution.turn_id[:8]}",
                )

    async def _schedule_cleanup(self, turn_id: str) -> None:
        """在 TTL 后从内存中移除 turn 状态。"""
        await asyncio.sleep(_EXECUTION_TTL)
        async with self._lock:
            self._executions.pop(turn_id, None)
        logger.debug("TurnRuntime: cleaned up turn_id=%s", turn_id)


# ---- 全局单例 ----

_manager: TurnRuntimeManager | None = None


def get_turn_runtime_manager() -> TurnRuntimeManager:
    """返回全局 TurnRuntimeManager 单例（懒加载）。"""
    global _manager
    if _manager is None:
        _manager = TurnRuntimeManager()
    return _manager
