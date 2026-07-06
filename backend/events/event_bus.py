"""
Event Bus
=========

全局异步事件总线，用于跨模块的 turn 完成通知。

【设计目的】
  capability 运行结束后，TurnRuntimeManager 发布 CAPABILITY_COMPLETE 事件；
  记忆更新、session 标题生成等后台任务订阅该事件，避免散落在各 API handler 里。

【典型订阅方（在 main.py lifespan 中注册）】
  get_event_bus().subscribe(EventType.CAPABILITY_COMPLETE, memory_update_handler)

【与 StreamBus 的区别】
  StreamBus — per-turn 事件流（token / tool_call / done 等实时内容）
  EventBus  — 全局 turn 完成通知（记忆、标题、统计等后台任务触发点）
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    CAPABILITY_COMPLETE = "capability_complete"


@dataclass
class CapabilityCompleteEvent:
    """一次 capability 运行完成后携带的上下文信息。"""

    type: EventType = EventType.CAPABILITY_COMPLETE
    turn_id: str = ""
    session_id: str = ""          # L2 摘要压缩 / 记忆更新等后台任务按 session 定位
    user_id: str = ""
    course_id: str = ""
    mode: str = ""
    user_message: str = ""
    agent_output: str = ""          # 最终 answer 内容（从 ANSWER 事件收集）
    metadata: dict[str, Any] = field(default_factory=dict)


# 处理器类型：接收 CapabilityCompleteEvent，返回 coroutine
Handler = Callable[[CapabilityCompleteEvent], Coroutine[Any, Any, None]]


class EventBus:
    """全局发布/订阅总线（单进程内使用）。"""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = {}

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """注册事件处理器。同一类型可注册多个，按注册顺序触发。"""
        self._handlers.setdefault(event_type, []).append(handler)
        logger.debug("EventBus: subscribed handler for %s", event_type.value)

    async def publish(self, event: CapabilityCompleteEvent) -> None:
        """发布事件，所有处理器以独立 asyncio.Task 运行（fire-and-forget）。"""
        for handler in self._handlers.get(event.type, []):
            asyncio.create_task(
                _safe_call(handler, event),
                name=f"event-{event.type.value}-{event.turn_id[:8]}",
            )


async def _safe_call(handler: Handler, event: CapabilityCompleteEvent) -> None:
    """包装 handler 调用，防止异常传播。"""
    try:
        await handler(event)
    except Exception:
        logger.exception(
            "EventBus: handler %s failed for turn_id=%s",
            getattr(handler, "__name__", repr(handler)),
            event.turn_id,
        )


# ---- 全局单例 ----

_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """返回全局 EventBus 单例（懒加载）。"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
