"""
Stream Bus
==========

单回合异步事件总线：能力（Capability）生产事件，WS/SSE 消费者订阅。

设计要点：
- emit() 接受 StreamEvent 或 plain dict（dict 自动转换，存量代码无需改动）
- subscribe() 先回放历史事件（支持断线重连），再实时接收，yield StreamEvent
- close() 向所有订阅者投递 None 结束迭代
- stage() 上下文管理器，在阶段前后自动发 stage_start / stage_end
- wait_for_input() / submit_input() 支持 ask_user 工具交互暂停
- 模块级全局注册表 register_bus / unregister_bus / get_bus，供 WS 层路由用户回复
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Union

from core.stream import StreamEvent


class StreamBus:
    """Fan-out async event bus for a single capability turn.

    【中文】多订阅者队列 + 历史回放；close() 向所有订阅者投递 None 结束迭代。
    emit() 同时接受 StreamEvent 和 dict，dict 自动通过 StreamEvent.from_dict() 转换。
    wait_for_input() 暂停 loop 等待用户回复；submit_input() 投递回复。
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[StreamEvent | None]] = []
        self._closed = False
        self._history: list[StreamEvent] = []
        self._input_queues: list[asyncio.Queue[str]] = []

    async def emit(self, event: Union[StreamEvent, dict[str, Any]]) -> None:
        """把 event 推送给所有活跃订阅者。接受 StreamEvent 或 plain dict。"""
        if self._closed:
            return
        if isinstance(event, dict):
            event = StreamEvent.from_dict(event)
        self._history.append(event)
        for q in self._subscribers:
            await q.put(event)

    async def subscribe(self) -> AsyncIterator[StreamEvent]:
        """订阅事件流，先回放历史，再实时接收，直到 close() 被调用。"""
        q: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._subscribers.append(q)
        replay_count = len(self._history)
        try:
            for event in self._history[:replay_count]:
                yield event
            if self._closed and q.empty():
                return
            while True:
                event = await q.get()
                if event is None:
                    break
                yield event
        finally:
            if q in self._subscribers:
                self._subscribers.remove(q)

    async def close(self) -> None:
        """通知所有订阅者流已结束。"""
        self._closed = True
        for q in self._subscribers:
            await q.put(None)

    # ---- 交互工具支持 ----

    async def wait_for_input(self, prompt: str = "") -> str:
        """暂停 agentic loop，等待用户回复（ask_user 工具使用）。

        发出 WAIT_FOR_INPUT 事件通知前端，挂起直到 submit_input() 被调用。
        """
        await self.emit({"type": "wait_for_input", "prompt": prompt})
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._input_queues.append(q)
        try:
            return await q.get()
        finally:
            if q in self._input_queues:
                self._input_queues.remove(q)

    def submit_input(self, content: str) -> None:
        """把用户回复投递给等待中的 wait_for_input 调用（非阻塞）。"""
        for q in self._input_queues:
            try:
                q.put_nowait(content)
            except asyncio.QueueFull:
                pass

    # ---- 便捷生产者 helpers ----

    @asynccontextmanager
    async def stage(self, name: str, source: str = ""):
        """包裹一个阶段，自动发 stage_start / stage_end。"""
        await self.emit({"type": "stage_start", "stage": name, "source": source})
        try:
            yield
        finally:
            await self.emit({"type": "stage_end", "stage": name, "source": source})

    async def progress(self, stage: str, status: str, **extra: Any) -> None:
        await self.emit({"type": "progress", "stage": stage, "status": status, **extra})

    async def error(self, message: str, source: str = "") -> None:
        await self.emit({"type": "error", "message": message, "source": source})

    # ---- consumer adapter ----

    @staticmethod
    def event_to_sse(event: StreamEvent) -> str:
        """序列化为 SSE data 行（含尾部双换行）。"""
        return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# 全局 turn_id → StreamBus 注册表
# 用途：WS 层收到 submit_user_reply 时，通过 turn_id 找到对应 bus 投递回复。
# ---------------------------------------------------------------------------

_bus_registry: dict[str, StreamBus] = {}


def register_bus(turn_id: str, bus: StreamBus) -> None:
    """在全局注册表中注册 turn_id → bus 映射。"""
    _bus_registry[turn_id] = bus


def unregister_bus(turn_id: str) -> None:
    """从全局注册表中移除 turn_id。"""
    _bus_registry.pop(turn_id, None)


def get_bus(turn_id: str) -> StreamBus | None:
    """按 turn_id 查找正在运行的 StreamBus，不存在返回 None。"""
    return _bus_registry.get(turn_id)
