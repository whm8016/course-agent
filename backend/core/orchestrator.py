"""
Course Orchestrator
===================

统一入口：将 UnifiedContext 路由到对应的 Capability，管理内部 StreamBus 生命周期。

【调用链（更新后）】
  TurnRuntimeManager._run_turn()
    → async for event in CourseOrchestrator.handle(context):
        await execution.bus.emit(event)          ← TRM 做 fan-out

  CourseOrchestrator.handle() 内部：
    → CapabilityRegistry.get(context.mode)
    → 创建内部 StreamBus，注册到全局 bus 注册表（供 submit_input 路由）
    → asyncio.create_task(capability.run(context, bus))
    → yield from bus.subscribe()                 ← 向 TRM 流式吐出事件

设计原则：
- Orchestrator 不执行任何业务逻辑，只负责"选能力、跑能力、处理异常、注册 bus"。
- handle() 改为 async generator，让 TRM 可在事件流中间插入持久化/EventBus 逻辑。
- bus 在 orchestrator 内创建并注册，能力写 bus；TRM 消费 generator 做 fan-out。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from core.context import UnifiedContext
from core.observability import log_flow
from core.registry import get_capability_registry
from core.stream import StreamEvent
from core.stream_bus import StreamBus, register_bus, unregister_bus

logger = logging.getLogger(__name__)


class CourseOrchestrator:
    """按 context.mode 分派到对应 Capability，yield StreamEvent 给调用方。"""

    def __init__(self) -> None:
        self._registry = get_capability_registry()

    async def handle(self, context: UnifiedContext) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        """执行单回合，以 async generator 形式 yield StreamEvent。

        调用方（TurnRuntimeManager）负责把事件 emit 到自己的 fan-out bus。
        """
        cap_name = context.mode or "chat"
        capability = self._registry.get(cap_name)

        if capability is None:
            available = self._registry.list_capabilities()
            log_flow("orchestrator.unknown_capability", level=logging.WARNING,
                     capability=cap_name, available=available)
            bus = StreamBus()
            await bus.error(
                f"未知能力：{cap_name}。可用能力：{available}",
                source="orchestrator",
            )
            await bus.close()
            async for event in bus.subscribe():
                yield event
            return

        log_flow("orchestrator.route", capability=cap_name,
                 course_id=context.course_id, user_id=context.user_id)

        bus = StreamBus()
        turn_id = str(context.metadata.get("turn_id", ""))
        if turn_id:
            register_bus(turn_id, bus)

        async def _run() -> None:
            try:
                await capability.run(context, bus)
            except Exception as exc:
                logger.exception("Capability '%s' raised: %s", cap_name, exc)
                log_flow("orchestrator.capability_error", level=logging.ERROR,
                         capability=cap_name, error=str(exc))
                await bus.error(str(exc), source=cap_name)
            finally:
                if turn_id:
                    unregister_bus(turn_id)
                await bus.close()

        task = asyncio.create_task(_run())
        try:
            async for event in bus.subscribe():
                yield event
        finally:
            if not task.done():
                task.cancel()

    def list_capabilities(self) -> list[str]:
        return self._registry.list_capabilities()

    def get_manifests(self) -> list[dict]:
        return self._registry.get_manifests()


# ---- 全局单例 ----

_orchestrator: CourseOrchestrator | None = None


def get_orchestrator() -> CourseOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CourseOrchestrator()
    return _orchestrator
