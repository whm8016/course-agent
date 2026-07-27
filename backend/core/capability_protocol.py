"""
Capability Protocol
===================

Level-2 能力契约：所有能力（chat、deep_solve、deep_research、quiz）的抽象基类。

【架构角色】
- CapabilityManifest：静态元数据，描述能力名称、说明、阶段列表。
- BaseCapability：子类实现 run()，在 StreamBus 上发事件，不关心底层 WS/SSE。
- CourseOrchestrator 通过 CapabilityRegistry 找到对应子类，调用 run()。

使用示例：

    class ChatCapability(BaseCapability):
        manifest = CapabilityManifest(
            name="chat",
            description="RAG 增强对话",
            stages=["routing", "responding"],
        )

        async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
            await run_agent_loop(context=context, stream=stream, system_prompt="...")
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


@dataclass
class CapabilityManifest:
    """能力的静态元数据。"""

    name: str
    description: str
    stages: list[str] = field(default_factory=list)


class BaseCapability(ABC):
    """所有课程能力的抽象基类。

    子类必须：
    1. 提供类属性 manifest: CapabilityManifest
    2. 实现 async def run(context, stream) -> None
    """

    manifest: CapabilityManifest

    @abstractmethod
    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        """执行完整能力流水线，通过 stream 发出事件。"""
        ...

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def stages(self) -> list[str]:
        return self.manifest.stages


class TrackedCapability(BaseCapability):
    """共享「计时 + 异常兜底」骨架的能力基类。

    solve / quiz / research 三个能力的 run() 同构：perf_counter 计时 → log_flow *.start
    → 跑 pipeline → log_flow *.complete + emit success 事件 → 异常时 log_flow *.error
    + logger.exception + stream.error。本类把「计时 + try/except + 错误兜底」三件套固化为
    模板方法 run()；子类只实现 run_with_tracking（跑流水线 + 自己打 start/complete 业务日志
    + emit success 事件），异常统一由 run() 兜底。

    chat 不继承本类——它是裸委托（无计时、无兜底），保持 BaseCapability 直系，行为不变。

    异常语义：run_with_tracking 抛出的任何异常都被 run() 捕获，发 log_flow <name>.error
    + logger.exception + stream.error(message=f"{error_label()}：{exc}", source=<name>)，
    与原先各 capability 手写的 except 块逐字等价。
    """

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        _t0 = time.perf_counter()
        try:
            await self.run_with_tracking(context, stream, t0=_t0)
        except Exception as exc:
            log_flow(
                f"{self.name}.error",
                level=logging.ERROR,
                elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                error=str(exc),
            )
            logger.exception("%s: pipeline failed", type(self).__name__)
            await stream.error(f"{self.error_label()}：{exc}", source=self.name)

    @abstractmethod
    async def run_with_tracking(
        self, context: UnifiedContext, stream: StreamBus, *, t0: float
    ) -> None:
        """跑流水线。

        子类在此打 start/complete 业务日志、emit success 事件；抛出的异常由 run() 统一兜底。
        t0 为 run() 入口的计时基准，complete 日志的 elapsed_ms 据此计算，与原先各自在 run()
        开头 perf_counter 的基准一致。
        """
        ...

    def error_label(self) -> str:
        """stream.error 的文案前缀（run() 会拼接「：{exc}」后缀）。

        默认返回能力名；子类覆盖为完整中文文案，使 stream.error 的 message 与原行为逐字一致。
        """
        return self.name
