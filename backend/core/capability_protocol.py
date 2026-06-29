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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.context import UnifiedContext
from core.stream_bus import StreamBus


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
