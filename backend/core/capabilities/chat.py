"""
Chat Capability
===============

薄壳：委托给 ChatPipeline 驱动 chat 模式的 agent loop。
能力路由（chat / deep_solve / deep_research / quiz）由上层 Orchestrator +
CapabilityRegistry 按 context.mode 完成，ChatPipeline 只负责 chat 模式。
"""
from __future__ import annotations

import logging

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class ChatCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="chat",
        description="RAG 增强多轮对话（Agent Loop 驱动；能力路由由上层 Orchestrator 按 context.mode 完成）",
        stages=["routing", "responding"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.capabilities.chat_pipeline import ChatPipeline

        await ChatPipeline().run(context, stream)
