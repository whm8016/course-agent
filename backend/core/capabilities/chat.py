"""
Chat Capability
===============

薄壳：委托给 ChatPipeline，由 ChatPipeline 负责意图路由和 agent loop 分发。
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
        description="RAG 增强多轮对话（意图路由 → Agent Loop / quiz / summarize / vision）",
        stages=["routing", "responding"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.capabilities.chat_pipeline import ChatPipeline

        await ChatPipeline().run(context, stream)
