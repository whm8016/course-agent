"""
Vision Capability（DEPRECATED）
==============================

前端 CAPABILITIES 仅暴露 chat/deep_solve/quiz/research，本能力未在前端入口暴露。图片问答
统一走 chat pipeline——主模型支持 vision 时 loop 内直注（见 loop._build_messages），不支持
时经 describe_images_into 两阶段降级。本壳保留以兼容旧 mode=vision 调用，其实现
_stream_vision_events 走独立 chat_stream 路径（无 agent loop/profile/skill/memory），功能
落后于主路径，后续应迁移或移除。
"""
from __future__ import annotations

import logging
import time

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class VisionCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="vision",
        description="图片分析：结合课程知识解析上传的图片",
        stages=["analyzing"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.agent.orchestrator import _stream_vision_events
        _t0 = time.perf_counter()
        log_flow("vision.start", has_image=bool(context.attachments))
        event_count = 0
        async for event in _stream_vision_events(context):
            await stream.emit(event)
            event_count += 1
        log_flow("vision.complete",
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                 events=event_count)
