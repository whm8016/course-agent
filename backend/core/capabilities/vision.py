"""
Vision Capability
=================

薄壳：图片上传场景，分析图片内容并结合课程知识作答。对齐 DeepTutor 独立 Capability 模式。
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
        log_flow("vision.start", has_image=bool(context.image_path or context.attachments))
        event_count = 0
        async for event in _stream_vision_events(context):
            await stream.emit(event)
            event_count += 1
        log_flow("vision.complete",
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                 events=event_count)
