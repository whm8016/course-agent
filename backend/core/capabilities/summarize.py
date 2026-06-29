"""
Summarize Capability
====================

薄壳：对当前会话历史生成学习小结。对齐 DeepTutor 独立 Capability 模式。
"""
from __future__ import annotations

import logging
import time

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class SummarizeCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="summarize",
        description="基于对话历史生成学习小结",
        stages=["summarizing"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.agent.orchestrator import _stream_summarize_events
        _t0 = time.perf_counter()
        log_flow("summarize.start")
        event_count = 0
        async for event in _stream_summarize_events(context):
            await stream.emit(event)
            event_count += 1
        log_flow("summarize.complete",
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                 events=event_count)
