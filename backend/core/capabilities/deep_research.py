"""
Deep Research Capability
========================

薄壳：从 context 提取参数，委托给 ResearchPipeline 执行。
ResearchPipeline 内部以 agent loop 驱动四阶段：rephrase → decompose → research → reporting。
"""
from __future__ import annotations

import time

from core.capability_protocol import CapabilityManifest, TrackedCapability
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus


class DeepResearchCapability(TrackedCapability):
    manifest = CapabilityManifest(
        name="deep_research",
        description="Agent loop 驱动的深度研究（rephrase → decompose → research → reporting）",
        stages=["planning", "researching", "reporting"],
    )

    async def run_with_tracking(
        self, context: UnifiedContext, stream: StreamBus, *, t0: float
    ) -> None:
        from core.research.pipeline import ResearchPipeline

        topic = context.metadata.get("topic") or context.user_message
        log_flow("research.start", topic=topic[:120])

        result = await ResearchPipeline().run(
            topic=topic,
            context=context,
            stream=stream,
        )
        log_flow(
            "research.complete",
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            report_chars=len(str(result.get("report") or "")),
        )
        await stream.emit({
            "type": "result",
            "research_id": result["research_id"],
            "report": result["report"],
            "metadata": result["metadata"],
        })

    def error_label(self) -> str:
        return "deep research 失败"
