"""
Deep Research Capability
========================

薄壳：从 context 提取参数，委托给 ResearchPipeline 执行。
ResearchPipeline 内部以 agent loop 驱动三阶段：planning → researching → reporting。
"""
from __future__ import annotations

import logging
import time

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class DeepResearchCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="deep_research",
        description="Agent loop 驱动的深度研究（planning → researching → reporting）",
        stages=["planning", "researching", "reporting"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.research.pipeline import ResearchPipeline

        topic = context.metadata.get("topic") or context.user_message
        _t0 = time.perf_counter()
        log_flow("research.start", topic=topic[:120])

        try:
            result = await ResearchPipeline().run(
                topic=topic,
                context=context,
                stream=stream,
            )
            log_flow("research.complete",
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                     report_chars=len(str(result.get("report") or "")))
            await stream.emit({
                "type": "result",
                "research_id": result["research_id"],
                "report": result["report"],
                "metadata": result["metadata"],
            })
        except Exception as exc:
            log_flow("research.error", level=logging.ERROR,
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000), error=str(exc))
            logger.exception("DeepResearchCapability: pipeline failed")
            await stream.error(f"deep research 失败：{exc}", source="deep_research")
