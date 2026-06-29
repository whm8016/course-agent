"""
Deep Solve Capability
=====================

薄壳：从 context 提取参数，委托给 DeepSolvePipeline 执行。
DeepSolvePipeline 内部以 agent loop 驱动三阶段：planning → solving → writing。
"""
from __future__ import annotations

import logging
import time

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class DeepSolveCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="deep_solve",
        description="Agent loop 驱动的深度解题（planning → solving → writing）",
        stages=["planning", "solving", "writing"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.solve.pipeline import DeepSolvePipeline

        question = context.metadata.get("question") or context.user_message
        _t0 = time.perf_counter()
        log_flow("solve.start", question=question[:120])

        try:
            result = await DeepSolvePipeline().run(
                question=question,
                context=context,
                stream=stream,
            )
            log_flow("solve.complete",
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                     answer_chars=len(str(result.get("final_answer") or "")))
            await stream.emit({
                "type": "result",
                "final_answer": result["final_answer"],
                "metadata": result["metadata"],
            })
        except Exception as exc:
            log_flow("solve.error", level=logging.ERROR,
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000), error=str(exc))
            logger.exception("DeepSolveCapability: pipeline failed")
            await stream.error(f"deep solve 失败：{exc}", source="deep_solve")
