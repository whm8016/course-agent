"""
Deep Solve Capability
=====================

薄壳：从 context 提取参数，委托给 DeepSolvePipeline 执行。
DeepSolvePipeline 内部以单 agent loop + solve 工具状态机（solve_plan /
solve_finish_step / solve_replan）驱动解题，无独立多阶段 pipeline。
"""
from __future__ import annotations

import time

from core.capability_protocol import CapabilityManifest, TrackedCapability
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus


class DeepSolveCapability(TrackedCapability):
    manifest = CapabilityManifest(
        name="deep_solve",
        description="Agent loop + solve 工具状态机驱动的深度解题（plan / finish_step / replan）",
        stages=["planning", "solving", "writing"],
    )

    async def run_with_tracking(
        self, context: UnifiedContext, stream: StreamBus, *, t0: float
    ) -> None:
        from core.solve.pipeline import DeepSolvePipeline

        question = context.metadata.get("question") or context.user_message
        log_flow("solve.start", question=question[:120])

        result = await DeepSolvePipeline().run(
            question=question,
            context=context,
            stream=stream,
        )
        log_flow(
            "solve.complete",
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            answer_chars=len(str(result.get("final_answer") or "")),
        )
        await stream.emit({
            "type": "result",
            "final_answer": result["final_answer"],
            "metadata": result["metadata"],
        })

    def error_label(self) -> str:
        return "deep solve 失败"
