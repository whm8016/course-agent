"""
Quiz Capability
===============

薄壳：从 context 提取参数，委托给 QuizPipeline 执行。
QuizPipeline 内部以 agent loop 驱动两阶段：ideation → generation。
"""
from __future__ import annotations

import time

from core.capability_protocol import CapabilityManifest, TrackedCapability
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus


class QuizCapability(TrackedCapability):
    manifest = CapabilityManifest(
        name="quiz",
        description="Agent loop 驱动的智能出题（ideation → generation）",
        stages=["ideation", "generation"],
    )

    async def run_with_tracking(
        self, context: UnifiedContext, stream: StreamBus, *, t0: float
    ) -> None:
        from core.question.pipeline import QuizPipeline

        meta = context.metadata
        requirement = meta.get("requirement") or context.user_message
        count = int(meta.get("count", 1))

        log_flow("quiz.start", requirement=requirement[:120], count=count)
        result = await QuizPipeline().run(
            requirement=str(requirement),
            context=context,
            stream=stream,
            count=count,
        )
        log_flow(
            "quiz.complete",
            elapsed_ms=int((time.perf_counter() - t0) * 1000),
            questions=len(result.get("questions") or []),
        )
        await stream.emit({
            "type": "result",
            "questions": result["questions"],
            "metadata": result["metadata"],
        })
        # 出题正常结束信号：pipeline 内部 run_agent_loop 已抑制中间 done，
        # 这里补发唯一一次最终 done，供前端 handleQuizStart 收尾（推题 + 关 WS）。
        await stream.emit({
            "type": "done",
            "metadata": {"mode": "quiz", "count": len(result["questions"])},
        })

    def error_label(self) -> str:
        return "出题失败"
