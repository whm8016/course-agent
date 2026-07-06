"""
Quiz Capability
===============

薄壳：从 context 提取参数，委托给 QuizPipeline 执行。
QuizPipeline 内部以 agent loop 驱动两阶段：ideation → generation。
"""
from __future__ import annotations

import logging
import time

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.observability import log_flow
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)


class QuizCapability(BaseCapability):
    manifest = CapabilityManifest(
        name="quiz",
        description="Agent loop 驱动的智能出题（ideation → generation）",
        stages=["ideation", "generation"],
    )

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        from core.question.pipeline import QuizPipeline

        meta = context.metadata
        requirement = meta.get("requirement") or context.user_message
        count = int(meta.get("count", 1))

        _t0 = time.perf_counter()
        log_flow("quiz.start", requirement=requirement[:120], count=count)
        try:
            result = await QuizPipeline().run(
                requirement=str(requirement),
                context=context,
                stream=stream,
                count=count,
            )
            log_flow("quiz.complete",
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                     questions=len(result.get("questions") or []))
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
        except Exception as exc:
            log_flow("quiz.error", level=logging.ERROR,
                     elapsed_ms=int((time.perf_counter() - _t0) * 1000), error=str(exc))
            logger.exception("QuizCapability: pipeline failed")
            await stream.error(f"出题失败：{exc}", source="quiz")
