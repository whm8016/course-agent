"""DeepSolvePipeline — 单 agent loop + solve 工具状态机（对标 DeepTutor）。

DeepTutor 的解题没有独立多阶段 pipeline：chat agent loop 就是解题器，
通过 solve_plan / solve_finish_step / solve_replan 三个工具 + SolveSession
状态机提供"确定性脊柱"（commit plan、不跳步、bounded replan），实际推理在
loop 出口由模型完成。

本实现复用 run_agent_loop（tool_calls），solve 工具注册在 tool_registry，
session_id 经 contextvar 注入（dispatch_tool_calls 不收 context）。门控靠
SolveSession + system prompt 强调"先 solve_plan"，代码层不强制（tool_calls 版
取舍：provider 保证结构化输出，流程门控靠 prompt 引导 + session 状态读回）。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.agentic.loop import _get_tool_schemas, run_agent_loop
from core.context import UnifiedContext
from core.observability import log_flow
from core.prompt_loader import load_prompt_dict
from core.solve.session import (
    DEFAULT_MAX_REPLANS,
    get_session,
    reset_current_solve_session,
    set_current_solve_session,
)
from core.stream_bus import StreamBus

logger = logging.getLogger(__name__)

# solve playbook（YAML 外部化，对标 DeepTutor capabilities/solve prompts/zh/system.md）
_SOLVE_PROMPT_PATH = Path(__file__).parent / "prompts" / "zh" / "system.yaml"

# solve 确定性脊柱三件套
_SOLVE_TOOLS = ["solve_plan", "solve_finish_step", "solve_replan"]

_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def _resolve_session_id(context: UnifiedContext) -> str:
    """解析本 solve turn 的内存 session key（per-turn，并发 turn 互不竞争）。"""
    raw = str(
        context.metadata.get("turn_id")
        or context.session_id
        or context.metadata.get("message_id")
        or "default"
    )
    cleaned = _UNSAFE_ID_CHARS.sub("_", raw).strip("_")
    return cleaned or "default"


class DeepSolvePipeline:
    """单 loop + solve 工具状态机的解题 pipeline。"""

    async def run(
        self,
        question: str,
        context: UnifiedContext,
        stream: StreamBus,
    ) -> dict[str, Any]:
        rag_enabled = bool(context.course_id and "rag" in context.enabled_tools)

        # 解析 solve 会话 id（per-turn），初始化 session + replan 预算
        sid = _resolve_session_id(context)
        context.metadata["solve_session_id"] = sid
        session = get_session(sid)
        session.max_replans = DEFAULT_MAX_REPLANS

        # solve playbook：强调先 solve_plan、逐步 finish_step、可 replan
        solve_cfg = load_prompt_dict(_SOLVE_PROMPT_PATH)
        system_prompt = solve_cfg.get("system") or ""

        # 可用工具：solve 三件套 + rag（可选）
        enabled = list(_SOLVE_TOOLS) + (["rag"] if rag_enabled else [])

        solve_ctx = replace(
            context,
            user_message=question,
            mode="deep_solve",
            enabled_tools=enabled,
        )

        log_flow("solve.pipeline.start", session_id=sid, rag_enabled=rag_enabled,
                 tools=enabled, question=question[:120])
        _t0 = time.perf_counter()
        # contextvar 注入 session_id（dispatch 不收 context，solve 工具读 contextvar）
        token = set_current_solve_session(sid)
        try:
            outcome = await run_agent_loop(
                context=solve_ctx,
                stream=stream,
                system_prompt=system_prompt,
                tool_schemas=_get_tool_schemas(solve_ctx),
                max_iterations=12,
            )
        finally:
            reset_current_solve_session(token)
        log_flow("solve.pipeline.complete",
                 elapsed_ms=int((time.perf_counter() - _t0) * 1000),
                 rounds=outcome.rounds, tools_used=outcome.tools_used)

        return {
            "final_answer": outcome.final_text,
            "metadata": {
                "tools_used": outcome.tools_used,
                "rounds": outcome.rounds,
                "plan_steps": session.map(),
                "replans_used": session.replans,
            },
        }


__all__ = ["DeepSolvePipeline"]
