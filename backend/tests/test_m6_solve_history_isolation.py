"""M-6：deep solve conversation_history 隔离测试。

修复前：solve pipeline 的 replace(context, ...) 漏了 conversation_history=[]，
导致 chat 积累的历史对话泄入 solve 的 _build_messages，被拼进解题 LLM 调用。

修复后：solve 的 replace 显式置 conversation_history=[]，即使原始 context 带长历史，
solve_ctx 传给 run_agent_loop 时也是空的——解题是独立回合，不该被闲聊历史污染。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.solve.pipeline import DeepSolvePipeline
from core.stream_bus import StreamBus


async def test_solve_isolates_conversation_history():
    """原始 context 带 3 条闲聊历史，solve_ctx 传给 run_agent_loop 时 conversation_history 必须为空。"""
    ctx = UnifiedContext(
        user_message="解 x+1=3",
        course_id="C1",
        user_id="U1",
        enabled_tools=["rag"],
        conversation_history=[
            {"role": "user", "content": "今天天气真好"},
            {"role": "assistant", "content": "是的，有什么可以帮您？"},
            {"role": "user", "content": "随便聊聊"},
        ],
    )
    bus = StreamBus()
    rt = ProfileRuntime()

    captured_ctx: dict = {}

    async def _capture_run(**kwargs):
        captured_ctx["context"] = kwargs["context"]
        return MagicMock(rounds=1, tools_used=[], final_text="x=2")

    with (
        patch("core.solve.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch(
            "core.solve.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=CommonContextLayers()),
        ),
        patch("core.solve.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.solve.pipeline.get_tool_schemas", return_value=[]),
        patch("core.solve.pipeline.run_agent_loop", new=AsyncMock(side_effect=_capture_run)),
    ):
        await DeepSolvePipeline().run("解 x+1=3", ctx, bus)

    solve_ctx = captured_ctx["context"]
    # 修复前：会是那 3 条闲聊历史（污染解题）
    # 修复后：空列表
    assert solve_ctx.conversation_history == [], (
        "solve 必须隔离 chat 历史，不应把闲聊对话泄入解题 LLM 调用"
    )
    # 原始 context 未被破坏（replace 不应原地改入参）
    assert len(ctx.conversation_history) == 3
    await bus.close()
