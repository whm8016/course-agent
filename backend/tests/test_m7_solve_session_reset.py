"""M-7：SolveSession pipeline 入口重置（防 sid 撞车读到 stale plan）。

修复前：get_session(sid) 惰性创建+复用，sid 撞车（同 turn 重试 / message_id 复用 /
LRU 驱逐后同 sid 再入）时返回上一轮残留的 plan/steps/replans，pipeline 入口只设
max_replans 不清状态 → 第二次解题读到第一次的 plan，状态机错乱。

修复后：SolveSession.reset() + pipeline 入口每 turn 调一次，清空 stale plan/进度/replan。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from core.context import UnifiedContext
from core.pipeline_common import CommonContextLayers, ProfileRuntime
from core.solve.pipeline import DeepSolvePipeline
from core.solve.session import SolveSession, get_session
from core.stream_bus import StreamBus


def test_reset_clears_stale_state():
    """reset() 把 analysis/steps/replans 清空，max_replans 保留。"""
    s = SolveSession(session_id="t")
    s.set_plan("旧分析", [("S1", "旧步骤")])
    s.replans = 1
    s.max_replans = 2

    s.reset()

    assert s.analysis == ""
    assert s.steps == []
    assert s.replans == 0
    # max_replans 保留（紧接 pipeline 会重设）
    assert s.max_replans == 2


async def test_pipeline_entry_resets_stale_plan_on_sid_collision():
    """同 sid 进两次 pipeline：第二次入口必须先 reset，看到空 plan 而非第一次残留。

    构造：第一次跑完，session 残留 plan/steps/replans（模拟 solve_plan 已写）。
    第二次跑前，session 仍有残留；修复后第二次入口 reset 掉，再被 fake_loop 写本次 plan。
    若入口没 reset，第二次看到的会是第一次的「本次分析」（其实是上一轮的）——无法区分。
    故用 reset 后立刻断言「入口时 session 是空的」来证明 reset 生效。
    """
    # 第二次 solve 用同 message_id → 同 sid（撞上第一次残留的 session）
    ctx2 = UnifiedContext(user_message="解 y+2=5", course_id="C1", user_id="U1",
                         metadata={"message_id": "msg_collide_1"})
    bus = StreamBus()

    # 先手动污染 session（模拟上一次 solve 留下的 stale 状态 + 直接走 get_session）
    sid = "msg_collide_1"
    stale = get_session(sid)
    stale.set_plan("上一轮的 STALE 分析", [("S1", "上一轮 stale 步骤")])
    stale.replans = 1

    # 在第二次 pipeline 入口后、run_agent_loop 调用前，抓取 session 状态
    captured_after_entry: dict = {}
    rt = ProfileRuntime()

    def _fake_loop_round2(*, context, **kw):
        # 此时 pipeline 入口已执行 reset()，session 应是空 plan
        sess = get_session(context.metadata.get("solve_session_id"))
        captured_after_entry["analysis"] = sess.analysis
        captured_after_entry["steps_len"] = len(sess.steps)
        captured_after_entry["replans"] = sess.replans
        return MagicMock(rounds=1, tools_used=[], final_text="ok")

    with (
        patch("core.solve.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch(
            "core.solve.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=CommonContextLayers()),
        ),
        patch("core.solve.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.solve.pipeline._get_tool_schemas", return_value=[]),
        patch("core.solve.pipeline.run_agent_loop", new=AsyncMock(side_effect=_fake_loop_round2)),
    ):
        await DeepSolvePipeline().run("解 y+2=5", ctx2, bus)

    # 修复前：入口没 reset，captured 会是上一轮的 stale（analysis="上一轮的 STALE 分析"，steps_len=1，replans=1）
    # 修复后：入口 reset 清空
    assert captured_after_entry["analysis"] == "", "入口应 reset 掉 stale analysis"
    assert captured_after_entry["steps_len"] == 0, "入口应 reset 掉 stale steps"
    assert captured_after_entry["replans"] == 0, "入口应 reset 掉 replan 计数"
    await bus.close()


async def test_pipeline_reset_does_not_break_in_turn_writes():
    """reset 只在入口调一次，同 turn 内 solve 工具多次 get_session 读写不受影响。

    模拟：入口 reset → solve_plan 写 plan → solve_finish_step 读到刚写的 plan（不被清）。
    """
    ctx = UnifiedContext(user_message="解 x+1=3", course_id="C1", user_id="U1",
                        metadata={"message_id": "msg_inturn"})
    bus = StreamBus()
    rt = ProfileRuntime()

    call_seq: list[str] = []

    async def _fake_loop(*, context, **kw):
        sid = context.metadata.get("solve_session_id")
        # 模拟 solve_plan
        sess = get_session(sid)
        sess.set_plan("本次分析", [("S1", "步骤一")])
        call_seq.append("after_plan_analysis=" + sess.analysis)
        # 模拟 solve_finish_step 再次 get_session 读取
        sess2 = get_session(sid)
        call_seq.append("reread_analysis=" + sess2.analysis)
        call_seq.append("reread_steps=" + str(len(sess2.steps)))
        return MagicMock(rounds=1, tools_used=[], final_text="ok")

    with (
        patch("core.solve.pipeline.resolve_profile_runtime", new=AsyncMock(return_value=rt)),
        patch(
            "core.solve.pipeline.build_common_context_layers",
            new=AsyncMock(return_value=CommonContextLayers()),
        ),
        patch("core.solve.pipeline.describe_images", new=AsyncMock(side_effect=lambda c, t, r: t)),
        patch("core.solve.pipeline._get_tool_schemas", return_value=[]),
        patch("core.solve.pipeline.run_agent_loop", new=AsyncMock(side_effect=_fake_loop)),
    ):
        await DeepSolvePipeline().run("解 x+1=3", ctx, bus)

    # solve_plan 写完后能读到自己写的（证明 reset 没在 get_session 里，不影响同 turn 读写）
    assert "after_plan_analysis=本次分析" in call_seq
    assert "reread_analysis=本次分析" in call_seq
    assert "reread_steps=1" in call_seq
    await bus.close()
