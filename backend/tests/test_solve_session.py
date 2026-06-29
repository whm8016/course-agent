"""SolveSession 状态机 + solve 工具（contextvar 注入）单测。

验证 Phase 2：deep_solve 的"确定性脊柱"——solve_plan / solve_finish_step /
solve_replan 三工具经 contextvar 拿 session_id，正确读写 SolveSession 状态。
用 asyncio.run 驱动 async 工具，不依赖 pytest-asyncio 的 mode 配置。

注意：set_plan 接收 (id, goal) 2-tuple（parse_steps 的输出格式），id 由 parse_steps
服务端生成（S1/S2…）；模型层只提供 [{goal}, ...]。
"""
import asyncio
import json

from core.agent.tool_registry import execute_tool
from core.solve.session import (
    SolveSession,
    parse_steps,
    reset_current_solve_session,
    set_current_solve_session,
)


def test_set_plan_and_step_progress():
    s = SolveSession(session_id="t1")
    s.set_plan("分析", [("S1", "移项"), ("S2", "验证")])
    assert [st.id for st in s.steps] == ["S1", "S2"]
    assert s.next_step().id == "S1"
    assert not s.all_done()
    s.mark_done("S1", "x=2")
    assert s.steps[0].done and s.next_step().id == "S2"
    s.mark_done("S2", "代入成立")
    assert s.all_done()


def test_replan_budget_gate():
    s = SolveSession(session_id="t2", max_replans=1)
    s.set_plan("a", [("S1", "g1")])
    assert s.replan("换思路", [("S1", "g2")]) is True
    assert s.replans == 1 and s.steps[0].goal == "g2"
    # 预算耗尽：replan 失败且 plan 不被覆盖
    assert s.replan("再换", [("S1", "g3")]) is False
    assert s.steps[0].goal == "g2"


def test_parse_steps_normalizes_ids():
    # dict 形式：id 服务端生成
    assert parse_steps([{"goal": "a"}, {"goal": "b"}]) == [("S1", "a"), ("S2", "b")]
    # 空 goal / None 被丢弃；非空裸值转字符串保留
    assert parse_steps([{"goal": ""}, "x", None]) == [("S1", "x")]


def test_solve_tools_via_contextvar():
    """solve 三工具经 contextvar 拿 session_id，正确读写 session（真实 execute_tool 路径）。"""

    async def _run():
        token = set_current_solve_session("turn-xyz")
        try:
            # solve_plan（模型传 [{goal}]，工具内部 parse_steps 生成 id）
            r = await execute_tool(
                "solve_plan", course_id="",
                analysis="解方程", steps=[{"goal": "移项"}, {"goal": "验证"}],
            )
            assert r.success
            d = json.loads(r.content)
            assert d["status"] == "planned" and d["next"]["id"] == "S1"

            # solve_finish_step
            r = await execute_tool(
                "solve_finish_step", course_id="", step_id="S1", summary="x=2",
            )
            assert r.success
            d = json.loads(r.content)
            assert d["completed"] == "S1" and d["next"]["id"] == "S2" and not d["all_done"]

            # solve_replan
            r = await execute_tool(
                "solve_replan", course_id="", reason="换思路", steps=[{"goal": "直接代入"}],
            )
            assert r.success
            d = json.loads(r.content)
            assert d["status"] == "replanned" and d["replans_used"] == 1
        finally:
            reset_current_solve_session(token)

        # contextvar reset 后：无 session，solve 工具不可用
        r = await execute_tool("solve_plan", course_id="", analysis="a", steps=[{"goal": "g"}])
        assert not r.success

    asyncio.run(_run())
