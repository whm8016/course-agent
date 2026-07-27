"""能力级消融 runner（scripts.eval_capabilities.ablation）回归测试。

不依赖真实 LLM：mock run_orchestrator（假事件流聚合结果）+ _ensemble_score（固定分），
验证开关 on/off 透传到 ctx.metadata、baseline/treatment/delta 计算、trace 辅助计数。
对齐 test_eval_capabilities.py 的 mock 粒度（judge 真跑留真机）。
"""
import pytest

# ablation → scorer/solver → inspect_ai；未装（精简 CI）跳过，绝不阻塞 L1
pytest.importorskip("inspect_ai")


@pytest.mark.asyncio
async def test_build_ctx_injects_switch(monkeypatch):
    from scripts.eval_capabilities.ablation import _SWITCH_KEY, _build_ctx

    sample = {"input": "q", "metadata": {"mode": "deep_research", "course_id": "c"}}
    on = _build_ctx(sample, "research", True)
    off = _build_ctx(sample, "research", False)
    assert on.metadata[_SWITCH_KEY["research"]] is True
    assert off.metadata[_SWITCH_KEY["research"]] is False
    assert on.metadata["turn_id"] == ""  # 评测仍不注册 bus
    assert on.mode == "deep_research"
    assert on.course_id == "c"


def test_trace_helpers_count():
    from scripts.eval_capabilities.ablation import _trace_replans, _trace_tool_results

    trace = [
        {"type": "tool_call", "tool": "solve_plan"},
        {"type": "tool_result"},
        {"type": "tool_call", "tool": "solve_replan"},
        {"type": "result"},
        {"type": "tool_call", "tool": "solve_finish_step"},
    ]
    assert _trace_tool_results(trace) == 2  # tool_result + result
    assert _trace_replans(trace) == 1


@pytest.mark.asyncio
async def test_run_capability_switch_propagation_and_delta(monkeypatch, tmp_path):
    """mock run_orchestrator + _ensemble_score → 验证 on/off 透传 + delta + 结果落盘。"""
    from scripts.eval_capabilities import ablation

    monkeypatch.setattr(ablation.config, "RESULTS_DIR", tmp_path)

    seen: list[bool] = []

    async def fake_run(ctx):
        switch = ctx.metadata["research_observer"]
        seen.append(switch)
        return {
            "answer": "better report" if switch else "base report",
            "quiz": [], "tools": [], "trace": [], "error": "",
        }

    async def fake_ensemble(prompt):
        # race/fact prompt 都含 answer；treatment 的 "better" → 高分
        return (0.9 if "better" in prompt else 0.5, "mock")

    monkeypatch.setattr(ablation, "run_orchestrator", fake_run)
    monkeypatch.setattr(ablation, "_ensemble_score", fake_ensemble)

    result = await ablation._run_capability("research")

    # 3 题 × (baseline off + treatment on)，开关值交替
    assert seen == [False, True, False, True, False, True]
    assert result["n"] == 3
    assert result["treatment"]["race"] > result["baseline"]["race"]
    assert result["delta"]["race"] > 0
    assert result["treatment"]["fact"] > result["baseline"]["fact"]
    # per_question 结构
    assert len(result["per_question"]) == 3
    assert set(result["per_question"][0]) == {"id", "baseline", "treatment", "delta"}
    # 结果落盘
    assert (tmp_path / "ablation_research.json").exists()
