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
    from scripts.eval_capabilities.ablation import _trace_replans, _trace_rounds, _trace_tool_results

    trace = [
        {"type": "tool_call", "tool": "solve_plan"},
        {"type": "tool_result"},
        {"type": "tool_call", "tool": "solve_replan"},
        {"type": "result"},
        {"type": "tool_call", "tool": "solve_finish_step"},
    ]
    assert _trace_tool_results(trace) == 2  # tool_result + result
    assert _trace_replans(trace) == 1
    assert _trace_rounds(trace) == 0  # 无 done 事件
    assert _trace_rounds([{"type": "done", "metadata": {"iterations": 4}}]) == 4


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


@pytest.mark.asyncio
async def test_build_ctx_injects_chat_switch(monkeypatch):
    """chat 消融开关 key=kb_seed，on/off 正确写入 ctx.metadata。"""
    from scripts.eval_capabilities.ablation import _SWITCH_KEY, _build_ctx

    assert _SWITCH_KEY["chat"] == "kb_seed"
    sample = {"input": "q", "metadata": {"mode": "chat", "course_id": "c"}}
    on = _build_ctx(sample, "chat", True)
    off = _build_ctx(sample, "chat", False)
    assert on.metadata["kb_seed"] is True
    assert off.metadata["kb_seed"] is False
    assert on.mode == "chat"


@pytest.mark.asyncio
async def test_score_chat_counts_rag_and_rounds(monkeypatch):
    """_score_chat：rag_calls 计 tools 里 rag 次数，rounds 取 done.iterations。"""
    from scripts.eval_capabilities import ablation

    async def fake_ensemble(prompt):
        return (0.8, "mock")

    monkeypatch.setattr(ablation, "_ensemble_score", fake_ensemble)
    sample = {"input": "q", "target": "ref"}
    result = {
        "answer": "ans",
        "tools": ["rag", "rag", "rag", "web_search"],
        "trace": [{"type": "done", "metadata": {"iterations": 2}}],
    }
    score = await ablation._score_chat(sample, result)
    assert score["rag_calls"] == 3
    assert score["rounds"] == 2
    assert score["race"] == 0.8


@pytest.mark.asyncio
async def test_run_capability_chat_switch_propagation(monkeypatch, tmp_path):
    """chat 消融全程：kb_seed on/off 透传 + rag_calls/rounds delta + 结果落盘。"""
    from scripts.eval_capabilities import ablation

    monkeypatch.setattr(ablation.config, "RESULTS_DIR", tmp_path)

    seen: list[bool] = []

    async def fake_run(ctx):
        switch = ctx.metadata["kb_seed"]
        seen.append(switch)
        # treatment(seed 开)→ 几乎不调 rag、1 轮作答；baseline(seed 关)→ 盲调 8 次、3 轮
        return {
            "answer": "good answer" if switch else "base answer",
            "quiz": [],
            "tools": ["rag"] * (1 if switch else 8),
            "trace": [{"type": "done", "metadata": {"iterations": 1 if switch else 3}}],
            "error": "",
        }

    async def fake_ensemble(prompt):
        return (0.9 if "good" in prompt else 0.5, "mock")

    monkeypatch.setattr(ablation, "run_orchestrator", fake_run)
    monkeypatch.setattr(ablation, "_ensemble_score", fake_ensemble)

    result = await ablation._run_capability("chat")

    # chat.jsonl 5 题 × (baseline off + treatment on)
    assert len(seen) == 10
    assert result["switch"] == "kb_seed"
    assert result["treatment"]["rag_calls"] < result["baseline"]["rag_calls"]
    assert result["treatment"]["rounds"] < result["baseline"]["rounds"]
    assert result["delta"]["rag_calls"] < 0
    assert (tmp_path / "ablation_chat.json").exists()
