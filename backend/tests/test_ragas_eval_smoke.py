"""RAG 评测管线冒烟测试（Phase 5）。

mock LightRAG（core.rag.get_retriever）与 RAGAS 求值结果对象，确保评测管线核心逻辑不崩。
不依赖真实 DashScope API / LightRAG 知识库；numpy/scipy/pandas 缺失时对应用例自动跳过。

覆盖：
  - Phase 1: rag_runner 上下文两步拆分（Bug 1：answer 不回填 contexts）
  - Phase 1/2: 逐条分数提取（Bug 3）+ NaN 容错
  - Phase 4: 分布 / Welch t-test 回归 / 历史对比 delta
  - Phase 5: 质量门禁阈值检查
"""
import pytest


# ---------------------------------------------------------------------------
# Phase 1: rag_runner 上下文拆分 + Bug 1（answer 不混入 contexts）
# ---------------------------------------------------------------------------
def test_split_context_string():
    from scripts.eval_rag.rag_runner import _split_context_string
    assert _split_context_string("") == []
    assert _split_context_string("   ") == []
    assert _split_context_string("单条") == ["单条"]
    assert _split_context_string("[证据1]a\n\n---\n\n[证据2]b") == ["[证据1]a", "[证据2]b"]


@pytest.mark.asyncio
async def test_run_lightrag_query_two_step_no_answer_leak(monkeypatch):
    """contexts 来自 retrieve_context（拼接字符串拆分），answer 来自主 LLM（chat_complete），
    两者独立；answer 不回填 contexts（Bug 1）。对齐 rag_runner 复刻生产路径的重构。"""
    import sys
    import types

    # 假 core.rag：retrieve_context 返回拼接字符串（生产路径格式）
    fake_mod = types.ModuleType("core.rag")
    fake_retriever = types.SimpleNamespace()

    async def fake_retrieve_context(*a, **k):
        return "[证据1]ctx-A"

    fake_retriever.retrieve_context = fake_retrieve_context
    fake_mod.get_retriever = lambda name: fake_retriever
    monkeypatch.setitem(sys.modules, "core.rag", fake_mod)

    # 假主 LLM：chat_complete 返回固定 answer（与 contexts 来源完全独立）
    fake_llm = types.ModuleType("core.llm.llm")

    async def fake_chat_complete(*a, **k):
        return "answer-A"

    fake_llm.chat_complete = fake_chat_complete
    monkeypatch.setitem(sys.modules, "core.llm.llm", fake_llm)

    from scripts.eval_rag.rag_runner import _run_lightrag_query
    res = await _run_lightrag_query("c1", "问题", "fact")
    assert res["contexts"] == ["[证据1]ctx-A"]
    assert res["answer"] == "answer-A"
    # Bug 1 核心断言：answer 不应混入 contexts
    assert "answer-A" not in res["contexts"]
    assert res["retrieve_ms"] >= 0 and res["query_ms"] >= 0


@pytest.mark.asyncio
async def test_run_lightrag_query_production_parity(monkeypatch):
    """production-parity 走 retrieve_context（拼接字符串），按 --- 拆成 list。"""
    import sys
    import types

    fake_mod = types.ModuleType("core.rag")
    fake_retriever = types.SimpleNamespace()

    async def fake_retrieve_context(*a, **k):
        return "[证据1]a\n\n---\n\n[证据2]b"

    async def fake_query(*a, **k):
        return {"answer": "ans"}

    fake_retriever.retrieve_context = fake_retrieve_context
    fake_retriever.query = fake_query
    fake_mod.get_retriever = lambda name: fake_retriever

    monkeypatch.setitem(sys.modules, "core.rag", fake_mod)

    from scripts.eval_rag.rag_runner import _run_lightrag_query
    res = await _run_lightrag_query("c1", "q", "naive", production_parity=True)
    assert res["contexts"] == ["[证据1]a", "[证据2]b"]


# ---------------------------------------------------------------------------
# Phase 1/2: 逐条分数提取（Bug 3）+ NaN 容错
# ---------------------------------------------------------------------------
def test_safe_float():
    from scripts.eval_rag.ragas_evaluator import _safe_float
    assert _safe_float(0.85) == 0.85
    assert _safe_float(float("nan")) == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(None) == 0.0
    assert _safe_float("abc") == 0.0


def test_extract_scores_from_scores_attr():
    from scripts.eval_rag.ragas_evaluator import _extract_scores

    class FakeResult:
        scores = [
            {"faithfulness": 0.9, "context_precision": 0.8},
            {"faithfulness": 1.0, "context_precision": 0.6},
        ]

    avg, per_q, invalid = _extract_scores(FakeResult(), ["faithfulness", "context_precision"])
    assert avg["faithfulness"] == pytest.approx(0.95)
    assert avg["context_precision"] == pytest.approx(0.7)
    assert len(per_q) == 2 and per_q[0]["faithfulness"] == 0.9
    # 全有效分数 → 无判崩样本
    assert invalid == {"faithfulness": [], "context_precision": []}


def test_extract_scores_from_to_pandas():
    """to_pandas fallback：每行一条样本的逐条分数。"""
    pytest.importorskip("pandas")
    import pandas as pd
    from scripts.eval_rag.ragas_evaluator import _extract_scores

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame({
                "user_input": ["q1", "q2"],
                "response": ["a1", "a2"],
                "faithfulness": [0.9, 0.7],
            })

    avg, per_q, invalid = _extract_scores(FakeResult(), ["faithfulness"])
    assert avg["faithfulness"] == pytest.approx(0.8)
    assert per_q[1]["faithfulness"] == 0.7
    assert invalid == {"faithfulness": []}


def test_extract_scores_nan_excluded_from_avg():
    """判崩（NaN）样本记入 invalid 清单，且不计入均值（RAGAS误判修复 Step2 核心）。"""
    from scripts.eval_rag.ragas_evaluator import _extract_scores

    class FakeResult:
        scores = [
            {"faithfulness": 0.9},
            {"faithfulness": float("nan")},  # 第2条判崩
            {"faithfulness": 0.7},
        ]

    avg, per_q, invalid = _extract_scores(FakeResult(), ["faithfulness"])
    # NaN 样本已剔出均值：(0.9+0.7)/2 = 0.8，而非 (0.9+0+0.7)/3≈0.533
    assert avg["faithfulness"] == pytest.approx(0.8)
    assert invalid == {"faithfulness": [1]}
    assert per_q[1]["faithfulness"] == 0.0  # 判崩样本填 0.0 保持 CSV 可用


# ---------------------------------------------------------------------------
# Phase 4: 统计引擎
# ---------------------------------------------------------------------------
def test_compute_distribution():
    pytest.importorskip("numpy")
    from scripts.eval_rag.stats import compute_distribution
    d = compute_distribution([0.2, 0.5, 0.8, 0.9, 1.0])
    assert d["mean"] == pytest.approx(0.68, abs=0.01)
    assert d["min"] == 0.2 and d["max"] == 1.0
    assert d["p50"] == pytest.approx(0.8, abs=0.01)
    assert d["p90"] >= d["p50"] >= d["min"]
    empty = compute_distribution([])
    assert empty["mean"] == 0.0


def test_regression_test_detects_drop():
    pytest.importorskip("scipy")
    from scripts.eval_rag.stats import regression_test
    r = regression_test([0.9, 0.88, 0.91, 0.89, 0.9], [0.5, 0.55, 0.48, 0.52, 0.5])
    assert r["regression"] is True
    r2 = regression_test([0.8, 0.82, 0.79], [0.81, 0.80, 0.82])
    assert r2["regression"] is False
    # 样本不足不误报
    assert regression_test([0.9], [0.5])["regression"] is False


def test_diff_against_baseline():
    from scripts.eval_rag.stats import diff_avg_against
    cur = {"mix": {"faithfulness": 0.9}}
    base = {"mix": {"faithfulness": 0.85}}
    assert diff_avg_against(cur, base, ["faithfulness"])["mix"]["faithfulness"] == pytest.approx(0.05)
    # 首次运行无 baseline：delta = current（不减）
    assert diff_avg_against(cur, None, ["faithfulness"])["mix"]["faithfulness"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Phase 5: 质量门禁
# ---------------------------------------------------------------------------
def test_quality_gate_pass():
    from scripts.eval_rag.quality_gate import check_quality_gate
    summary = {"avg_scores": {"mix": {"faithfulness": 0.9}}, "modes": ["mix"], "latency": {}}
    passed, fails = check_quality_gate(summary)
    assert passed and fails == []


def test_quality_gate_fail_on_low_score():
    from scripts.eval_rag.quality_gate import check_quality_gate
    summary = {"avg_scores": {"mix": {"faithfulness": 0.5}}, "modes": ["mix"], "latency": {}}
    passed, fails = check_quality_gate(summary)
    assert not passed
    assert any("faithfulness" in f for f in fails)


def test_quality_gate_fail_on_high_noise():
    from scripts.eval_rag.quality_gate import check_quality_gate
    summary = {"avg_scores": {"mix": {"noise_sensitivity": 0.5}}, "modes": ["mix"], "latency": {}}
    passed, fails = check_quality_gate(summary)
    assert not passed
    assert any("noise_sensitivity" in f for f in fails)


def test_quality_gate_fail_on_latency():
    from scripts.eval_rag.quality_gate import check_quality_gate
    summary = {
        "avg_scores": {"mix": {}}, "modes": ["mix"],
        "latency": {"mix": {"total_ms": {"p95": 6000}}},  # 超 5000 阈值
    }
    passed, fails = check_quality_gate(summary)
    assert not passed
    assert any("延迟" in f for f in fails)


def test_quality_gate_skips_unevaluated_metrics():
    """本轮没评测的指标（avg_scores 无该 metric）不应判失败。"""
    from scripts.eval_rag.quality_gate import check_quality_gate
    summary = {"avg_scores": {"mix": {"faithfulness": 0.9}}, "modes": ["mix"], "latency": {}}
    passed, fails = check_quality_gate(summary)
    # factual_correctness 等未评测，不应出现在 failures
    assert passed
