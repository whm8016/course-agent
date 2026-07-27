"""eval_capabilities（L3 四能力评测）回归测试。

对齐 test_ragas_eval_smoke.py：mock orchestrator / 不依赖真实 LLM key / 缺 inspect_ai 自动跳过。
覆盖可独立验证的核心逻辑（判分纯函数 + 事件流聚合 + 门禁 + EvalLog 解析 + 配置契约）；
依赖真实 LLM 的 judge（quiz_quality / solve_answer / research_*）与完整 inspect eval 留真机验证
（对齐 ragas smoke 不测 RAGAS 完整 evaluate 的粒度）。

注：inspect_ai 0.3.249 的 TaskState.__init__ 要求 model/sample_id/epoch/input/messages 全填，
input 接受 str，input_text 即从 input 推导，故 messages 留空即可（无需构造 ChatMessage 对象）。
"""
import types

import pytest

# inspect_ai 是 L3 底座；未装（精简 CI）则整文件跳过，绝不阻塞 L1
pytest.importorskip("inspect_ai")


# ---------------------------------------------------------------------------
# 假 orchestrator：handle(ctx) 返回固定事件流（模拟真实 capability 的产出）
# ---------------------------------------------------------------------------
class _FakeOrchestrator:
    def __init__(self, events):
        self._events = events

    def handle(self, ctx):
        return self._async_handle()

    async def _async_handle(self):
        for ev in self._events:
            yield ev


def _patch_orchestrator(monkeypatch, events):
    """solver.run_orchestrator 内部 get_orchestrator().handle(ctx) → 替成假流。"""
    monkeypatch.setattr(
        "scripts.eval_capabilities.solver.get_orchestrator",
        lambda: _FakeOrchestrator(events),
    )


def _make_state(metadata=None):
    """构造最小 TaskState（model/sample_id/epoch/input/messages 全必填，messages 留空）。"""
    from inspect_ai.solver import TaskState

    state = TaskState(model="test/mock", sample_id=0, epoch=0, input="test", messages=[])
    for k, v in (metadata or {}).items():
        state.metadata[k] = v
    return state


# ---------------------------------------------------------------------------
# solver：事件流聚合
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_orchestrator_aggregates_answer_quiz_tools(monkeypatch):
    from core.stream import StreamEvent, StreamEventType
    from scripts.eval_capabilities.solver import build_context, run_orchestrator

    events = [
        StreamEvent(type=StreamEventType.ANSWER, source="cap", payload={"content": "基尔霍夫"}),
        StreamEvent(type=StreamEventType.TOKEN, source="cap", payload={"content": "定律"}),
        StreamEvent(type=StreamEventType.QUIZ, source="quiz", payload={"question": "q", "answer": "a"}),
        StreamEvent(type=StreamEventType.TOOL_CALL, source="tool", payload={"tool": "solve_plan"}),
        StreamEvent(type=StreamEventType.DONE, source="cap", payload={}),
    ]
    _patch_orchestrator(monkeypatch, events)

    ctx = build_context(_make_state({"mode": "chat", "course_id": "c", "rag_mode": "naive"}))
    res = await run_orchestrator(ctx)

    assert res["answer"] == "基尔霍夫定律"  # ANSWER + TOKEN 拼接进 completion
    assert res["quiz"] == [{"question": "q", "answer": "a"}]
    assert res["tools"] == ["solve_plan"]
    assert res["error"] == ""
    assert len(res["trace"]) == 5  # 全量事件入 trace


@pytest.mark.asyncio
async def test_run_orchestrator_captures_error_event(monkeypatch):
    """ERROR 事件降级记入 error，不向上抛（capability 异常被 orchestrator 封成 ERROR）。"""
    from core.stream import StreamEvent, StreamEventType
    from scripts.eval_capabilities.solver import build_context, run_orchestrator

    events = [
        StreamEvent(type=StreamEventType.ANSWER, source="cap", payload={"content": "部分"}),
        StreamEvent(type=StreamEventType.ERROR, source="cap", payload={"content": "模型超时"}),
    ]
    _patch_orchestrator(monkeypatch, events)

    ctx = build_context(_make_state())
    res = await run_orchestrator(ctx)

    assert res["error"] == "模型超时"
    assert res["answer"] == "部分"  # ERROR 之前的 answer 仍保留


# ---------------------------------------------------------------------------
# scorer 判分纯函数（确定性逻辑用确定性断言，不浪费 LLM judge）
# ---------------------------------------------------------------------------
def test_extract_score_01_handles_formats():
    from scripts.eval_capabilities.scorer import _extract_score_01

    assert _extract_score_01("85/100") == pytest.approx(0.85)
    assert _extract_score_01("总分：92/100") == pytest.approx(0.92)
    assert _extract_score_01("分数：0.8") == pytest.approx(0.8)
    assert _extract_score_01("") == 0.0
    assert _extract_score_01("无分数输出") == 0.0


def test_valid_quiz_item_schema():
    from scripts.eval_capabilities.scorer import _valid_quiz_item

    assert _valid_quiz_item({"question": "q", "answer": "a"}) is True
    assert _valid_quiz_item({"stem": "q", "correct": "a"}) is True
    assert _valid_quiz_item({"question": "q"}) is False  # 缺答案
    assert _valid_quiz_item({"answer": "a"}) is False  # 缺题干
    assert _valid_quiz_item("not-a-dict") is False


def test_trajectory_legal_reads_unfolded_trace():
    """_trajectory_legal 读 event.to_dict() 后的 trace（payload 字段已展开到顶层）。"""
    from scripts.eval_capabilities.scorer import _trajectory_legal

    ok = [
        {"type": "tool_call", "tool": "solve_plan"},
        {"type": "tool_call", "tool": "solve_finish_step"},
    ]
    assert _trajectory_legal(ok) is True
    assert _trajectory_legal([{"type": "tool_call", "tool": "other"}]) is False
    assert _trajectory_legal([]) is False


def test_scorer_factories_instantiate():
    """@scorer 装配不崩（chat 用内置 model_graded_qa，其余自定义 mean_score scorer）。"""
    from scripts.eval_capabilities import scorer as sc

    assert sc.chat_scorer() is not None
    assert sc.quiz_validity() is not None
    assert sc.quiz_quality() is not None
    assert sc.solve_trajectory() is not None
    assert sc.solve_answer() is not None
    assert sc.research_race() is not None
    assert sc.research_fact() is not None


# ---------------------------------------------------------------------------
# 门禁
# ---------------------------------------------------------------------------
def test_check_gate_pass():
    from scripts.eval_capabilities.gate import check_gate

    passed, fails = check_gate("chat", {"accuracy": 0.85})
    assert passed and fails == []


def test_check_gate_fail_on_low():
    from scripts.eval_capabilities.gate import check_gate

    passed, fails = check_gate("chat", {"accuracy": 0.5})
    assert not passed
    assert any("accuracy" in f for f in fails)


def test_check_gate_skips_missing_metrics():
    """本轮未产出的指标（scores 无该 key）跳过，不误判失败——与 eval_rag 一致。"""
    from scripts.eval_capabilities.gate import check_gate

    passed, fails = check_gate("solve", {"trajectory_legal": 1.0})
    # answer_correctness 缺失 → 跳过，仅凭 trajectory_legal 通过
    assert passed and fails == []


# ---------------------------------------------------------------------------
# EvalLog 解析：scorer 名 → gate metric key 映射
# ---------------------------------------------------------------------------
class _FakeMetric:
    def __init__(self, v):
        self.value = v


class _FakeScore:
    def __init__(self, name, metrics):
        self.name = name
        self.metrics = metrics


def test_extract_scores_maps_scorer_to_metric():
    from scripts.eval_capabilities.run import _extract_scores

    log = types.SimpleNamespace(
        results=types.SimpleNamespace(
            scores=[
                _FakeScore("model_graded_qa", {"accuracy": _FakeMetric(0.85)}),
                _FakeScore("quiz_validity", {"mean_score": _FakeMetric(0.95)}),
                _FakeScore("solve_trajectory", {"mean_score": _FakeMetric(1.0)}),
            ]
        )
    )
    scores = _extract_scores(log, "chat")
    assert scores == {"accuracy": 0.85, "validity": 0.95, "trajectory_legal": 1.0}


def test_extract_scores_handles_missing_results():
    from scripts.eval_capabilities.run import _extract_scores

    log = types.SimpleNamespace(results=None)
    assert _extract_scores(log, "chat") == {}


# ---------------------------------------------------------------------------
# 配置契约：gate 每个 metric key 必须有 scorer 产出，否则门禁形同虚设
# ---------------------------------------------------------------------------
def test_gate_metrics_all_mappable():
    from scripts.eval_capabilities import config
    from scripts.eval_capabilities.run import _SCORER_TO_METRIC

    mappable = set(_SCORER_TO_METRIC.values())
    for cap, gates in config.QUALITY_GATES.items():
        for metric in gates:
            assert metric in mappable, (
                f"{cap}.{metric} 无对应 scorer 产出，check_gate 永远跳过该指标"
            )
