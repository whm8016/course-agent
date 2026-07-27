"""能力级消融实验：开关 on/off 对比，复用 run_orchestrator + scorer 判分函数。

为什么不复用 Inspect AI 的 task（tasks.py）+ inspect_eval：mean_score 全局聚合，无法按
「开关 on/off」分组出分。这里直接对 dataset 每题跑两遍（baseline 开关关 / treatment 开关开），
用 scorer 的判分 prompt + _ensemble_score 出分，输出 on/off delta 对比表。

真跑需 LLM key（judge 走 _ensemble_score）；mock 测试 monkeypatch 本模块 run_orchestrator +
_ensemble_score 验证开关透传 + delta 逻辑，不依赖真 LLM。

用法：python -m scripts.eval_capabilities.ablation --capability research|solve|all
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from core.agent.mode_normalize import normalize_mode
from core.context import UnifiedContext

from . import config
from .scorer import (
    _ensemble_score,
    _fact_prompt,
    _race_prompt,
    _solve_answer_prompt,
    _trajectory_legal,
)
from .solver import run_orchestrator


# capability → 消融开关 metadata key（research pipeline 读 research_observer，
# solve pipeline 读 solve_force_replan，二者默认 False）
_SWITCH_KEY = {
    "research": "research_observer",
    "solve": "solve_force_replan",
}

# 对比用指标（不含 _explain 私有解释）
_METRICS = {
    "research": ["race", "fact", "retrievals"],
    "solve": ["answer_correctness", "trajectory_legal", "replans"],
}


class _Target:
    """scorer 判分 prompt 只访问 target.text，用本地 stub 避免依赖 inspect Target 构造签名。"""

    def __init__(self, text: str) -> None:
        self.text = text


def _load_samples(name: str) -> list[dict[str, Any]]:
    path = config.DATASETS_DIR / f"{name}.jsonl"
    samples: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def _build_ctx(sample: dict[str, Any], capability: str, switch_on: bool) -> UnifiedContext:
    """构造评测 ctx：在 sample metadata 上注入消融开关值（baseline=False / treatment=True）。

    复用 solver.build_context 的构造范式，额外把开关 key 按 capability 写进 metadata，
    run_orchestrator → orchestrator → pipeline 读到对应开关。
    """
    meta = dict(sample.get("metadata", {}) or {})
    meta[_SWITCH_KEY[capability]] = switch_on
    return UnifiedContext(
        user_message=sample["input"],
        mode=normalize_mode(meta.get("mode", "chat")),
        course_id=meta.get("course_id", ""),
        rag_mode=meta.get("rag_mode", "naive"),
        conversation_history=meta.get("history", []) or [],
        language=meta.get("language", "zh"),
        metadata={**meta, "turn_id": ""},
    )


def _trace_tool_results(trace: list[dict]) -> int:
    """trace 里 tool_result 事件数（检索次数近似，对齐 _fact_prompt 抽来源方式）。"""
    return sum(1 for ev in trace or [] if ev.get("type") in ("tool_result", "result"))


def _trace_replans(trace: list[dict]) -> int:
    """solve replan 次数（tool_call 含 solve_replan）。"""
    return sum(
        1 for ev in trace or []
        if ev.get("type") == "tool_call"
        and "replan" in str(ev.get("tool") or ev.get("name", ""))
    )


async def _score_research(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer", "")
    trace = result.get("trace", [])
    race, race_expl = await _ensemble_score(
        _race_prompt(answer, sample["input"], _Target(sample.get("target", "")))
    )
    fact, fact_expl = await _ensemble_score(_fact_prompt(answer, trace))
    return {
        "race": round(race, 4),
        "fact": round(fact, 4),
        "retrievals": _trace_tool_results(trace),
        "_explain": {"race": race_expl, "fact": fact_expl},
    }


async def _score_solve(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer", "")
    trace = result.get("trace", [])
    ans, ans_expl = await _ensemble_score(
        _solve_answer_prompt(answer, sample["input"], _Target(sample.get("target", "")))
    )
    return {
        "answer_correctness": round(ans, 4),
        "trajectory_legal": 1.0 if _trajectory_legal(trace) else 0.0,
        "replans": _trace_replans(trace),
        "_explain": {"answer": ans_expl},
    }


_SCORERS = {"research": _score_research, "solve": _score_solve}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


async def _run_capability(capability: str) -> dict[str, Any]:
    samples = _load_samples(capability)
    scorer = _SCORERS[capability]
    metrics = _METRICS[capability]
    per_question: list[dict[str, Any]] = []

    for sample in samples:
        # baseline（开关关）/ treatment（开关开）串行跑：进程内 session/contextvar 天然隔离，
        # 但串行最稳（对齐 quiz-stage3-parallel 的 fork 并发教训），每题两跑串行代价可接受。
        baseline_result = await run_orchestrator(_build_ctx(sample, capability, False))
        treatment_result = await run_orchestrator(_build_ctx(sample, capability, True))

        base_score = await scorer(sample, baseline_result)
        treat_score = await scorer(sample, treatment_result)
        delta = {m: round(treat_score[m] - base_score[m], 4) for m in metrics}
        per_question.append({
            "id": sample.get("id", ""),
            "baseline": base_score,
            "treatment": treat_score,
            "delta": delta,
        })

    summary_base = {m: _mean([pq["baseline"] for pq in per_question], m) for m in metrics}
    summary_treat = {m: _mean([pq["treatment"] for pq in per_question], m) for m in metrics}
    summary_delta = {m: round(summary_treat[m] - summary_base[m], 4) for m in metrics}

    result = {
        "capability": capability,
        "switch": _SWITCH_KEY[capability],
        "n": len(samples),
        "baseline": summary_base,
        "treatment": summary_treat,
        "delta": summary_delta,
        "per_question": per_question,
    }

    out = config.RESULTS_DIR / f"ablation_{capability}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _print_table(result: dict[str, Any]) -> None:
    cap = result["capability"]
    metrics = _METRICS[cap]
    print(f"\n{'=' * 60}\n消融对比：{cap}（开关 {result['switch']}，{result['n']} 题）\n{'=' * 60}")
    print(f"{'指标':<22}{'baseline':>12}{'treatment':>12}{'delta':>12}")
    print("-" * 58)
    for m in metrics:
        b = result["baseline"][m]
        t = result["treatment"][m]
        d = result["delta"][m]
        print(f"{m:<22}{b:>12.4f}{t:>12.4f}{d:>+12.4f}")
    print(f"\n详情见 {config.RESULTS_DIR / f'ablation_{cap}.json'}")


async def _main(capabilities: list[str]) -> None:
    for cap in capabilities:
        result = await _run_capability(cap)
        _print_table(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="能力级消融实验（开关 on/off 对比）")
    parser.add_argument("--capability", default="all", choices=["research", "solve", "all"])
    args = parser.parse_args()
    caps = ["research", "solve"] if args.capability == "all" else [args.capability]
    asyncio.run(_main(caps))


if __name__ == "__main__":
    main()
