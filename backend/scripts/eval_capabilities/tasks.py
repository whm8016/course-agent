"""四能力 Inspect AI task 装配。

每个 @task = dataset(jsonl→Sample) + orchestrator_solver + 对应 scorer + pass^k epochs。
solver 跳过 Inspect 模型层，直调 orchestrator.handle(ctx)，故 model 参数对评测无意义
（judge 走 scorer 内部的 get_model）。
"""
from __future__ import annotations

import json

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from . import config
from .scorer import (
    chat_scorer,
    quiz_quality,
    quiz_validity,
    research_fact,
    research_race,
    solve_answer,
    solve_trajectory,
)
from .solver import orchestrator_solver


def _load_dataset(name: str) -> MemoryDataset:
    """读 datasets/<name>.jsonl → Sample 列表（每行 {id,input,target,metadata}）。

    自己解析而非 inspect 的 json_dataset，完全掌控 metadata 字段映射（mode/course_id 等）。
    """
    path = config.DATASETS_DIR / f"{name}.jsonl"
    samples: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        samples.append(
            Sample(
                id=d.get("id", ""),
                input=d["input"],
                target=d.get("target", ""),
                metadata=d.get("metadata", {}) or {},
            )
        )
    return MemoryDataset(samples)


@task
def chat_task() -> Task:
    return Task(
        dataset=_load_dataset("chat"),
        solver=orchestrator_solver(),
        scorer=[chat_scorer()],
        epochs=Epochs(config.PASS_K["chat"]),
    )


@task
def quiz_task() -> Task:
    return Task(
        dataset=_load_dataset("quiz"),
        solver=orchestrator_solver(),
        scorer=[quiz_validity(), quiz_quality()],
        epochs=Epochs(config.PASS_K["quiz"]),
    )


@task
def solve_task() -> Task:
    return Task(
        dataset=_load_dataset("solve"),
        solver=orchestrator_solver(),
        scorer=[solve_trajectory(), solve_answer()],
        epochs=Epochs(config.PASS_K["solve"]),
    )


@task
def research_task() -> Task:
    return Task(
        dataset=_load_dataset("research"),
        solver=orchestrator_solver(),
        scorer=[research_race(), research_fact()],
        epochs=Epochs(config.PASS_K["research"]),
    )


TASKS = {"chat": chat_task, "quiz": quiz_task, "solve": solve_task, "research": research_task}
