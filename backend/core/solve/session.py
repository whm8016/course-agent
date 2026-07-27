"""SolveSession — 单 turn 内存状态机。

解题 turn 是一次性的：session 只在该 turn 存活，由 solve pipeline 通过 contextvar
注入的 session_id 索引。它持有模型编写的 plan、逐步完成门控、replan 预算——
这是 chat loop 驱动的"确定性脊柱"：智能在 loop 出口（模型自己规划求解），
确定性（commit plan、不跳步、bounded replan）是 engine state 通过工具读写。

存储是有界的进程内 OrderedDict：solve turn 在单进程内运行，session 小且短命，
上界防止长期运行的服务器泄漏。并发 turn 用不同 id，互不竞争。

session_id 注入路径：dispatch_tool_calls 不收 context（签名只有 course_id/enabled_tools/
stream），故用 contextvar——solve pipeline 调 run_agent_loop 前 set，solve 工具
execute 时 get。这是 tool_calls 版的"augment_kwargs"等价物。
"""
from __future__ import annotations

import contextvars
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MAX_REPLANS = 2
_MAX_STEPS = 12


@dataclass
class SolveStep:
    """一个 plan 步骤。模型调 solve_finish_step 时 done 翻转。"""

    id: str
    goal: str
    done: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "goal": self.goal, "done": self.done}


@dataclass
class SolveSession:
    """一个 solve turn 的 plan + 进度 + replan 预算。"""

    session_id: str
    analysis: str = ""
    steps: list[SolveStep] = field(default_factory=list)
    replans: int = 0
    max_replans: int = DEFAULT_MAX_REPLANS
    # force replan 信号（消融开关默认关）：finish_step 连续未命中有效步骤时计数累加；
    # session.force_replan_gate 开 + 计数达阈值 → tool_registry 返回里追加"请 replan"强制提示。
    # pipeline.run 按 context.metadata["solve_force_replan"] 设置开关；默认 False 行为零变化。
    consecutive_finish_failures: int = 0
    force_replan_gate: bool = False

    def set_plan(self, analysis: str, steps: list[tuple[str, str]]) -> None:
        self.analysis = analysis
        self.steps = [SolveStep(id=sid, goal=goal) for sid, goal in steps][:_MAX_STEPS]

    def replan(self, analysis: str, steps: list[tuple[str, str]]) -> bool:
        """替换 plan，replan 计数+1。预算用尽返回 False（plan 不变）。"""
        if self.replans >= self.max_replans:
            return False
        self.replans += 1
        self.set_plan(analysis, steps)
        return True

    def mark_done(self, step_id: str, summary: str) -> SolveStep | None:
        for step in self.steps:
            if step.id == step_id:
                step.done = True
                step.summary = summary.strip()
                # 成功推进→连续失败归零（与 force replan 门的失败计数配对）
                self.consecutive_finish_failures = 0
                return step
        # 未知 step_id（模型传错 / 计划不适用）→连续失败+1，达阈值时 finish_step 强制提示 replan
        self.consecutive_finish_failures += 1
        return None

    def next_step(self) -> SolveStep | None:
        return next((step for step in self.steps if not step.done), None)

    def map(self) -> list[dict[str, object]]:
        return [step.to_dict() for step in self.steps]

    def all_done(self) -> bool:
        return bool(self.steps) and all(step.done for step in self.steps)

    def reset(self) -> None:
        """清空 plan / 进度 / replan 计数，回到「未解题」初态（max_replans 保留）。

        M-7：solve pipeline 入口（每 turn 一次）调用，防止 sid 撞车时读到上一轮的 stale
        plan——session 存在进程内 OrderedDict，key 由 turn_id/message_id 解析而来，若同
        turn 重试 / message_id 复用 / LRU 驱逐后同 sid 再入，get_session 会复用旧 session。
        reset 不能放进 get_session（同 turn 内 solve 工具会多次读它读写状态，那里重置会清掉
        刚写的 plan），只能由 pipeline 入口每 turn 调一次。
        """
        self.analysis = ""
        self.steps = []
        self.replans = 0
        self.consecutive_finish_failures = 0
        self.force_replan_gate = False  # 回初态（pipeline.run 会按开关重设）


_SESSIONS: "OrderedDict[str, SolveSession]" = OrderedDict()
_MAX_SESSIONS = 256


def get_session(session_id: str) -> SolveSession:
    """取（或惰性创建）turn 的 session，超过上限驱逐最旧。"""
    sid = (session_id or "").strip() or "default"
    session = _SESSIONS.get(sid)
    if session is None:
        session = SolveSession(session_id=sid)
        _SESSIONS[sid] = session
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
    return session


def parse_steps(raw_steps: Any) -> list[tuple[str, str]]:
    """把模型编写的步骤列表校验为 (id, goal) 对。id 服务端生成（S1/S2…），模型不控制存储键。"""
    if not isinstance(raw_steps, list):
        return []
    steps: list[tuple[str, str]] = []
    for raw in raw_steps:
        if isinstance(raw, dict):
            goal = str(raw.get("goal") or "").strip()
        else:
            goal = str(raw or "").strip()
        if not goal:
            continue
        steps.append((f"S{len(steps) + 1}", goal))
    return steps


# ── 当前 turn 的 solve session id（contextvar，绕过不收 context 的 dispatch）──
_current_solve_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "solve_session_id", default=""
)


def set_current_solve_session(sid: str) -> contextvars.Token[str]:
    return _current_solve_session_id.set((sid or "").strip())


def reset_current_solve_session(token: contextvars.Token[str]) -> None:
    try:
        _current_solve_session_id.reset(token)
    except (LookupError, ValueError):
        pass


def current_solve_session_id() -> str:
    return _current_solve_session_id.get()


__all__ = [
    "DEFAULT_MAX_REPLANS",
    "SolveSession",
    "SolveStep",
    "get_session",
    "parse_steps",
    "set_current_solve_session",
    "reset_current_solve_session",
    "current_solve_session_id",
]
