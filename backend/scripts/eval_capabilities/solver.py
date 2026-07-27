"""Inspect AI solver：把 orchestrator.handle(ctx) 包成评测对象（零 HTTP）。

@solver 是纯 async 函数，Inspect 官方允许 solver "optionally 调模型 / 其他 elicitation"——
我们直接 async for event in orchestrator.handle(ctx)，不调 Inspect 的模型层（generate 参数
忽略）。聚合 ANSWER/TOKEN→completion、QUIZ→_quiz、TOOL_CALL→_tools、全量→_trace。

run_orchestrator() 单独抽出，便于 mock 测试（不依赖 Inspect TaskState / 真实 orchestrator /
真实 LLM）——mock get_orchestrator().handle 返回假 StreamEvent 流，即可验证聚合逻辑。
"""
from __future__ import annotations

from typing import Any

from inspect_ai.solver import Generate, TaskState, solver

from core.agent.mode_normalize import normalize_mode
from core.context import UnifiedContext
from core.orchestrator import get_orchestrator
from core.stream import StreamEventType

# 内容类事件：拼进 completion（inspect 内置 scorer 从 state.output.completion 读模型输出）
_CONTENT_TYPES = {StreamEventType.ANSWER, StreamEventType.TOKEN}
# 出题类事件：结构化产物，交给 quiz scorer（不拼进文本 completion，否则污染判分）
_QUIZ_TYPES = {StreamEventType.QUIZ, StreamEventType.QUIZ_QUESTION}


def build_context(state: TaskState) -> UnifiedContext:
    """从 Inspect Sample 的 input/metadata 构造 UnifiedContext（对齐 api/chat.py:108 构造范式）。"""
    meta = state.metadata or {}
    return UnifiedContext(
        user_message=state.input_text,
        mode=normalize_mode(meta.get("mode", "chat")),
        course_id=meta.get("course_id", ""),
        rag_mode=meta.get("rag_mode", "naive"),
        conversation_history=meta.get("history", []) or [],
        language=meta.get("language", "zh"),
        # merge sample metadata（含消融开关 research_observer / solve_force_replan），
        # 让开关经 UnifiedContext.metadata 流到对应 pipeline；turn_id 仍空（评测不注册 bus）。
        metadata={**(state.metadata or {}), "turn_id": ""},
    )


async def run_orchestrator(ctx: UnifiedContext) -> dict[str, Any]:
    """直调 orchestrator.handle，聚合事件流。

    返回 {answer, quiz, tools, trace, error}。capability 抛出的异常被 orchestrator 封成
    ERROR 事件（不向上抛出），此处记录到 error 字段、不崩整轮——对齐 RAGAS 判崩容错思路，
    让 scorer 能把"跑挂的 case"和"正常 case"区分开。
    """
    answer: list[str] = []
    quiz: list[dict] = []
    tools: list[str] = []
    trace: list[dict] = []
    error = ""
    async for event in get_orchestrator().handle(ctx):
        t = event.type
        if t in _CONTENT_TYPES:
            answer.append(str(event.payload.get("content", "")))
        elif t in _QUIZ_TYPES:
            quiz.append(dict(event.payload))
        elif t == StreamEventType.TOOL_CALL:
            tools.append(str(event.payload.get("tool") or event.payload.get("name", "")))
        elif t == StreamEventType.ERROR:
            error = str(event.payload.get("content") or event.payload or "")
        trace.append(event.to_dict())
    return {
        "answer": "".join(answer),
        "quiz": quiz,
        "tools": tools,
        "trace": trace,
        "error": error,
    }


@solver
def orchestrator_solver():
    """Inspect solver：跑 orchestrator，结果回填 state.output.completion + metadata。"""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ctx = build_context(state)
        result = await run_orchestrator(ctx)
        if result["error"]:
            state.output.completion = f"[ERROR] {result['error']}"
        else:
            state.output.completion = result["answer"] or "(no answer)"
        # 结构化产物供自定义 scorer 读取（quiz 题集 / solve 状态机轨迹 / 调用过的工具）
        state.metadata["_quiz"] = result["quiz"]
        state.metadata["_tools"] = result["tools"]
        state.metadata["_trace"] = result["trace"]
        return state

    return solve
