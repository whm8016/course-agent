"""LangGraph multi-agent orchestrator.

Implements a Director Graph:
    user message -> router -> (teach | quiz | summarize | vision) -> response

Each step yields structured events so the API layer can stream them as SSE.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, TEXT_MODEL, VISION_MODEL
from core.prompts import (
    QUIZ_PROMPT,
    ROUTER_PROMPT,
    SUMMARY_PROMPT,
    get_course_prompt,
)
from core.rag import retrieve
from core.llm import _image_to_data_url
from core.llm import chat_stream

logger = logging.getLogger(__name__)


def _merge_events(existing: list[dict], new: list[dict]) -> list[dict]:
    """Reducer: append new events to the existing list."""
    return existing + new


class AgentState(TypedDict):
    course_id: str
    message: str
    history: list[dict]
    image_path: str | None
    mode: str
    memory_context: str
    intent: str
    events: Annotated[list[dict], _merge_events]


def _get_llm(model: str | None = None) -> ChatOpenAI:
    # Use streaming=False with .invoke() — streaming=True here can destabilize
    # the HTTP worker and show up as ECONNRESET / 502 behind Vite proxy.
    return ChatOpenAI(
        model=model or TEXT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.7,
        max_tokens=2048,
        streaming=False,
    )


def normalize_mode(mode: str | None) -> str:
    allowed = {"chat", "deep_solve", "quiz", "research", "vision", "summarize"}
    if not mode:
        return "chat"
    normalized = mode.strip().lower()
    return normalized if normalized in allowed else "chat"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """Classify user intent using LLM."""
    forced_mode = normalize_mode(state.get("mode"))
    forced_intent_map = {
        "deep_solve": "teach",
        "research": "teach",
        "quiz": "quiz",
        "vision": "vision",
        "summarize": "summarize",
    }
    if forced_mode in forced_intent_map:
        forced_intent = forced_intent_map[forced_mode]
        forced_labels = {
            "chat": "通用问答模式",
            "deep_solve": "深度解题模式",
            "research": "深度研究模式",
            "quiz": "测验出题模式",
            "vision": "视觉分析模式",
            "summarize": "学习总结模式",
        }
        return {
            "intent": forced_intent,
            "events": [{"type": "thinking", "content": f"模式已切换：{forced_labels[forced_mode]}"}],
        }

    if state.get("image_path"):
        return {"intent": "vision", "events": [
            {"type": "thinking", "content": "检测到图片上传，启动视觉分析..."}
        ]}

    llm = _get_llm()
    result = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=state["message"]),
    ])

    intent = "teach"
    try:
        parsed = json.loads(result.content)
        if parsed.get("intent") in ("teach", "quiz", "summarize", "vision"):
            intent = parsed["intent"]
    except (json.JSONDecodeError, AttributeError):
        pass

    intent_labels = {
        "teach": "知识问答模式",
        "quiz": "测验出题模式",
        "summarize": "学习总结模式",
    }
    label = intent_labels.get(intent, intent)
    logger.info("Router classified intent=%s for msg=%s", intent, state["message"][:50])

    return {"intent": intent, "events": [
        {"type": "thinking", "content": f"意图识别完成：{label}"}
    ]}


def teach_node(state: AgentState) -> dict:
    """RAG-augmented teaching response."""
    events: list[dict] = []

    events.append({"type": "thinking", "content": "正在从知识库检索相关内容..."})
    events.append({"type": "tool_call", "tool": "search_knowledge",
                    "input": {"query": state["message"], "course_id": state["course_id"]}})

    chunks = retrieve(state["course_id"], state["message"], top_k=4)
    events.append({"type": "tool_result", "tool": "search_knowledge", "chunks": chunks})

    system_prompt = get_course_prompt(state["course_id"])
    if chunks:
        context = "\n\n---\n\n".join(c["content"] for c in chunks)
        system_prompt += (
            f"\n\n【参考资料】以下是从课程知识库中检索到的相关内容，请结合这些资料回答：\n\n{context}\n\n---\n"
        )

    messages = [SystemMessage(content=system_prompt)]
    for msg in state["history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=state["message"]))

    llm = _get_llm()
    response = llm.invoke(messages)
    events.append({"type": "answer", "content": response.content})

    return {"events": events}


def quiz_node(state: AgentState) -> dict:
    """Generate quiz questions based on course knowledge."""
    events: list[dict] = []

    events.append({"type": "thinking", "content": "正在检索知识点并生成测验题..."})
    events.append({"type": "tool_call", "tool": "generate_quiz",
                    "input": {"topic": state["message"], "course_id": state["course_id"]}})

    chunks = retrieve(state["course_id"], state["message"], top_k=3)
    events.append({"type": "tool_result", "tool": "search_knowledge", "chunks": chunks})

    context = "\n\n".join(c["content"] for c in chunks) if chunks else "无可用知识内容"

    prompt = f"""{QUIZ_PROMPT}

课程知识内容：
{context}

学生的要求：{state["message"]}
请生成 3 道测验题。"""

    llm = _get_llm()
    result = llm.invoke([SystemMessage(content=prompt)])

    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        quiz_data = json.loads(content)
        events.append({"type": "quiz", "quiz": quiz_data})
    except (json.JSONDecodeError, IndexError):
        events.append({"type": "answer", "content": result.content})

    return {"events": events}


def summarize_node(state: AgentState) -> dict:
    """Summarize conversation and learning progress."""
    events: list[dict] = []
    events.append({"type": "thinking", "content": "正在分析对话历史，生成学习小结..."})

    history_text = ""
    for msg in state["history"][-20:]:
        role_label = "学生" if msg["role"] == "user" else "助教"
        history_text += f"{role_label}: {msg['content']}\n\n"

    if not history_text.strip():
        events.append({"type": "answer", "content": "当前还没有足够的对话内容来生成总结。请先提几个问题吧！"})
        return {"events": events}

    prompt = f"""{SUMMARY_PROMPT}

以下是对话历史：
{history_text}

请生成学习小结。"""

    llm = _get_llm()
    result = llm.invoke([SystemMessage(content=prompt)])
    events.append({"type": "answer", "content": result.content})

    return {"events": events}


def vision_node(state: AgentState) -> dict:
    """Analyze uploaded image with vision model."""
    events: list[dict] = []
    events.append({"type": "thinking", "content": "正在分析图片内容..."})
    events.append({"type": "tool_call", "tool": "analyze_image",
                    "input": {"image_path": state["image_path"]}})

    system_prompt = get_course_prompt(state["course_id"])
    data_url = _image_to_data_url(state["image_path"])

    messages = [SystemMessage(content=system_prompt)]
    for msg in state["history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": state["message"] or "请赏析这张邮票图片。"},
    ]))

    llm = _get_llm(model=VISION_MODEL)
    result = llm.invoke(messages)
    events.append({"type": "answer", "content": result.content})

    return {"events": events}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def _route_intent(state: AgentState) -> str:
    return state["intent"]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("teach", teach_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("vision", vision_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", _route_intent, {
        "teach": "teach",
        "quiz": "quiz",
        "summarize": "summarize",
        "vision": "vision",
    })

    graph.add_edge("teach", END)
    graph.add_edge("quiz", END)
    graph.add_edge("summarize", END)
    graph.add_edge("vision", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


async def run_agent(
    course_id: str,
    message: str,
    history: list[dict],
    image_path: str | None = None,
    mode: str = "chat",
    memory_context: str = "",
) -> list[dict]:
    """Run the full agent pipeline and collect all events."""
    events: list[dict] = []
    async for event in run_agent_stream(
        course_id,
        message,
        history,
        image_path,
        mode=mode,
        memory_context=memory_context,
    ):
        events.append(event)
    return events


async def _stream_teach_events(state: AgentState) -> AsyncGenerator[dict, None]:
    teach_mode = normalize_mode(state.get("mode"))
    thinking_labels = {
        "chat": "正在从知识库检索相关内容...",
        "deep_solve": "正在进行深度解题：先检索依据，再分步推导...",
        "research": "正在进行主题研究：先检索证据，再组织结构化结论...",
    }
    yield {"type": "thinking", "content": thinking_labels.get(teach_mode, thinking_labels["chat"])}
    yield {
        "type": "tool_call",
        "tool": "search_knowledge",
        "input": {"query": state["message"], "course_id": state["course_id"]},
    }

    chunks = retrieve(state["course_id"], state["message"], top_k=4)
    yield {"type": "tool_result", "tool": "search_knowledge", "chunks": chunks}

    system_prompt = get_course_prompt(state["course_id"])
    if state.get("memory_context"):
        system_prompt += f"\n\n{state['memory_context']}"
    if chunks:
        context = "\n\n---\n\n".join(c["content"] for c in chunks)
        system_prompt += (
            f"\n\n【参考资料】以下是从课程知识库中检索到的相关内容，请结合这些资料回答：\n\n{context}\n\n---\n"
        )
    if teach_mode == "deep_solve":
        system_prompt += "\n\n请采用“题目理解 -> 已知条件 -> 分步推导 -> 最终结论 -> 易错点”的结构作答。"
    elif teach_mode == "research":
        system_prompt += "\n\n请采用“核心观点 -> 关键证据 -> 对比分析 -> 实践建议”的结构输出。"

    answer_parts: list[str] = []
    async for token in chat_stream(
        system_prompt=system_prompt,
        history=state["history"],
        user_message=state["message"],
        image_path=None,
    ):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    yield {"type": "answer", "content": "".join(answer_parts)}


async def _stream_summarize_events(state: AgentState) -> AsyncGenerator[dict, None]:
    yield {"type": "thinking", "content": "正在分析对话历史，生成学习小结..."}

    history_text = ""
    for msg in state["history"][-20:]:
        role_label = "学生" if msg["role"] == "user" else "助教"
        history_text += f"{role_label}: {msg['content']}\n\n"

    if not history_text.strip():
        yield {"type": "answer", "content": "当前还没有足够的对话内容来生成总结。请先提几个问题吧！"}
        return

    system_prompt = f"""{SUMMARY_PROMPT}

以下是对话历史：
{history_text}
"""
    if state.get("memory_context"):
        system_prompt += f"\n\n{state['memory_context']}"
    user_message = "请生成学习小结。"
    answer_parts: list[str] = []
    async for token in chat_stream(
        system_prompt=system_prompt,
        history=[],
        user_message=user_message,
        image_path=None,
    ):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    yield {"type": "answer", "content": "".join(answer_parts)}


async def _stream_vision_events(state: AgentState) -> AsyncGenerator[dict, None]:
    yield {"type": "thinking", "content": "正在分析图片内容..."}
    yield {
        "type": "tool_call",
        "tool": "analyze_image",
        "input": {"image_path": state["image_path"]},
    }

    system_prompt = get_course_prompt(state["course_id"])
    if state.get("memory_context"):
        system_prompt += f"\n\n{state['memory_context']}"
    user_message = state["message"] or "请赏析这张邮票图片。"
    answer_parts: list[str] = []
    async for token in chat_stream(
        system_prompt=system_prompt,
        history=state["history"],
        user_message=user_message,
        image_path=state["image_path"],
    ):
        answer_parts.append(token)
        yield {"type": "token", "content": token}

    yield {"type": "answer", "content": "".join(answer_parts)}


async def run_agent_stream(
    course_id: str,
    message: str,
    history: list[dict],
    image_path: str | None = None,
    mode: str = "chat",
    memory_context: str = "",
) -> AsyncGenerator[dict, None]:
    """Run the full agent pipeline and stream structured events."""
    state: AgentState = {
        "course_id": course_id,
        "message": message,
        "history": history,
        "image_path": image_path,
        "mode": normalize_mode(mode),
        "memory_context": memory_context,
        "intent": "",
        "events": [],
    }

    route_result = router_node(state)
    intent = route_result.get("intent", "teach")
    router_events = route_result.get("events", [])
    tools_used: set[str] = set()

    for event in router_events:
        yield event

    if intent == "teach":
        async for event in _stream_teach_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
    elif intent == "summarize":
        async for event in _stream_summarize_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
    elif intent == "vision":
        async for event in _stream_vision_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
    else:
        result = quiz_node(state)
        for event in result.get("events", []):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event

    yield {
        "type": "done",
        "metadata": {
            "intent": intent,
            "mode": state["mode"],
            "tools_used": list(tools_used),
        },
    }
