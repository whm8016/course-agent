"""LangGraph multi-agent orchestrator.

Implements a Director Graph:
    user message -> router -> (teach | quiz | summarize | vision) -> response

Each step yields structured events so the API layer can stream them as SSE.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
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
from core.llm import client as async_openai_client

logger = logging.getLogger(__name__)

_LLM_SEMAPHORE = asyncio.Semaphore(int(__import__("os").getenv("MAX_CONCURRENT_LLM", "25")))


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

async def router_node(state: AgentState) -> dict:
    """Classify user intent using LLM (async HTTP; does not block the event loop)."""
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

    async with _LLM_SEMAPHORE:
        completion = await async_openai_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": state["message"]},
            ],
            temperature=0.7,
            max_tokens=256,
        )
    raw = (completion.choices[0].message.content or "").strip()

    intent = "teach"
    try:
        parsed = json.loads(raw)
        if parsed.get("intent") in ("teach", "quiz", "summarize", "vision", "off_topic"):
            intent = parsed["intent"]
    except (json.JSONDecodeError, TypeError):
        pass

    intent_labels = {
        "teach": "知识问答模式",
        "quiz": "测验出题模式",
        "summarize": "学习总结模式",
        "off_topic": "话题无关",
    }
    label = intent_labels.get(intent, intent)
    logger.info(
        "Router: intent=%s (%s) question=「%s」",
        intent, label, state["message"][:80],
    )

    return {"intent": intent, "events": [
        {"type": "thinking", "content": f"意图识别完成：{label}"}
    ]}


async def teach_node(state: AgentState) -> dict:
    """RAG-augmented teaching response (non-blocking)."""
    events: list[dict] = []

    events.append({"type": "thinking", "content": "正在从知识库检索相关内容..."})
    events.append({"type": "tool_call", "tool": "search_knowledge",
                    "input": {"query": state["message"], "course_id": state["course_id"]}})

    chunks = await asyncio.to_thread(retrieve, state["course_id"], state["message"], top_k=4)
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

    async with _LLM_SEMAPHORE:
        llm = _get_llm()
        response = await asyncio.to_thread(llm.invoke, messages)
    events.append({"type": "answer", "content": response.content})

    return {"events": events}


async def quiz_node(state: AgentState) -> dict:
    """Generate quiz questions based on course knowledge (non-blocking)."""
    events: list[dict] = []

    events.append({"type": "thinking", "content": "正在检索知识点并生成测验题..."})
    events.append({"type": "tool_call", "tool": "generate_quiz",
                    "input": {"topic": state["message"], "course_id": state["course_id"]}})

    chunks = await asyncio.to_thread(retrieve, state["course_id"], state["message"], top_k=3)
    events.append({"type": "tool_result", "tool": "search_knowledge", "chunks": chunks})

    context = "\n\n".join(c["content"] for c in chunks) if chunks else "无可用知识内容"

    prompt = f"""{QUIZ_PROMPT}

课程知识内容：
{context}

学生的要求：{state["message"]}
请生成 3 道测验题。"""

    async with _LLM_SEMAPHORE:
        llm = _get_llm()
        result = await asyncio.to_thread(llm.invoke, [SystemMessage(content=prompt)])

    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0]
        quiz_data = json.loads(content)
        events.append({"type": "quiz", "quiz": quiz_data})
    except (json.JSONDecodeError, IndexError):
        events.append({"type": "answer", "content": result.content})

    return {"events": events}


async def summarize_node(state: AgentState) -> dict:
    """Summarize conversation and learning progress (non-blocking)."""
    events: list[dict] = []
    events.append({"type": "thinking", "content": "正在分析对话历史，生成学习小结..."})

    history_text = ""
    for msg in state["history"][-14:]:
        role_label = "学生" if msg["role"] == "user" else "助教"
        history_text += f"{role_label}: {msg['content']}\n\n"

    if not history_text.strip():
        events.append({"type": "answer", "content": "当前还没有足够的对话内容来生成总结。请先提几个问题吧！"})
        return {"events": events}

    prompt = f"""{SUMMARY_PROMPT}

以下是对话历史：
{history_text}

请生成学习小结。"""

    async with _LLM_SEMAPHORE:
        llm = _get_llm()
        result = await asyncio.to_thread(llm.invoke, [SystemMessage(content=prompt)])
    events.append({"type": "answer", "content": result.content})

    return {"events": events}


async def vision_node(state: AgentState) -> dict:
    """Analyze uploaded image with vision model (non-blocking)."""
    events: list[dict] = []
    events.append({"type": "thinking", "content": "正在分析图片内容..."})
    events.append({"type": "tool_call", "tool": "analyze_image",
                    "input": {"image_path": state["image_path"]}})

    system_prompt = get_course_prompt(state["course_id"])
    data_url = await asyncio.to_thread(_image_to_data_url, state["image_path"])

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

    async with _LLM_SEMAPHORE:
        llm = _get_llm(model=VISION_MODEL)
        result = await asyncio.to_thread(llm.invoke, messages)
    events.append({"type": "answer", "content": result.content})

    return {"events": events}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

OFF_TOPIC_REPLY = (
    "😊 我是你的课程助教，专注于课程相关的学习问题。"
    "你这个问题似乎和课程内容没有太大关系，我就不班门弄斧啦～\n\n"
    "有任何课程知识上的疑问，随时向我提问！"
)


async def off_topic_node(state: AgentState) -> dict:
    """Return a polite refusal for off-topic questions."""
    return {"events": [{"type": "answer", "content": OFF_TOPIC_REPLY}]}

def _route_intent(state: AgentState) -> str:
    return state["intent"]


def build_graph() -> StateGraph:
    # router_node is async; run the compiled graph with ainvoke(), not invoke().
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("teach", teach_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("vision", vision_node)
    graph.add_node("off_topic", off_topic_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", _route_intent, {
        "teach": "teach",
        "quiz": "quiz",
        "summarize": "summarize",
        "vision": "vision",
        "off_topic": "off_topic",
    })

    graph.add_edge("teach", END)
    graph.add_edge("quiz", END)
    graph.add_edge("summarize", END)
    graph.add_edge("vision", END)
    graph.add_edge("off_topic", END)

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
    t0 = time.perf_counter()
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

    logger.info("  RAG retrieve start course=%s query=「%s」", state["course_id"], state["message"][:60])
    chunks = await asyncio.to_thread(retrieve, state["course_id"], state["message"], top_k=4)
    logger.info(
        "  RAG retrieve done chunks=%d elapsed=%dms",
        len(chunks), int((time.perf_counter() - t0) * 1000),
    )
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
    logger.info("  LLM stream start mode=%s", teach_mode)
    llm_t0 = time.perf_counter()
    await _LLM_SEMAPHORE.acquire()
    try:
        async for token in chat_stream(
            system_prompt=system_prompt,
            history=state["history"][-10:],
            user_message=state["message"],
            image_path=None,
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    finally:
        _LLM_SEMAPHORE.release()

    full_answer = "".join(answer_parts)
    logger.info(
        "  LLM stream done answer_chars=%d llm_time=%dms",
        len(full_answer), int((time.perf_counter() - llm_t0) * 1000),
    )
    yield {"type": "answer", "content": full_answer}


async def _stream_summarize_events(state: AgentState) -> AsyncGenerator[dict, None]:
    yield {"type": "thinking", "content": "正在分析对话历史，生成学习小结..."}

    history_text = ""
    for msg in state["history"][-14:]:
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
    await _LLM_SEMAPHORE.acquire()
    try:
        async for token in chat_stream(
            system_prompt=system_prompt,
            history=[],
            user_message=user_message,
            image_path=None,
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    finally:
        _LLM_SEMAPHORE.release()

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
    await _LLM_SEMAPHORE.acquire()
    try:
        async for token in chat_stream(
            system_prompt=system_prompt,
            history=state["history"][-10:],
            user_message=user_message,
            image_path=state["image_path"],
        ):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
    finally:
        _LLM_SEMAPHORE.release()

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
    t0 = time.perf_counter()

    def _ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    normalized_mode = normalize_mode(mode)
    logger.info(
        "━━ Pipeline START course=%s mode=%s history_len=%d has_image=%s question=「%s」",
        course_id, normalized_mode, len(history), bool(image_path), message[:100],
    )

    state: AgentState = {
        "course_id": course_id,
        "message": message,
        "history": history,
        "image_path": image_path,
        "mode": normalized_mode,
        "memory_context": memory_context,
        "intent": "",
        "events": [],
    }

    route_result = await router_node(state)
    intent = route_result.get("intent", "teach")
    router_events = route_result.get("events", [])
    tools_used: set[str] = set()
    logger.info("━━ Router done intent=%s t=%dms", intent, _ms())

    for event in router_events:
        yield event

    if intent == "teach":
        logger.info("━━ ▶ teach node start t=%dms", _ms())
        async for event in _stream_teach_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
        logger.info("━━ ◀ teach node done t=%dms", _ms())
    elif intent == "summarize":
        logger.info("━━ ▶ summarize node start t=%dms", _ms())
        async for event in _stream_summarize_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
        logger.info("━━ ◀ summarize node done t=%dms", _ms())
    elif intent == "vision":
        logger.info("━━ ▶ vision node start t=%dms", _ms())
        async for event in _stream_vision_events(state):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
        logger.info("━━ ◀ vision node done t=%dms", _ms())
    elif intent == "off_topic":
        logger.info("━━ ▶ off_topic node start t=%dms", _ms())
        yield {"type": "answer", "content": OFF_TOPIC_REPLY}
        logger.info("━━ ◀ off_topic node done t=%dms", _ms())
    else:
        logger.info("━━ ▶ quiz node start t=%dms", _ms())
        result = await quiz_node(state)
        for event in result.get("events", []):
            if event.get("type") == "tool_call" and event.get("tool"):
                tools_used.add(event["tool"])
            yield event
        logger.info("━━ ◀ quiz node done t=%dms", _ms())

    logger.info(
        "━━ Pipeline END intent=%s mode=%s tools=%s total_time=%dms question=「%s」",
        intent, state["mode"], list(tools_used), _ms(), message[:60],
    )

    yield {
        "type": "done",
        "metadata": {
            "intent": intent,
            "mode": state["mode"],
            "tools_used": list(tools_used),
        },
    }
