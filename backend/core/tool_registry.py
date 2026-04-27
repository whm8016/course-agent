from __future__ import annotations
import asyncio, json, logging
from dataclasses import dataclass, field
from typing import Any
from core.tool_protocol import ToolResult

logger = logging.getLogger(__name__)


# ── RAG Tool ────────────────────────────────────────────────────────────────

async def _execute_rag(course_id: str, query: str, **kwargs) -> ToolResult:
    """调用 LightRAG 检索，返回 ToolResult。"""
    from core.lightrag_engine import _get_instance, _build_query_param, LIGHTRAG_QUERY_MODE
    try:
        rag = await _get_instance(course_id)
        mode = kwargs.get("mode") or LIGHTRAG_QUERY_MODE or "mix"
        param = _build_query_param(mode, [], only_need_context=True)
        if hasattr(param, "stream"):
            param.stream = False
        raw = await rag.aquery(query, param=param)
        content = raw.strip() if isinstance(raw, str) else str(raw or "").strip()
        if len(content) > 6000:
            content = content[:6000] + "\n...(truncated)"
        _preview = (content[:800] + "…") if len(content) > 800 else content
        logger.info(
            "tool_registry [rag] course=%s mode=%s query_chars=%d retrieved_chars=%d empty=%s\n"
            "--- RAG 检索结果预览（前 800 字）---\n%s\n--- end preview ---",
            course_id,
            mode,
            len(query),
            len(content),
            not bool(content),
            _preview or "（空）",
        )
        if content:
            logger.debug("tool_registry [rag] full retrieved_context:\n%s", content)
        return ToolResult(content=content, sources=[{"type": "rag", "query": query}])
    except Exception as e:
        logger.exception("rag tool failed")
        return ToolResult(content=f"（知识库检索失败：{e}）", success=False)


# ── WebSearch Tool ─────────────────────────────────────────────────────────

async def _execute_web_search(query: str, **kwargs) -> ToolResult:
    """调用 DuckDuckGo 免费搜索，返回 ToolResult。"""
    try:
        from duckduckgo_search import DDGS
        results = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=5))
        )
        if not results:
            return ToolResult(content="（未找到相关网页）", sources=[])
        lines = []
        sources = []
        for i, r in enumerate(results):
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"[{i+1}] {title}\n{url}\n{body}")
            sources.append({"type": "web", "title": title, "url": url})
        content = "\n\n".join(lines)
        return ToolResult(content=content, sources=sources)
    except ImportError:
        return ToolResult(content="（未安装 duckduckgo-search，请 pip install duckduckgo-search）", success=False)
    except Exception as e:
        logger.exception("web_search tool failed")
        return ToolResult(content=f"（网络搜索失败：{e}）", success=False)


# ── Registry ───────────────────────────────────────────────────────────────

TOOLS_OPENAI_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag",
            "description": "从课程知识库中检索与问题最相关的内容片段，适合回答课程知识点问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词，提炼自用户问题的核心关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "搜索互联网获取最新信息，适合知识库里没有的最新动态、时事或补充资料。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询词"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_LIST_TEXT = """\
- rag(query): 从课程知识库检索相关内容，适合回答课程知识点问题
- web_search(query): 搜索互联网，适合知识库里没有的最新动态或补充资料"""


async def execute_tool(name: str, course_id: str, **kwargs) -> ToolResult:
    if name == "rag":
        return await _execute_rag(course_id=course_id, **kwargs)
    if name == "web_search":
        return await _execute_web_search(**kwargs)
    return ToolResult(content=f"（未知工具：{name}）", success=False)