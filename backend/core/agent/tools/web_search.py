"""Web 搜索 + 网页抓取工具。

支持多引擎：DuckDuckGo（免费、无需 key）/ Tavily（可选）/ Brave（可选）。
参考 MathClaw 的 web.py 实现，简化为课程 agent 场景。

安全：内置 SSRF 防护（禁止内网地址）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Annotated
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_TIMEOUT = 15
_MAX_FETCH_CHARS = 8000

_PRIVATE_NETS = re.compile(
    r"^(127\.|10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|0\.|localhost|::1|\[::1\])"
)


def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or _PRIVATE_NETS.match(host):
            return False
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


async def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """使用 DuckDuckGo（ddgs 库）搜索。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


async def _search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """使用 Tavily API 搜索（需要 TAVILY_API_KEY）。"""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return await _search_duckduckgo(query, max_results)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": max_results, "api_key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]
    except Exception as e:
        logger.warning("Tavily search failed, falling back to DDG: %s", e)
        return await _search_duckduckgo(query, max_results)


@tool
async def web_search(
    query: Annotated[str, "搜索查询词"],
    max_results: Annotated[int, "最大返回结果数"] = 5,
) -> str:
    """在互联网上搜索信息。用于需要最新资料或课程知识库中没有的内容时。"""
    logger.info("Tool web_search: query=%s", query[:80])
    results = await _search_tavily(query, max_results)
    if not results:
        return json.dumps({"results": [], "message": "未找到相关网页"}, ensure_ascii=False)
    return json.dumps({"results": results}, ensure_ascii=False)


@tool
async def web_fetch(
    url: Annotated[str, "要抓取的网页 URL"],
) -> str:
    """抓取指定网页的文本内容。用于深入阅读搜索结果中的页面。"""
    logger.info("Tool web_fetch: url=%s", url[:120])
    if not _is_safe_url(url):
        return json.dumps({"error": "URL 不安全或不可访问"}, ensure_ascii=False)

    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, max_redirects=5
        ) as client:
            resp = await client.get(url, headers={"User-Agent": _USER_AGENT})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return json.dumps({"error": "非文本内容，无法抓取"}, ensure_ascii=False)
            html_text = resp.text
    except Exception as e:
        return json.dumps({"error": f"抓取失败: {e}"}, ensure_ascii=False)

    text = _extract_text(html_text)
    if len(text) > _MAX_FETCH_CHARS:
        text = text[:_MAX_FETCH_CHARS] + "\n...[内容已截断]"

    return json.dumps({"url": url, "content": text}, ensure_ascii=False)


def _extract_text(html: str) -> str:
    """从 HTML 中提取纯文本（简易版）。"""
    try:
        from readability import Document
        doc = Document(html)
        summary = doc.summary()
        clean = re.sub(r"<[^>]+>", "", summary)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
    except ImportError:
        pass

    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
