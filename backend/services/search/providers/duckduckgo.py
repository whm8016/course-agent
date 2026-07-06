"""DuckDuckGo search provider — 使用 ``ddgs`` 库（>=9.9.1）。

``ddgs`` 9.x 是 ``duckduckgo-search`` 的重命名修复版：
- 旧 ``duckduckgo-search`` 8.x 在源码里硬编码 ``backends=["bing"]``（国内不可用）；
- 绕过库手抓 ``html.duckduckgo.com`` 已被反爬（403 / SSL 断连 / 空 202 挑战页）。

``ddgs`` 库自带反爬处理（UA 轮换、TLS、endpoint、重试），经 Clash 代理稳定可搜。
对标 （同样用 ``from ddgs import DDGS`` + ``ddgs.text()``）。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..base import BaseSearchProvider
from ..types import Citation, SearchResult, WebSearchResponse
from . import register_provider

_logger = logging.getLogger(__name__)


@register_provider("duckduckgo")
class DuckDuckGoProvider(BaseSearchProvider):
    """DuckDuckGo via ``ddgs`` 库（免 key，国内需配代理）。"""

    display_name = "DuckDuckGo"
    description = "DuckDuckGo search (no API key; needs proxy in CN)"
    supports_answer = False
    requires_api_key = False
    API_KEY_ENV_VARS = ()

    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: int = 20,
        **kwargs: Any,
    ) -> WebSearchResponse:
        from ddgs import DDGS

        count = max(1, min(int(max_results), 10))
        ddgs = DDGS(proxy=self.proxy, timeout=timeout)
        try:
            rows = list(ddgs.text(query, max_results=count) or [])
        except Exception as exc:
            _logger.warning("[duckduckgo] ddgs.text failed for %r: %s", query, exc)
            rows = []

        citations: list[Citation] = []
        search_results: list[SearchResult] = []
        for idx, row in enumerate(rows, 1):
            title = str(row.get("title", ""))
            # ddgs.text 返回 href/body；兼容 url/snippet 字段名
            url = str(row.get("href") or row.get("url") or "")
            snippet = str(row.get("body") or row.get("snippet") or "")
            search_results.append(
                SearchResult(title=title, url=url, snippet=snippet, source="DuckDuckGo")
            )
            citations.append(
                Citation(
                    id=idx,
                    reference=f"[{idx}]",
                    url=url,
                    title=title,
                    snippet=snippet,
                    source="DuckDuckGo",
                )
            )
        return WebSearchResponse(
            query=query,
            answer="",
            provider="duckduckgo",
            timestamp=datetime.now().isoformat(),
            model="duckduckgo",
            citations=citations,
            search_results=search_results,
            metadata={"finish_reason": "stop"},
        )
