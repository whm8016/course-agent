"""Unit tests for services/search module."""
from __future__ import annotations

import os
import sys

# Ensure backend is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_citation_defaults(self):
        from services.search.types import Citation

        c = Citation(id=1, reference="[1]", url="https://example.com")
        assert c.id == 1
        assert c.type == "web"
        assert c.title == ""

    def test_search_result_defaults(self):
        from services.search.types import SearchResult

        r = SearchResult(title="T", url="https://x.com", snippet="snip")
        assert r.score == 0.0
        assert r.sitelinks == []

    def test_web_search_response_to_dict(self):
        from services.search.types import Citation, SearchResult, WebSearchResponse

        resp = WebSearchResponse(
            query="test",
            answer="ans",
            provider="duckduckgo",
            citations=[Citation(id=1, reference="[1]", url="https://a.com", title="A")],
            search_results=[SearchResult(title="A", url="https://a.com", snippet="s")],
        )
        d = resp.to_dict()
        assert d["query"] == "test"
        assert d["answer"] == "ans"
        assert d["provider"] == "duckduckgo"
        assert len(d["citations"]) == 1
        assert d["citations"][0]["url"] == "https://a.com"
        assert len(d["search_results"]) == 1

    def test_web_search_response_metadata_extra_keys(self):
        from services.search.types import WebSearchResponse

        resp = WebSearchResponse(
            query="q",
            answer="a",
            provider="p",
            metadata={"finish_reason": "stop", "custom_key": "val"},
        )
        d = resp.to_dict()
        assert d["custom_key"] == "val"
        assert "finish_reason" not in d  # excluded from top-level per implementation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        from services.search.config import resolve_search_config
        import services.search.config as sc

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEARCH_PROVIDER", None)
            os.environ.pop("SEARCH_API_KEY", None)
            os.environ.pop("SEARCH_MAX_RESULTS", None)
            # 清 .env 经 settings 固化进模块常量的默认值（嵌套化后 SEARCH__* 会固化），
            # 还原「无任何配置 → duckduckgo」的测试前提。
            with patch.object(sc, "SEARCH_PROVIDER", ""), patch.object(sc, "SEARCH_API_KEY", ""), \
                 patch.object(sc, "SEARCH_PROXY", ""):
                cfg = resolve_search_config()

        assert cfg.provider == "duckduckgo"
        assert cfg.api_key == ""
        assert cfg.max_results == 5
        assert cfg.proxy is None

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("SEARCH_API_KEY", "mykey")  # brave 需要 API key 避免 fallback
        monkeypatch.setenv("SEARCH_MAX_RESULTS", "10")
        monkeypatch.setenv("SEARCH_PROXY", "http://proxy:8080")

        from services.search.config import resolve_search_config

        cfg = resolve_search_config()
        assert cfg.provider == "brave"  # 有 api_key 时不会 fallback
        assert cfg.api_key == "mykey"
        assert cfg.max_results == 10
        assert cfg.proxy == "http://proxy:8080"


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_list_providers_includes_builtin(self):
        from services.search.providers import list_providers

        providers = list_providers()
        assert "duckduckgo" in providers
        assert "brave" in providers
        assert "tavily" in providers

    def test_get_provider_unknown_raises(self):
        from services.search.providers import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_xyz")

    def test_get_provider_duckduckgo(self):
        from services.search.providers import get_provider

        p = get_provider("duckduckgo")
        assert p.name == "duckduckgo"
        assert p.requires_api_key is False

    def test_get_providers_info_structure(self):
        from services.search.providers import get_providers_info

        info = get_providers_info()
        assert isinstance(info, list)
        assert len(info) > 0
        for item in info:
            assert "id" in item
            assert "name" in item
            assert "supports_answer" in item
            assert "requires_api_key" in item


# ---------------------------------------------------------------------------
# DuckDuckGo provider (mock network)
# ---------------------------------------------------------------------------

class TestDuckDuckGoProvider:
    def test_search_returns_response(self):
        from services.search.providers import duckduckgo as dd

        rows = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]
        mock_inst = MagicMock()
        mock_inst.text.return_value = rows

        # provider 内 `from ddgs import DDGS` 函数内导入，patch 源头 ddgs.DDGS 即可生效
        with patch("ddgs.DDGS", return_value=mock_inst):
            resp = dd.DuckDuckGoProvider().search("python tutorial", max_results=2)

        assert resp.provider == "duckduckgo"
        assert resp.query == "python tutorial"
        assert len(resp.search_results) == 2
        assert len(resp.citations) == 2
        assert resp.search_results[0].title == "Result 1"
        assert resp.search_results[0].url == "https://example.com/1"
        assert resp.search_results[0].snippet == "Snippet 1"
        assert resp.citations[0].reference == "[1]"
        # query + max_results 透传给 ddgs.text
        mock_inst.text.assert_called_once_with("python tutorial", max_results=2)

    def test_search_empty_results(self):
        from services.search.providers import duckduckgo as dd

        mock_inst = MagicMock()
        mock_inst.text.return_value = []

        with patch("ddgs.DDGS", return_value=mock_inst):
            resp = dd.DuckDuckGoProvider().search("nothing found")

        assert resp.search_results == []
        assert resp.citations == []
        assert resp.answer == ""

    def test_search_ddgs_exception_returns_empty(self):
        """ddgs.text() 抛异常时 provider 容错返回空结果（不向上抛，避免拖垮调用方）。

        回归：旧实现手抓 html.duckduckgo.com 被反爬（403 / SSL 断连 / 空 202 挑战页）；
        改用 ddgs 库后，库内异常由 provider 捕获、返回空 WebSearchResponse。
        """
        from services.search.providers import duckduckgo as dd

        mock_inst = MagicMock()
        mock_inst.text.side_effect = RuntimeError("network error")

        with patch("ddgs.DDGS", return_value=mock_inst):
            resp = dd.DuckDuckGoProvider().search("circuit analysis", max_results=2)

        assert resp.search_results == []
        assert resp.provider == "duckduckgo"

    def test_is_available(self):
        from services.search.providers.duckduckgo import DuckDuckGoProvider

        p = DuckDuckGoProvider()
        assert p.is_available() is True


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    def _make_response(self, n: int = 3):
        from services.search.types import Citation, SearchResult, WebSearchResponse

        citations = [
            Citation(id=i, reference=f"[{i}]", url=f"https://ex.com/{i}", title=f"T{i}", snippet=f"S{i}")
            for i in range(1, n + 1)
        ]
        search_results = [
            SearchResult(title=f"T{i}", url=f"https://ex.com/{i}", snippet=f"S{i}")
            for i in range(1, n + 1)
        ]
        return WebSearchResponse(
            query="what is AI",
            answer="",
            provider="duckduckgo",
            citations=citations,
            search_results=search_results,
        )

    def test_template_consolidation_adds_answer(self):
        from services.search.consolidation import AnswerConsolidator

        resp = self._make_response()
        consolidator = AnswerConsolidator(use_llm=False)
        result = consolidator.consolidate(resp)
        assert result.answer != ""
        assert result.provider == "duckduckgo"

    def test_template_consolidation_preserves_citations(self):
        from services.search.consolidation import AnswerConsolidator

        resp = self._make_response(n=2)
        consolidator = AnswerConsolidator(use_llm=False)
        result = consolidator.consolidate(resp)
        assert len(result.citations) == 2


# ---------------------------------------------------------------------------
# web_search() function  (no real network)
# ---------------------------------------------------------------------------

class TestWebSearch:
    def test_web_search_duckduckgo(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
        monkeypatch.setenv("SEARCH_ENABLED", "true")

        from services.search import web_search

        rows = [{"title": "T1", "href": "https://a.com", "body": "snip1"}]
        mock_inst = MagicMock()
        mock_inst.text.return_value = rows
        with patch("ddgs.DDGS", return_value=mock_inst):
            result = web_search("hello world", provider="duckduckgo")

        assert result["provider"] == "duckduckgo"
        assert result["query"] == "hello world"
        assert isinstance(result["citations"], list)
        assert isinstance(result["search_results"], list)

    def test_web_search_disabled(self, monkeypatch):
        monkeypatch.setenv("SEARCH_ENABLED", "false")

        from services.search import web_search

        result = web_search("test query", provider="duckduckgo")
        assert result["provider"] == "disabled"
        assert "disabled" in result["answer"].lower()

    def test_web_search_provider_none(self, monkeypatch):
        monkeypatch.setenv("SEARCH_ENABLED", "true")

        from services.search import web_search

        result = web_search("test", provider="none")
        assert result["provider"] == "none"

    def test_web_search_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("SEARCH_ENABLED", "true")

        from services.search import web_search

        with pytest.raises(ValueError, match="Unknown search provider"):
            web_search("test", provider="fakeXYZ123")

    def test_web_search_saves_output(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
        monkeypatch.setenv("SEARCH_ENABLED", "true")

        from services.search import web_search

        rows = [{"title": "T1", "href": "https://a.com", "body": "snip1"}]
        mock_inst = MagicMock()
        mock_inst.text.return_value = rows
        with patch("ddgs.DDGS", return_value=mock_inst):
            result = web_search("save test", provider="duckduckgo", output_dir=str(tmp_path))

        assert "result_file" in result
        import json
        with open(result["result_file"], encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["query"] == "save test"


# ---------------------------------------------------------------------------
# get_current_config()
# ---------------------------------------------------------------------------

class TestGetCurrentConfig:
    def test_returns_dict_with_expected_keys(self, monkeypatch):
        monkeypatch.setenv("SEARCH_PROVIDER", "duckduckgo")
        monkeypatch.setenv("SEARCH_ENABLED", "true")

        from services.search import get_current_config

        cfg = get_current_config()
        assert "enabled" in cfg
        assert "provider" in cfg
        assert "max_results" in cfg
        assert "providers" in cfg
        assert isinstance(cfg["providers"], list)
