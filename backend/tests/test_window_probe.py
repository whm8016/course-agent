"""window_probe 测试：payload 字段扫描/别名匹配 + HTTP 探测(mock) + 缓存读写/TTL。

对标 DeepTutor ``context_window_detection`` 的探测逻辑；缓存用 tmp_path 隔离，绝不污染
真实 data/context_window_cache.json。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from core.agentic import window_probe
from core.agentic.window_probe import (
    _coerce_positive_int,
    _extract_context_window_from_payload,
    _model_aliases,
    _recursive_context_window,
    detect_context_window,
    read_probe_cache,
    write_probe_cache,
)

_BASE = "https://api.example.com/v1"
_KEY = "sk-test"


# ---------------------------------------------------------------------------
# 纯函数：_coerce_positive_int
# ---------------------------------------------------------------------------
class TestCoercePositiveInt:
    def test_int(self):
        assert _coerce_positive_int(64000) == 64000

    def test_float_whole(self):
        assert _coerce_positive_int(64000.0) == 64000

    def test_float_nonwhole_is_none(self):
        assert _coerce_positive_int(64000.5) is None

    def test_str_numeric(self):
        assert _coerce_positive_int("64000") == 64000

    def test_str_with_spaces(self):
        assert _coerce_positive_int(" 64000 ") == 64000

    def test_bool_excluded(self):
        # bool 是 int 子类，必须先排除（True 不能被当成窗口=1）
        assert _coerce_positive_int(True) is None

    def test_negative_is_none(self):
        assert _coerce_positive_int(-5) is None

    def test_zero_is_none(self):
        assert _coerce_positive_int(0) is None

    def test_garbage_str_is_none(self):
        assert _coerce_positive_int("abc") is None

    def test_none_is_none(self):
        assert _coerce_positive_int(None) is None


# ---------------------------------------------------------------------------
# 纯函数：_model_aliases
# ---------------------------------------------------------------------------
class TestModelAliases:
    def test_plain(self):
        assert _model_aliases("qwen-plus") == {"qwen-plus"}

    def test_slash_split(self):
        assert _model_aliases("openai/gpt-4o") == {"openai/gpt-4o", "gpt-4o"}

    def test_colon_split(self):
        assert _model_aliases("model:v2") == {"model:v2", "v2"}

    def test_empty(self):
        assert _model_aliases("") == set()

    def test_case_insensitive(self):
        assert _model_aliases("Qwen-Plus") == {"qwen-plus"}


# ---------------------------------------------------------------------------
# 纯函数：_recursive_context_window
# ---------------------------------------------------------------------------
class TestRecursiveContextWindow:
    def test_flat_dict_first_key_wins(self):
        assert _recursive_context_window({"context_window": 64000, "other": 1}) == 64000

    def test_all_8_keys_recognized(self):
        for k in window_probe._CONTEXT_WINDOW_KEYS:
            assert _recursive_context_window({k: 32000}) == 32000, k

    def test_nested_dict(self):
        assert _recursive_context_window({"a": {"b": {"max_model_len": 128000}}}) == 128000

    def test_list(self):
        assert _recursive_context_window([{"max_input_tokens": 200000}]) == 200000

    def test_no_match(self):
        assert _recursive_context_window({"id": "x", "created": 1}) is None

    def test_ignores_nonpositive(self):
        assert _recursive_context_window({"context_window": 0, "context_length": -1}) is None


# ---------------------------------------------------------------------------
# 纯函数：_extract_context_window_from_payload
# ---------------------------------------------------------------------------
class TestExtractFromPayload:
    def test_data_array_exact_match(self):
        payload = {"data": [
            {"id": "other-model", "context_window": 8000},
            {"id": "qwen-plus", "context_window": 1_000_000},
        ]}
        assert _extract_context_window_from_payload(payload, "qwen-plus") == 1_000_000

    def test_models_key_wrapper(self):
        payload = {"models": [{"id": "gpt-4o", "max_context_tokens": 128000}]}
        assert _extract_context_window_from_payload(payload, "gpt-4o") == 128000

    def test_bare_list_payload(self):
        payload = [{"id": "x", "max_model_len": 64000}]
        assert _extract_context_window_from_payload(payload, "x") == 64000

    def test_partial_slash_match(self):
        # 目标 "openai/gpt-4o"，记录 id "gpt-4o" -> / 分隔 partial 命中
        payload = {"data": [{"id": "gpt-4o", "context_window": 128000}]}
        assert _extract_context_window_from_payload(payload, "openai/gpt-4o") == 128000

    def test_no_match_returns_none(self):
        payload = {"data": [{"id": "unrelated", "context_window": 8000}]}
        assert _extract_context_window_from_payload(payload, "qwen-plus") is None

    def test_empty_model_returns_none(self):
        assert _extract_context_window_from_payload({"data": [{"id": "x"}]}, "") is None

    def test_exact_preferred_over_partial(self):
        # 精确匹配记录先扫；两条都有窗口时精确的赢
        payload = {"data": [
            {"id": "qwen-plus", "context_window": 1_000_000},
            {"id": "provider/qwen-plus", "context_window": 999},
        ]}
        assert _extract_context_window_from_payload(payload, "qwen-plus") == 1_000_000

    def test_record_id_from_name_field(self):
        # 有些网关用 name 而非 id
        payload = {"data": [{"name": "qwen-plus", "context_window": 131072}]}
        assert _extract_context_window_from_payload(payload, "qwen-plus") == 131072


# ---------------------------------------------------------------------------
# detect_context_window（httpx MockTransport）
# ---------------------------------------------------------------------------
def _patch_httpx(monkeypatch, handler):
    """让 window_probe.detect_context_window 用的 httpx.AsyncClient 走 MockTransport(handler)。"""
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("transport", None)  # 防调用方误传
        return real_async_client(transport=transport, **kwargs)

    monkeypatch.setattr(window_probe.httpx, "AsyncClient", _factory)


@pytest.mark.asyncio
async def test_detect_success(monkeypatch):
    def handler(req):
        assert req.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "qwen-plus", "context_window": 1_000_000}]})
    _patch_httpx(monkeypatch, handler)
    assert await detect_context_window("qwen-plus", _BASE, _KEY) == 1_000_000


@pytest.mark.asyncio
async def test_detect_auth_header_sent(monkeypatch):
    captured = {}

    def handler(req):
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "m", "context_window": 64000}]})
    _patch_httpx(monkeypatch, handler)
    await detect_context_window("m", _BASE, "sk-secret")
    assert captured["auth"] == "Bearer sk-secret"


@pytest.mark.asyncio
async def test_detect_no_api_key_omits_auth(monkeypatch):
    captured = {}

    def handler(req):
        captured["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={"data": [{"id": "m", "context_window": 64000}]})
    _patch_httpx(monkeypatch, handler)
    await detect_context_window("m", _BASE, "")
    assert captured["auth"] is None  # 无 key 不发 Authorization


@pytest.mark.asyncio
async def test_detect_non200_returns_none(monkeypatch):
    _patch_httpx(monkeypatch, lambda req: httpx.Response(404))
    assert await detect_context_window("qwen-plus", _BASE, _KEY) is None


@pytest.mark.asyncio
async def test_detect_network_error_returns_none(monkeypatch):
    def handler(req):
        raise httpx.ConnectError("boom")
    _patch_httpx(monkeypatch, handler)
    assert await detect_context_window("qwen-plus", _BASE, _KEY) is None


@pytest.mark.asyncio
async def test_detect_no_window_fields_returns_none(monkeypatch):
    # OpenAI 原生 /models 只回 id/created/owned_by，无窗口字段 -> None（退回表）
    payload = {"data": [{"id": "qwen-plus", "created": 1, "owned_by": "openai"}]}
    _patch_httpx(monkeypatch, lambda req: httpx.Response(200, json=payload))
    assert await detect_context_window("qwen-plus", _BASE, _KEY) is None


@pytest.mark.asyncio
async def test_detect_empty_args_returns_none():
    # 无 model/base_url 不发请求（短路）
    assert await detect_context_window("", _BASE, _KEY) is None
    assert await detect_context_window("qwen-plus", "", _KEY) is None


@pytest.mark.asyncio
async def test_detect_strips_trailing_slash(monkeypatch):
    paths = []

    def handler(req):
        paths.append(str(req.url))
        return httpx.Response(200, json={"data": [{"id": "m", "context_window": 32000}]})
    _patch_httpx(monkeypatch, handler)
    await detect_context_window("m", "https://api.example.com/v1/", _KEY)
    assert paths and paths[0].startswith("https://api.example.com/v1/models")


# ---------------------------------------------------------------------------
# 缓存读写（tmp_path 隔离）
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """把探测缓存指向 tmp_path，并清进程内缓存，确保不污染真实 data 目录。"""
    cache_file = tmp_path / "context_window_cache.json"
    monkeypatch.setattr(window_probe, "_cache_path", lambda: str(cache_file))
    window_probe._reset_mem_cache()
    return cache_file


def test_read_miss_when_no_file(isolated_cache):
    assert not isolated_cache.exists()
    assert read_probe_cache(_BASE, "qwen-plus") is None


def test_write_then_read_hit(isolated_cache):
    write_probe_cache(_BASE, "qwen-plus", 1_000_000)
    assert read_probe_cache(_BASE, "qwen-plus") == 1_000_000


def test_read_normalizes_key_casing_and_slash(isolated_cache):
    # 写时带尾斜杠/大写，读时不带 -> 归一化后命中同一 key
    write_probe_cache("https://api.example.com/v1/", "Qwen-Plus", 1_000_000)
    assert read_probe_cache(_BASE, "qwen-plus") == 1_000_000


def test_read_miss_for_different_model(isolated_cache):
    write_probe_cache(_BASE, "qwen-plus", 1_000_000)
    assert read_probe_cache(_BASE, "qwen-max") is None


def test_read_miss_for_different_base_url(isolated_cache):
    write_probe_cache(_BASE, "qwen-plus", 1_000_000)
    assert read_probe_cache("https://other.example.com/v1", "qwen-plus") is None


def test_read_expired_returns_none(isolated_cache):
    # 手工写一条 detected_at 远过期（> 默认 7 天 TTL）的条目
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    data = {f"{_BASE.lower().rstrip('/')}|qwen-plus": {"window": 1_000_000, "detected_at": old}}
    isolated_cache.write_text(json.dumps(data), encoding="utf-8")
    window_probe._reset_mem_cache()  # 清进程内缓存，强制读盘
    assert read_probe_cache(_BASE, "qwen-plus") is None


def test_read_corrupt_timestamp_returns_none(isolated_cache):
    data = {f"{_BASE.lower().rstrip('/')}|qwen-plus": {"window": 1_000_000, "detected_at": "not-a-date"}}
    isolated_cache.write_text(json.dumps(data), encoding="utf-8")
    window_probe._reset_mem_cache()
    assert read_probe_cache(_BASE, "qwen-plus") is None


def test_read_corrupt_json_returns_none(isolated_cache):
    isolated_cache.write_text("{not valid json", encoding="utf-8")
    window_probe._reset_mem_cache()
    assert read_probe_cache(_BASE, "qwen-plus") is None
    # 坏文件不抛、不阻断；且进程内缓存空 dict，重复读不重复解析


def test_read_empty_dict_file_returns_none(isolated_cache):
    # 文件存在但为 {}（被清空）：复用进程内缓存（不重复读盘），返回 None 不崩
    isolated_cache.write_text("{}", encoding="utf-8")
    window_probe._reset_mem_cache()
    assert read_probe_cache(_BASE, "qwen-plus") is None
    assert read_probe_cache(_BASE, "qwen-max") is None  # 第二次走 mtime 复用路径


def test_write_invalid_window_noop(isolated_cache):
    write_probe_cache(_BASE, "qwen-plus", 0)
    write_probe_cache(_BASE, "qwen-plus", -10)
    assert not isolated_cache.exists() or read_probe_cache(_BASE, "qwen-plus") is None


def test_write_empty_identifiers_noop(isolated_cache):
    write_probe_cache("", "qwen-plus", 1_000_000)
    write_probe_cache(_BASE, "", 1_000_000)
    assert not isolated_cache.exists()


def test_write_preserves_existing_entries(isolated_cache):
    write_probe_cache(_BASE, "qwen-plus", 1_000_000)
    write_probe_cache(_BASE, "qwen-max", 32768)
    assert read_probe_cache(_BASE, "qwen-plus") == 1_000_000
    assert read_probe_cache(_BASE, "qwen-max") == 32768


def test_write_atomic_replaces(isolated_cache):
    # 写两次覆写同一 key，文件仍是合法 JSON 且读到最后值
    write_probe_cache(_BASE, "qwen-plus", 1_000_000)
    write_probe_cache(_BASE, "qwen-plus", 2_000_000)
    assert read_probe_cache(_BASE, "qwen-plus") == 2_000_000
    # 无残留 .tmp
    assert not (isolated_cache.parent / "context_window_cache.json.tmp").exists()
