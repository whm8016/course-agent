# search/config.py
from __future__ import annotations

import json
import os as _os
from dataclasses import dataclass
from pathlib import Path

from settings import get_settings
SEARCH_API_KEY = get_settings().search.api_key.get_secret_value()
SEARCH_BASE_URL = get_settings().search.base_url
SEARCH_CONFIG_PATH = get_settings().paths.search_config_path
SEARCH_MAX_RESULTS = get_settings().search.max_results
SEARCH_PROVIDER = get_settings().search.provider
SEARCH_PROXY = get_settings().search.proxy


@dataclass(frozen=True)
class SearchConfig:
    provider: str
    api_key: str
    base_url: str
    max_results: int
    proxy: str | None
    # Informational fields for UI display
    requested_provider: str = ""
    status: str = "active"
    missing_credentials: bool = False
    fallback_reason: str | None = None


def _admin_config_path() -> Path:
    """admin 全局默认配置文件：backend/data/search_config.json。"""
    return Path(SEARCH_CONFIG_PATH)


def load_admin_default() -> dict:
    """同步读 admin 全局默认；缺失/损坏→{}（回退 env）。"""
    try:
        path = _admin_config_path()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_admin_default(data: dict) -> dict:
    """写 admin 全局默认到 data/search_config.json（仅保留已知字段）。"""
    path = _admin_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": str(data.get("provider") or "").strip(),
        "api_key": str(data.get("api_key") or ""),
        "base_url": str(data.get("base_url") or "").strip(),
        "max_results": int(data.get("max_results") or 0),
        "proxy": str(data.get("proxy") or "").strip(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _first(*vals):
    """第一个真值（None / '' / 0 视为空）；全空返回 None。"""
    for v in vals:
        if v not in (None, "", 0):
            return v
    return None


def resolve_search_config(user_override: dict | None = None) -> SearchConfig:
    """合并三层：user_override > admin 默认(data/search_config.json) > env/config。

    字段级合并：每字段取第一个非空值（user 非空优先于 admin，admin 非空优先于 env）。
    本函数同步、不查 DB——user_override 由调用方（async 层）读好传入。
    """
    admin = load_admin_default()
    u = user_override or {}

    # 四层优先级：user > admin > config固化值 > 实时env（测试用monkeypatch时能生效）
    env_provider = _os.environ.get("SEARCH_PROVIDER", "")
    env_api_key = _os.environ.get("SEARCH_API_KEY", "")
    env_base_url = _os.environ.get("SEARCH_BASE_URL", "")
    env_max_results = _os.environ.get("SEARCH_MAX_RESULTS", "")
    env_proxy = _os.environ.get("SEARCH_PROXY", "")

    requested = str(
        _first(u.get("provider"), admin.get("provider"), env_provider, SEARCH_PROVIDER) or "duckduckgo"
    ).strip().lower()
    provider = requested
    api_key = str(_first(u.get("api_key"), admin.get("api_key"), env_api_key, SEARCH_API_KEY) or "")
    base_url = str(_first(u.get("base_url"), admin.get("base_url"), env_base_url, SEARCH_BASE_URL) or "")
    mr = _first(u.get("max_results"), admin.get("max_results"), env_max_results, SEARCH_MAX_RESULTS)
    max_results = int(mr) if mr not in (None, "", 0) else SEARCH_MAX_RESULTS
    proxy = _first(u.get("proxy"), admin.get("proxy"), env_proxy, SEARCH_PROXY) or None

    # Detect credential gaps for informational display only
    missing = False
    fallback_reason: str | None = None
    if provider in {"brave", "tavily", "jina", "perplexity", "serper"} and not api_key:
        missing = True
        if provider not in {"perplexity", "serper"}:
            fallback_reason = f"{provider} missing API key; would fall back to duckduckgo"
            provider = "duckduckgo"
    elif provider == "searxng" and not base_url:
        fallback_reason = "searxng missing base_url; would fall back to duckduckgo"
        provider = "duckduckgo"

    return SearchConfig(
        provider=provider,
        requested_provider=requested,
        api_key=api_key,
        base_url=base_url,
        max_results=max_results,
        proxy=proxy,
        status="active",
        missing_credentials=missing,
        fallback_reason=fallback_reason,
    )
