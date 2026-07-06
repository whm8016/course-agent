"""搜索配置三层合并（user > admin > env）+ admin 文件读写 + probe 测试。

resolve_search_config 是纯函数（同步、不查 DB），最易测且最关键。
"""
from services.search import probe_search
from services.search.config import (
    load_admin_default,
    resolve_search_config,
    save_admin_default,
)


def _no_admin(monkeypatch):
    monkeypatch.setattr("services.search.config.load_admin_default", lambda: {})


def _no_env_defaults(monkeypatch):
    """清掉 .env 经 settings 固化进模块常量的默认值（嵌套化后 SEARCH__* 会固化），
    使 resolve 仅受实时 env / 参数驱动 —— 还原「无任何配置」的测试前提。"""
    import services.search.config as sc
    monkeypatch.setattr(sc, "SEARCH_PROVIDER", "")
    monkeypatch.setattr(sc, "SEARCH_API_KEY", "")
    monkeypatch.setattr(sc, "SEARCH_BASE_URL", "")


def test_resolve_env_only(monkeypatch):
    _no_admin(monkeypatch)
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("SEARCH_API_KEY", "env-k")  # brave 需要 API key 避免 fallback
    cfg = resolve_search_config()
    assert cfg.provider == "brave"
    assert cfg.api_key == "env-k"


def test_resolve_admin_overrides_env(monkeypatch):
    monkeypatch.setattr(
        "services.search.config.load_admin_default",
        lambda: {"provider": "tavily", "api_key": "admin-k"},
    )
    cfg = resolve_search_config()
    assert cfg.provider == "tavily"
    assert cfg.api_key == "admin-k"


def test_resolve_user_overrides_admin(monkeypatch):
    monkeypatch.setattr(
        "services.search.config.load_admin_default",
        lambda: {"provider": "brave", "api_key": "admin-k"},
    )
    cfg = resolve_search_config(user_override={"provider": "jina", "api_key": "user-k"})
    assert cfg.provider == "jina"
    assert cfg.api_key == "user-k"


def test_resolve_field_level_merge(monkeypatch):
    """user 只覆盖 api_key，provider 留空 → 用 admin 的 provider + user 的 key。"""
    monkeypatch.setattr(
        "services.search.config.load_admin_default",
        lambda: {"provider": "brave", "api_key": "admin-k"},
    )
    cfg = resolve_search_config(user_override={"api_key": "user-k"})
    assert cfg.provider == "brave"
    assert cfg.api_key == "user-k"


def test_resolve_fallback_when_key_missing(monkeypatch):
    """brave 无 key → 降级 duckduckgo（与原 env 行为一致）。"""
    _no_admin(monkeypatch)
    _no_env_defaults(monkeypatch)  # 清固化 key，否则 brave 会拿到 .env 的 key 不降级
    cfg = resolve_search_config(user_override={"provider": "brave"})  # 无 key
    assert cfg.provider == "duckduckgo"
    assert cfg.missing_credentials is True
    assert "missing API key" in (cfg.fallback_reason or "")


def test_resolve_default_duckduckgo(monkeypatch):
    _no_admin(monkeypatch)
    _no_env_defaults(monkeypatch)  # 清 .env 固化的 tavily，还原「无配置」前提
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    cfg = resolve_search_config()
    assert cfg.provider == "duckduckgo"


def test_admin_save_load(tmp_path, monkeypatch):
    p = tmp_path / "search_config.json"
    monkeypatch.setattr("services.search.config._admin_config_path", lambda: p)
    saved = save_admin_default(
        {"provider": "duckduckgo", "api_key": "x", "base_url": "", "max_results": 7, "proxy": ""}
    )
    assert saved["provider"] == "duckduckgo"
    assert saved["max_results"] == 7
    loaded = load_admin_default()
    assert loaded["api_key"] == "x"
    assert loaded["max_results"] == 7


def test_probe_unsupported_provider():
    r = probe_search("not_a_provider", "k")
    assert r["ok"] is False
    assert r["provider"] == "not_a_provider"


def test_probe_disabled():
    r = probe_search("none")
    assert r["ok"] is False


def test_probe_ok(monkeypatch):
    class _FakeSP:
        def search(self, q, **kw):
            return {"answer": "", "citations": [], "search_results": []}

    monkeypatch.setattr("services.search.get_provider", lambda name, **kw: _FakeSP())
    r = probe_search("brave", "k")
    assert r["ok"] is True
    assert r["provider"] == "brave"


def test_probe_error(monkeypatch):
    def _boom(name, **kw):
        raise RuntimeError("boom-no-key")

    monkeypatch.setattr("services.search.get_provider", _boom)
    r = probe_search("brave", "k")
    assert r["ok"] is False
    assert "boom-no-key" in r["error"]
