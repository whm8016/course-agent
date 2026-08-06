"""Settings 回归测试 —— 捕获配置行为基线（嵌套式）。

行为契约（Phase 0 在扁平结构上建立，Phase 1 嵌套化后只改访问路径/env 名，
值断言保持不变 —— 用于捕获重构回归）：
  - env 注入（嵌套名 LLM__API_KEY 等）
  - _apply_catalog：model_catalog.json 覆盖默认
  - 跨组 fallback：embedding←llm、vision←embedding
  - _check_prod：production 安全门（JWT / CORS / PROVIDER_ENCRYPTION_KEY）
  - paths 默认值组装（BASE_DIR）
  - legacy alias / rag backend 平台解析 / llamaparse fallback
  - 全字段注入冒烟（防静默回落默认）

隔离手法：Settings(_env_file=None) 跳过 .env 文件；_fresh() 清理干扰 env 后按入参
重新设置，保证每个用例环境独立。catalog 默认 mock 为空（_set_env 内），catalog
专项用例自行覆盖。
"""
from __future__ import annotations

import os

import pytest

from settings import BASE_DIR, Settings

# 可能干扰断言的 env（嵌套式新名 + 顶层扁平名）。
_INTERFERING_ENV = [
    # 顶层扁平
    "ENVIRONMENT", "LOG_LEVEL", "TESTING", "BACKEND_WORKERS",
    "MAX_UPLOAD_MB", "MAX_KB_UPLOAD_MB",
    "LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
    # LLM
    "LLM__API_KEY", "LLM__BASE_URL", "LLM__BINDING", "LLM__API_VERSION",
    "LLM__TEXT_MODEL", "LLM__FAST_MODEL", "LLM__EXTRACT_MODEL", "LLM__KEYWORD_MODEL",
    "LLM__TIMEOUT_SEC",
    # Vision
    "VISION__MODEL", "VISION__INDEX_MODEL", "VISION__API_KEY", "VISION__BASE_URL",
    # Embedding
    "EMBEDDING__MODEL", "EMBEDDING__API_KEY", "EMBEDDING__BASE_URL",
    "EMBEDDING__DIM", "EMBEDDING__BATCH_SIZE",
    # Fallback
    "FALLBACK__API_KEY", "FALLBACK__BASE_URL", "FALLBACK__MODEL",
    # DB
    "DB__URL", "DB__REDIS_URL", "DB__POOL_SIZE", "DB__MAX_OVERFLOW",
    # Security
    "SECURITY__JWT_SECRET", "SECURITY__JWT_EXPIRE_HOURS",
    "SECURITY__ALLOWED_ORIGINS", "SECURITY__ADMIN_USERNAME",
    "SECURITY__PROVIDER_ENCRYPTION_KEY",
    # LlamaParse
    "LLAMAPARSE__CLOUD_API_KEY", "LLAMAPARSE__PARSE_API_KEY",
    # Chunking
    "CHUNKING__SIZE", "CHUNKING__OVERLAP", "CHUNKING__TOP_K",
    "CHUNKING__INGEST_SIZE", "CHUNKING__INGEST_OVERLAP",
    # LightRAG
    "LIGHTRAG__TOP_K", "LIGHTRAG__QUERY_MODE", "LIGHTRAG__ENABLED",
    # Mem0 / Summary
    "MEM0__TIME_DECAY_ENABLED", "SUMMARY__WINDOW_SIZE",
    # Paths（.env 可能显式设相对路径，需清理才能测默认值）
    "PATHS__UPLOAD_DIR", "PATHS__KNOWLEDGE_DIR", "PATHS__DB_PATH",
    "PATHS__QUESTION_LOG_DIR", "PATHS__LIGHTRAG_WORKDIR",
    "PATHS__KB_STORE_DIR", "PATHS__TUTORBOT_WORKSPACE_DIR", "PATHS__SEARCH_CONFIG_PATH",
    "PATHS__MCP_CONFIG_PATH", "PATHS__MCP_SESSIONS_DIR", "PATHS__OUTPUT_CARDS_PATH",
]


def _set_env(monkeypatch, **env) -> None:
    """清理干扰 env → 按 env 设置 + mock catalog 为空。

    不实例化，供需要在 pytest.raises 内实例化的用例。catalog 专项用例自行覆盖
    _load_model_catalog。
    """
    for k in _INTERFERING_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setattr("settings.base._load_model_catalog", lambda: {})


def _fresh(monkeypatch, **env) -> Settings:
    """清理干扰 env → 按 env 设置 → mock catalog 空 → 实例化（跳过 .env 文件）。"""
    _set_env(monkeypatch, **env)
    return Settings(_env_file=None)


# ── 默认实例化 ──────────────────────────────────────────────────────────────

def test_defaults_instantiate(monkeypatch):
    s = _fresh(monkeypatch)
    assert s.environment == "development"
    assert s.log_level == "INFO"
    assert s.testing is False
    assert s.llm.binding == "dashscope"
    assert s.llm.text_model == "qwen-plus"
    assert s.llm.fast_model == "qwen-turbo"
    assert s.vision.model == "qwen-vl-plus"
    assert s.embedding.model == "text-embedding-v3"
    assert s.lightrag.top_k == 6
    assert s.chunking.size == 500
    assert s.mem0.time_decay_enabled is False
    assert s.summary.window_size == 5


# ── env 注入（嵌套名）──────────────────────────────────────────────────────

def test_env_injection_nested(monkeypatch):
    s = _fresh(
        monkeypatch,
        LLM__API_KEY="sk-abc",
        LLM__TEXT_MODEL="deepseek-v4",
        DB__URL="postgresql+asyncpg://user:pw@host/db",
        SECURITY__JWT_SECRET="super-secret-value",
        LIGHTRAG__TOP_K="33",
        CHUNKING__SIZE="600",
    )
    assert s.llm.api_key.get_secret_value() == "sk-abc"
    assert s.llm.text_model == "deepseek-v4"
    assert s.db.url.get_secret_value() == "postgresql+asyncpg://user:pw@host/db"
    assert s.security.jwt_secret.get_secret_value() == "super-secret-value"
    assert s.lightrag.top_k == 33
    assert s.chunking.size == 600


def test_allowed_origins_split(monkeypatch):
    s = _fresh(monkeypatch, SECURITY__ALLOWED_ORIGINS="https://a.com, https://b.com")
    assert s.security.allowed_origins == ["https://a.com", "https://b.com"]


def test_allowed_origins_wildcard(monkeypatch):
    assert _fresh(monkeypatch, SECURITY__ALLOWED_ORIGINS="*").security.allowed_origins == ["*"]
    assert _fresh(monkeypatch, SECURITY__ALLOWED_ORIGINS="").security.allowed_origins == ["*"]


def test_truthy_bool_parsing(monkeypatch):
    assert _fresh(monkeypatch, MEM0__TIME_DECAY_ENABLED="true").mem0.time_decay_enabled is True
    assert _fresh(monkeypatch, MEM0__TIME_DECAY_ENABLED="yes").mem0.time_decay_enabled is True
    assert _fresh(monkeypatch, MEM0__TIME_DECAY_ENABLED="0").mem0.time_decay_enabled is False


# ── catalog 覆盖（_apply_catalog）─────────────────────────────────────────────

def test_catalog_overrides_defaults(monkeypatch):
    cat = {
        "binding": "openai",
        "api_key": "sk-cat",
        "base_url": "https://api.openai.com/v1",
        "api_version": "2024-02-15",
        "text_model": "gpt-4o",
        "fast_model": "gpt-4o-mini",
        "vision_model": "",
        "embedding_model": "text-embedding-3-large",
        "embedding_api_key": "sk-emb-cat",
        "embedding_base_url": "https://api.openai.com/v1",
        "fallback_api_key": "sk-fb",
        "fallback_base_url": "",
        "fallback_model": "gpt-4o-fallback",
    }
    _set_env(monkeypatch)  # 清 env（_set_env 内 mock 空 catalog，下方覆盖为真实 cat）
    monkeypatch.setattr("settings.base._load_model_catalog", lambda: cat)
    s = Settings(_env_file=None)
    assert s.llm.binding == "openai"
    assert s.llm.api_key.get_secret_value() == "sk-cat"
    assert s.llm.base_url == "https://api.openai.com/v1"
    assert s.llm.api_version == "2024-02-15"
    assert s.llm.text_model == "gpt-4o"
    assert s.llm.fast_model == "gpt-4o-mini"
    assert s.embedding.model == "text-embedding-3-large"
    assert s.embedding.api_key.get_secret_value() == "sk-emb-cat"
    assert s.fallback.api_key.get_secret_value() == "sk-fb"
    assert s.fallback.model == "gpt-4o-fallback"


def test_catalog_empty_keeps_env_defaults(monkeypatch):
    s = _fresh(monkeypatch, LLM__TEXT_MODEL="from-env")
    assert s.llm.text_model == "from-env"
    assert s.llm.binding == "dashscope"


def test_catalog_does_not_override_vision(monkeypatch):
    """vision.model 不从 catalog 覆盖（全局独立视觉模型语义）。"""
    cat = {"vision_model": "should-be-ignored", "text_model": "gpt-4o"}
    _set_env(monkeypatch, VISION__MODEL="qwen-vl-max")
    monkeypatch.setattr("settings.base._load_model_catalog", lambda: cat)
    s = Settings(_env_file=None)
    assert s.vision.model == "qwen-vl-max"


# ── 跨组 fallback（_apply_legacy_and_fallbacks）──────────────────────────────

def test_fallback_embedding_from_llm(monkeypatch):
    s = _fresh(monkeypatch, LLM__API_KEY="sk-main")
    # embedding 空 → 回退 llm
    assert s.embedding.api_key.get_secret_value() == "sk-main"
    assert s.embedding.base_url == s.llm.base_url


def test_fallback_vision_from_embedding(monkeypatch):
    s = _fresh(monkeypatch, LLM__API_KEY="sk-main")
    # vision 空 → 回退 embedding（已 fallback 到 llm）
    assert s.vision.api_key.get_secret_value() == "sk-main"
    assert s.vision.base_url == s.embedding.base_url


def test_explicit_vision_not_overridden(monkeypatch):
    s = _fresh(
        monkeypatch,
        LLM__API_KEY="sk-main",
        VISION__API_KEY="sk-vis-explicit",
        VISION__BASE_URL="https://vis.example.com",
    )
    assert s.vision.api_key.get_secret_value() == "sk-vis-explicit"
    assert s.vision.base_url == "https://vis.example.com"


def test_llamaparse_cloud_from_parse(monkeypatch):
    s = _fresh(monkeypatch, LLAMAPARSE__PARSE_API_KEY="lp-x")
    assert s.llamaparse.cloud_api_key.get_secret_value() == "lp-x"


# ── prod 安全门（_check_prod）─────────────────────────────────────────────────

def test_prod_rejects_default_jwt(monkeypatch):
    _set_env(monkeypatch, ENVIRONMENT="production")  # SECURITY__JWT_SECRET 清空 → 默认 dev-secret
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        Settings(_env_file=None)


def test_prod_rejects_wildcard_cors(monkeypatch):
    _set_env(monkeypatch, ENVIRONMENT="production", SECURITY__JWT_SECRET="strong-prod-secret-value-32chars!")
    # SECURITY__ALLOWED_ORIGINS 清空 → 默认 ["*"]
    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        Settings(_env_file=None)


def test_prod_rejects_missing_provider_key(monkeypatch):
    _set_env(
        monkeypatch,
        ENVIRONMENT="production",
        SECURITY__JWT_SECRET="strong-prod-secret-value-32chars!",
        SECURITY__ALLOWED_ORIGINS="https://example.com",
    )
    # SECURITY__PROVIDER_ENCRYPTION_KEY 清空 → 默认空
    with pytest.raises(RuntimeError, match="PROVIDER_ENCRYPTION_KEY"):
        Settings(_env_file=None)


def test_prod_passes_with_full_config(monkeypatch):
    """production + 合规 JWT/CORS/PROVIDER_KEY → 正常实例化。"""
    s = _fresh(
        monkeypatch,
        ENVIRONMENT="production",
        SECURITY__JWT_SECRET="strong-prod-secret-value-32chars!",
        SECURITY__ALLOWED_ORIGINS="https://example.com",
        SECURITY__PROVIDER_ENCRYPTION_KEY="some-fernet-key",
    )
    assert s.environment == "production"
    assert s.is_production is True


def test_dev_warns_default_jwt(monkeypatch, recwarn):
    """dev 用默认 jwt_secret 不抛错（仅告警）。"""
    s = _fresh(monkeypatch)  # SECURITY__JWT_SECRET 清空 → 默认 dev-secret
    assert "dev-secret-change-in-production" in s.security.jwt_secret.get_secret_value()


# ── paths 默认值 ─────────────────────────────────────────────────────────────

def test_paths_defaults(monkeypatch):
    s = _fresh(monkeypatch)
    assert s.paths.upload_dir == os.path.join(BASE_DIR, "uploads")
    assert s.paths.knowledge_dir == os.path.join(BASE_DIR, "knowledge")
    assert s.paths.lightrag_workdir == os.path.join(BASE_DIR, "lightrag_store")
    assert s.paths.kb_store_dir == os.path.join(BASE_DIR, "kb_store")
    assert s.paths.mcp_config_path == os.path.join(BASE_DIR, "data", "mcp.json")


def test_paths_explicit_not_overridden(monkeypatch, tmp_path):
    custom = str(tmp_path / "my-uploads")
    s = _fresh(monkeypatch, PATHS__UPLOAD_DIR=custom)
    assert s.paths.upload_dir == custom


# ── lightrag 计算方法（原 rag_config 计算函数，内聚至 LightRAGConfig）─────────

def test_lightrag_compute_methods(monkeypatch):
    s = _fresh(monkeypatch, LIGHTRAG__TOP_K="20")
    assert s.lightrag.top_k == 20
    # chunk_top_k_value = min(chunk_top_k=5, top_k=20) = 5
    assert s.lightrag.chunk_top_k_value() == 5
    mt = s.lightrag.max_tokens_config()
    assert set(mt.keys()) == {"total", "entity", "relation"}
    # LRU 缩放：lru_capacity(10) // backend_workers(4) = 2
    assert s.lightrag_lru_capacity_scaled == 2


# ── 全字段注入冒烟（防 R1：静默回落默认）──────────────────────────────────────

def test_full_env_injection_smoke(monkeypatch):
    """所有关键 .env 字段都能正确注入到对应 settings 嵌套字段（防漏改静默用默认）。"""
    env = {
        "LLM__API_KEY": "sk-1",
        "LLM__BASE_URL": "https://llm.example.com/v1",
        "LLM__BINDING": "openai",
        "LLM__TEXT_MODEL": "m-text",
        "LLM__FAST_MODEL": "m-fast",
        "VISION__MODEL": "m-vis",
        "EMBEDDING__MODEL": "m-emb",
        "FALLBACK__MODEL": "m-fb",
        "DB__URL": "postgresql+asyncpg://db-host/db",
        "DB__REDIS_URL": "redis://r-host:6379/0",
        "SECURITY__JWT_SECRET": "jwt-smoke",
        "SECURITY__ALLOWED_ORIGINS": "https://a.com,https://b.com",
        "LIGHTRAG__TOP_K": "33",
        "LIGHTRAG__QUERY_MODE": "hybrid",
        "CHUNKING__SIZE": "600",
        "CHUNKING__TOP_K": "8",
        "MEM0__TIME_DECAY_ENABLED": "true",
        "SUMMARY__WINDOW_SIZE": "7",
        "MAX_UPLOAD_MB": "25",
    }
    s = _fresh(monkeypatch, **env)
    assert s.llm.api_key.get_secret_value() == "sk-1"
    assert s.llm.base_url == "https://llm.example.com/v1"
    assert s.llm.binding == "openai"
    assert s.llm.text_model == "m-text"
    assert s.llm.fast_model == "m-fast"
    assert s.vision.model == "m-vis"
    assert s.embedding.model == "m-emb"
    assert s.fallback.model == "m-fb"
    assert s.db.url.get_secret_value() == "postgresql+asyncpg://db-host/db"
    assert s.db.redis_url.get_secret_value() == "redis://r-host:6379/0"
    assert s.security.jwt_secret.get_secret_value() == "jwt-smoke"
    assert s.security.allowed_origins == ["https://a.com", "https://b.com"]
    assert s.lightrag.top_k == 33
    assert s.lightrag.query_mode == "hybrid"
    assert s.chunking.size == 600
    assert s.chunking.top_k == 8
    assert s.mem0.time_decay_enabled is True
    assert s.summary.window_size == 7
    assert s.max_upload_mb == 25


# ── get_settings 单例 ────────────────────────────────────────────────────────

def test_get_settings_cached(monkeypatch):
    import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    a = settings_mod.get_settings()
    b = settings_mod.get_settings()
    assert a is b
    settings_mod.get_settings.cache_clear()


# ── M-24：BACKEND_WORKERS 运行时校验（validate_runtime_workers）──────────────

def _no_worker_warning(warnings_list):
    """辅助：列表里是否含 BACKEND_WORKERS mismatch 告警。"""
    return any("BACKEND_WORKERS" in str(w.message) for w in warnings_list)


def test_validate_runtime_workers_match(monkeypatch):
    """显式注入 known_workers 与 backend_workers 相等 → 返回 True，无告警。"""
    import warnings as _warnings
    s = _fresh(monkeypatch, BACKEND_WORKERS="4")
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert s.validate_runtime_workers(known_workers=4) is True
    assert not _no_worker_warning(rec)


def test_validate_runtime_workers_mismatch_warns(monkeypatch):
    """known_workers 与 backend_workers 不等 → 返回 False + 告警（不抛异常）。"""
    s = _fresh(monkeypatch, BACKEND_WORKERS="4")
    with pytest.warns(UserWarning, match="BACKEND_WORKERS .* does not match actual worker count"):
        result = s.validate_runtime_workers(known_workers=8)
    assert result is False


def test_validate_runtime_workers_from_web_concurrency(monkeypatch):
    """无显式注入时读 WEB_CONCURRENCY 约定；与 backend_workers 不等 → 告警。"""
    _set_env(monkeypatch, BACKEND_WORKERS="4")
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    s = Settings(_env_file=None)
    with pytest.warns(UserWarning, match="does not match"):
        assert s.validate_runtime_workers() is False


def test_validate_runtime_workers_no_env_skips(monkeypatch):
    """无 WEB_CONCURRENCY 也无显式注入（dev/pytest 单进程）→ 跳过校验，返回 True。"""
    import warnings as _warnings
    s = _fresh(monkeypatch, BACKEND_WORKERS="4")
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert s.validate_runtime_workers() is True
    assert not _no_worker_warning(rec)


def test_validate_runtime_workers_invalid_env_skips(monkeypatch):
    """WEB_CONCURRENCY 非法值（非数字）→ 跳过，不阻断启动。"""
    import warnings as _warnings
    _set_env(monkeypatch, BACKEND_WORKERS="4")
    monkeypatch.setenv("WEB_CONCURRENCY", "not-a-number")
    s = Settings(_env_file=None)
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert s.validate_runtime_workers() is True
    assert not _no_worker_warning(rec)


def test_validate_runtime_workers_nonpositive_skips(monkeypatch):
    """known_workers <= 0 → 无法判定，跳过（返回 True），不制造噪音。"""
    import warnings as _warnings
    s = _fresh(monkeypatch, BACKEND_WORKERS="4")
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        assert s.validate_runtime_workers(known_workers=0) is True
        assert s.validate_runtime_workers(known_workers=-1) is True
    assert not _no_worker_warning(rec)

