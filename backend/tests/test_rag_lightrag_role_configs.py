"""LightRAG 分角色模型（role_llm_configs）接线测试。

覆盖 llm_adapter.build_role_llm_configs()：
- extract/keyword 皆空 → 返回 None（不传 role_llm_configs，LightRAG 全角色回退 base，行为不变）。
- 配 extract/keyword → 返回 dict 仅含对应角色；未现角色（query/vlm）交给 LightRAG 回退 base。
- 角色 func 调用时，openai_complete_if_cache 的 model 路由到对应模型名，凭证复用 INDEX_LLM_*。
- 角色 func 失败仍写入共享 _llm_error_log（take_llm_errors 调用方零感知）。
- base _llm_model_func 重构后行为不变（model = INDEX_LLM_MODEL）。
"""
from __future__ import annotations

import pytest


def test_role_configs_none_when_unconfigured(monkeypatch):
    """extract/keyword 皆空 → None（默认配置，行为与改动前完全一致）。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "INDEX_LLM_EXTRACT_MODEL", "")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_KEYWORD_MODEL", "")
    assert llm_adapter.build_role_llm_configs() is None


def test_role_configs_extract_only(monkeypatch):
    """只配 extract → dict 仅含 extract key。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "INDEX_LLM_EXTRACT_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_KEYWORD_MODEL", "")
    cfgs = llm_adapter.build_role_llm_configs()
    assert cfgs is not None
    assert list(cfgs.keys()) == ["extract"]


def test_role_configs_both(monkeypatch):
    """extract + keyword 都配 → dict 含两个 key。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "INDEX_LLM_EXTRACT_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_KEYWORD_MODEL", "deepseek-v4-flash")
    cfgs = llm_adapter.build_role_llm_configs()
    assert cfgs is not None
    assert set(cfgs.keys()) == {"extract", "keyword"}


@pytest.mark.asyncio
async def test_role_func_routes_model_and_credentials(monkeypatch):
    """extract 角色 func：model=extract_model，凭证=INDEX_LLM_*（同 provider 复用）。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "INDEX_LLM_EXTRACT_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_KEYWORD_MODEL", "")

    captured: dict = {}

    async def _fake_complete(model, prompt, **kwargs):
        captured["model"] = model
        captured["api_key"] = kwargs.get("api_key")
        captured["base_url"] = kwargs.get("base_url")
        return "ok"

    monkeypatch.setattr(llm_adapter, "openai_complete_if_cache", _fake_complete)

    cfgs = llm_adapter.build_role_llm_configs()
    assert cfgs is not None
    result = await cfgs["extract"].func("prompt-text", system_prompt="sys")
    assert result == "ok"
    assert captured["model"] == "deepseek-v4-flash"
    # 凭证复用 base 专属 provider（不跨 provider）
    assert captured["api_key"] == llm_adapter.INDEX_LLM_API_KEY
    assert captured["base_url"] == llm_adapter.INDEX_LLM_BASE_URL


@pytest.mark.asyncio
async def test_role_func_records_errors_to_shared_buffer(monkeypatch):
    """角色 func 失败时写入共享 _llm_error_log（take_llm_errors 调用方零感知）。"""
    from core.rag.lightrag import llm_adapter

    monkeypatch.setattr(llm_adapter, "INDEX_LLM_EXTRACT_MODEL", "deepseek-v4-flash")
    monkeypatch.setattr(llm_adapter, "INDEX_LLM_KEYWORD_MODEL", "")
    llm_adapter._llm_error_log.clear()

    async def _boom(model, prompt, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_adapter, "openai_complete_if_cache", _boom)

    cfgs = llm_adapter.build_role_llm_configs()
    with pytest.raises(RuntimeError):
        await cfgs["extract"].func("p")
    errs = llm_adapter.take_llm_errors()
    assert len(errs) == 1
    assert "boom" in str(errs[0])


@pytest.mark.asyncio
async def test_base_llm_func_unchanged_after_refactor(monkeypatch):
    """base _llm_model_func 重构后 model 仍 = INDEX_LLM_MODEL（回归底线）。"""
    from core.rag.lightrag import llm_adapter

    captured: dict = {}

    async def _fake_complete(model, prompt, **kwargs):
        captured["model"] = model
        return "ok"

    monkeypatch.setattr(llm_adapter, "openai_complete_if_cache", _fake_complete)

    await llm_adapter._llm_model_func("prompt", system_prompt="sys")
    assert captured["model"] == llm_adapter.INDEX_LLM_MODEL
