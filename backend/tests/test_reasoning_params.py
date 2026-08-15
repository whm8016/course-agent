"""思考开关按型号分流（照 DeepTutor reasoning_params 设计）测试。

三层验证：
1. reasoning_kwargs 表判定——flash 关 / 推理档开 / gemini 关(none|minimal) / 不在表不传；
2. resolve_provider 反查——base_url 域名 > 模型名 > binding（binding 是 legacy 错标签）；
3. apply_reasoning_kwargs 注入/剥除——含 _create_with_image_fallback 咽喉注入与
   fallback 换模型后重算（deepseek 主模型参数不串到 dashscope）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.llm.llm import _create_with_image_fallback
from core.llm.reasoning_params import (
    apply_reasoning_kwargs,
    reasoning_kwargs,
    resolve_provider,
)
from core.llm.reliability import LLMRetryError
from settings import get_settings


# ── 1. 表判定 ─────────────────────────────────────────────────────────────────


def test_flash_default_disabled():
    """deepseek-v4-flash：名字叫快、出厂却开思考——默认关。"""
    assert reasoning_kwargs("deepseek", "deepseek-v4-flash") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_deepseek_reasoning_tier_enabled():
    """推理档（v4-pro / reasoner）：显式开 + high（先于默认关表判定）。"""
    for model in ("deepseek-v4-pro", "deepseek-reasoner"):
        assert reasoning_kwargs("deepseek", model) == {
            "reasoning_effort": "high",
            "extra_body": {"thinking": {"type": "enabled"}},
        }


def test_dashscope_reasoning_tier_enabled():
    """dashscope 推理族：enable_thinking=True。"""
    for model in ("qwen3-max", "qwen-plus", "qwq-32b", "qwen-3-235b"):
        assert reasoning_kwargs("dashscope", model) == {
            "extra_body": {"enable_thinking": True}
        }


def test_gemini_default_off_none():
    """gemini-2.5 系默认关（思考吃光 max_tokens 致正文空）。"""
    assert reasoning_kwargs("gemini", "gemini-2.5-flash") == {"reasoning_effort": "none"}


def test_gemini_minimal_only():
    """gemini-3* / gemini-2.5-pro 拒绝 none，只能 minimal。"""
    assert reasoning_kwargs("gemini", "gemini-3-pro") == {"reasoning_effort": "minimal"}
    assert reasoning_kwargs("gemini", "gemini-2.5-pro") == {"reasoning_effort": "minimal"}


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-4o"),          # 无 thinking API，乱塞参数会 400
        ("anthropic", "claude-sonnet-5"),
        ("deepseek", "deepseek-v3"),   # 非推理档非 flash，跟供应商默认
        (None, "whatever"),
        ("deepseek", None),
    ],
)
def test_not_in_tables_pass_nothing(provider, model):
    """不在表内 → 空 dict，kwargs 一根毫毛不动。"""
    assert reasoning_kwargs(provider, model) == {}


# ── 总闸 LLM__REASONING_EFFORT 覆盖整表 ───────────────────────────────────────


def test_env_override_minimal_forces_off(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "reasoning_effort", "minimal")
    assert reasoning_kwargs("deepseek", "deepseek-v4-pro") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert reasoning_kwargs("dashscope", "qwq-32b") == {"extra_body": {"enable_thinking": False}}


def test_env_override_high_forces_on(monkeypatch):
    monkeypatch.setattr(get_settings().llm, "reasoning_effort", "high")
    assert reasoning_kwargs("deepseek", "deepseek-v4-flash") == {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    }


# ── 2. provider 反查（base_url > 模型名 > binding）───────────────────────────


def test_find_by_base_url_matches_and_disambiguates():
    """从 default_api_base 派生，无平行事实源；本地三家靠端口区分。"""
    from core.llm.provider_registry import find_by_base_url

    assert find_by_base_url("https://api.deepseek.com/v1").name == "deepseek"
    assert find_by_base_url("https://dashscope.aliyuncs.com/compatible-mode/v1").name == "dashscope"
    # localhost 三家（ollama/vllm/lm_studio）域名撞车，端口说了算
    assert find_by_base_url("http://localhost:11434/v1").name == "ollama"
    assert find_by_base_url("http://localhost:8000/v1").name == "vllm"
    assert find_by_base_url(None) is None
    assert find_by_base_url("https://unknown.example.com/v1") is None


def test_resolve_provider_url_beats_wrong_binding():
    """现状复刻：binding=dashscope（legacy 默认）但实际打 deepseek 端点——域名说了算。"""
    assert (
        resolve_provider("dashscope", "deepseek-v4-flash", "https://api.deepseek.com/v1")
        == "deepseek"
    )


def test_resolve_provider_url_dashscope():
    assert (
        resolve_provider("deepseek", "qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        == "dashscope"
    )


def test_resolve_provider_model_name_fallback():
    assert resolve_provider(None, "deepseek-v4-flash", None) == "deepseek"
    assert resolve_provider(None, "gemini-2.5-pro", None) == "gemini"


def test_resolve_provider_binding_last():
    assert resolve_provider("azure", None, None) == "azure_openai"
    assert resolve_provider("claude", None, None) == "anthropic"


# ── 3. 注入 / 剥除 ────────────────────────────────────────────────────────────


def _fake_client(base_url: str):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create = AsyncMock(return_value=MagicMock())
    return client


async def test_choke_point_injects_flash_disabled():
    """咽喉层注入：binding 错标签也不怕，flash 请求带上 thinking disabled。"""
    client = _fake_client("https://api.deepseek.com/v1")
    kwargs = {"model": "deepseek-v4-flash", "messages": [], "stream": True}
    await _create_with_image_fallback(
        client, kwargs, binding="dashscope", model="deepseek-v4-flash", circuit_breaker=None
    )
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in sent


async def test_choke_point_unknown_model_untouched():
    """不在表内：kwargs 与传入逐字节一致（不加任何思考参数）。"""
    client = _fake_client("https://api.openai.com/v1")
    kwargs = {"model": "gpt-4o", "messages": [], "stream": True}
    await _create_with_image_fallback(
        client, kwargs, binding="openai", model="gpt-4o", circuit_breaker=None
    )
    sent = client.chat.completions.create.call_args.kwargs
    assert sent == {"model": "gpt-4o", "messages": [], "stream": True}


async def test_fallback_swap_recomputes_for_fallback_model(monkeypatch):
    """主模型(deepseek flash)熔断 → fallback(qwen-plus@dashscope)：剥 thinking
    disabled、按 fallback 供应商重算 enable_thinking，绝不跨供应商串参数。"""
    import core.llm.llm as llm_mod

    primary = _fake_client("https://api.deepseek.com/v1")
    fallback = _fake_client("https://dashscope.aliyuncs.com/compatible-mode/v1")

    calls: list = []

    async def _fake_wrapped(call, *, retry_config, circuit_breaker):
        calls.append(1)
        if len(calls) == 1:
            raise LLMRetryError("primary down")
        return await call()

    monkeypatch.setattr(llm_mod, "with_retry_and_circuit", _fake_wrapped)
    monkeypatch.setattr(llm_mod, "_fallback_client", fallback)
    monkeypatch.setattr(llm_mod, "_fallback_binding", "dashscope")
    monkeypatch.setattr(llm_mod, "FALLBACK_MODEL", "qwen-plus")

    kwargs = {"model": "deepseek-v4-flash", "messages": [], "stream": True}
    await _create_with_image_fallback(
        primary, kwargs, binding="deepseek", model="deepseek-v4-flash",
        circuit_breaker=MagicMock(),  # 非 None 才走 fallback 分支
    )

    assert not primary.chat.completions.create.await_count  # 主路径没真打
    sent = fallback.chat.completions.create.call_args.kwargs
    assert sent["model"] == "qwen-plus"
    assert sent["extra_body"] == {"enable_thinking": True}


def test_apply_strips_then_reapplies():
    """幂等：带着上一轮的思考参数重调，先剥后加，不叠不串。"""
    kwargs = {
        "model": "qwen-plus",
        "extra_body": {"thinking": {"type": "disabled"}},  # 主模型遗留
        "reasoning_effort": "none",
    }
    apply_reasoning_kwargs(kwargs, "dashscope", "qwen-plus", _fake_client("https://dashscope.aliyuncs.com/compatible-mode/v1"))
    assert kwargs == {"model": "qwen-plus", "extra_body": {"enable_thinking": True}}
