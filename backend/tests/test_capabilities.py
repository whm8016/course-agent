"""core/llm/capabilities.py supports_vision 能力表测试。

能力表驱动 Stage-2 失败降级：模型不在 vision 白名单时，调用失败才剥图重试。
"""
from __future__ import annotations

from core.llm.capabilities import supports_vision


def test_vision_models_supported():
    assert supports_vision("dashscope", "qwen-vl-plus") is True
    assert supports_vision("dashscope", "qwen2.5-vl-72b") is True
    assert supports_vision("openai", "gpt-4o") is True
    assert supports_vision("anthropic", "claude-3-5-sonnet") is True
    assert supports_vision("gemini", "gemini-2.0-flash") is True


def test_text_models_not_vision():
    assert supports_vision("dashscope", "qwen-plus") is False
    assert supports_vision("dashscope", "qwen-turbo") is False
    assert supports_vision("deepseek", "deepseek-chat") is False
    assert supports_vision("openai", "gpt-3.5-turbo") is False


def test_provider_default_when_no_model_override():
    # anthropic provider 默认 True（claude 系列支持 vision）
    assert supports_vision("anthropic", "claude-something-new") is True
    # deepseek provider 默认 False
    assert supports_vision("deepseek", "unknown-deepseek-model") is False


def test_unknown_provider_defaults_false():
    assert supports_vision("unknown_binding", "whatever") is False


def test_qwen_prefix_specificity():
    """qwen-vl 前缀长于 qwen，降序匹配时 qwen-vl-plus 命中 True 而非 qwen 的 False。"""
    assert supports_vision("dashscope", "qwen-vl-plus") is True
    assert supports_vision("dashscope", "qwen-plus") is False
