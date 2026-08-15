"""思考（thinking / reasoning）开关按型号分流 —— 照 DeepTutor reasoning_params.py 设计移植。

问题：供应商出厂往往默认开思考（推理模型尤甚），聊天被拖成「先想半分钟再出字」，
思考还占输出额度。DeepTutor 不信出厂默认，按型号自己定策略：

- 快档（deepseek-v4-flash）→ 默认关思考（首字 ~2s 的关键）；
- 会吞正文额度的档（gemini-2.5* / gemini-3*，思考吃光 max_tokens 致 content 空）
  → reasoning_effort=none（API 拒 none 的型号降 minimal）；
- 推理档（deepseek-v4-pro / deepseek-reasoner / qwen3 系 / qwq / qwen-plus）
  → 显式开思考 + high；
- 其它型号 → 不传开关，跟供应商默认。

总闸：settings.llm.reasoning_effort（env ``LLM__REASONING_EFFORT``）。空=按上表自动
分流；``minimal``=全局关；``high``=全局开。设了总闸整表被覆盖。

注入点：core.llm.llm._create_with_image_fallback（全项目主聊天链路的统一咽喉）。
judge / quiz / vision / bot / LightRAG 等自建 client 的路径不经此模块，行为不变。
"""
from __future__ import annotations

import logging
from typing import Any

from settings import get_settings
from core.llm.provider_registry import find_by_base_url, find_by_model, find_by_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 策略表（照 DeepTutor）
# ---------------------------------------------------------------------------

# 默认关思考：快档 + 会吞正文额度的档（两条策略一张表）
_DEFAULT_OFF_PATTERNS: dict[str, tuple[str, ...]] = {
    # 「名字叫快、出厂却开思考」：一开思考 flash 就变 pro 体感（DeepTutor v1.4.0）
    "deepseek": ("deepseek-v4-flash",),
    # 思考吃光 max_tokens 致正文截断/空（DeepTutor issue #489）
    "gemini": ("gemini-2.5", "gemini-3"),
}
# gemini-3* / gemini-2.5-pro 的 API 拒绝 reasoning_effort=none，只能降到 minimal
_GEMINI_MINIMAL_ONLY_PATTERNS: tuple[str, ...] = ("gemini-3", "gemini-2.5-pro")

# 推理档：选它就是为了想——显式开思考 + high
_PROVIDER_REASONING_PATTERNS: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-v4-pro", "deepseek-reasoner"),
    "dashscope": ("qwen3", "qwen-3", "qwq", "qwen-plus"),
}

# 各供应商「关 / 开」两态对应的请求参数（key 与 OpenAI 兼容 create kwargs 对齐）
_DISABLE_KWARGS: dict[str, dict[str, Any]] = {
    "deepseek": {"extra_body": {"thinking": {"type": "disabled"}}},
    "dashscope": {"extra_body": {"enable_thinking": False}},
    "gemini": {"reasoning_effort": "none"},
}
_ENABLE_KWARGS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "reasoning_effort": "high",
        "extra_body": {"thinking": {"type": "enabled"}},
    },
    "dashscope": {"extra_body": {"enable_thinking": True}},
    "gemini": {"reasoning_effort": "high"},
}

# 本模块在 create kwargs 里「持有」的 key：换模型（fallback）时先剥再按新供应商/型号重算
_MANAGED_KEYS: tuple[str, ...] = ("reasoning_effort", "extra_body")


def resolve_provider(
    binding: str | None,
    model: str | None,
    base_url: str | None = None,
) -> str | None:
    """反查实际供应商 canonical name：base_url 域名 → 模型名关键字 → binding。

    binding 是 legacy 配置标签（.env 未设时默认 "dashscope"，实际端点可能是
    deepseek），端点说了算，故 base_url 优先。返回 provider_registry 的 canonical
    name（如 "deepseek"）；三者都查不到返回 None（调用方按「不在表内 → 不传」处理）。
    """
    spec = find_by_base_url(base_url) or find_by_model(model) or find_by_name(binding)
    return spec.name if spec else None


def reasoning_kwargs(provider: str | None, model: str | None) -> dict[str, Any]:
    """按 (provider, model) 算思考参数；返回空 dict = 不传（跟供应商默认）。

    优先级（照 DeepTutor）：总闸覆盖 > 推理档表(开) > 默认关表(关) > 不传。
    """
    if not provider or not model:
        return {}

    effort = (get_settings().llm.reasoning_effort or "").strip().lower()
    if effort == "minimal":
        return _DISABLE_KWARGS.get(provider, {})
    if effort == "high":
        return _ENABLE_KWARGS.get(provider, {})
    if effort:
        logger.warning(
            "reasoning_params: 未知 LLM__REASONING_EFFORT=%r（支持 minimal/high，空=按型号自动），按自动分流处理",
            effort,
        )

    m = model.lower()

    # 先查推理档表：命中就强制开（flash 不在 pro/reasoner 名单，不会误伤）
    patterns = _PROVIDER_REASONING_PATTERNS.get(provider)
    if patterns and any(p in m for p in patterns):
        return _ENABLE_KWARGS[provider]

    # 再查默认关表（minimal-only 型号本身都是 gemini- 前缀，无需再判 provider）
    off_patterns = _DEFAULT_OFF_PATTERNS.get(provider)
    if off_patterns and any(p in m for p in off_patterns):
        if any(p in m for p in _GEMINI_MINIMAL_ONLY_PATTERNS):
            return {"reasoning_effort": "minimal"}
        return _DISABLE_KWARGS[provider]

    return {}


def apply_reasoning_kwargs(
    kwargs: dict[str, Any],
    binding: str | None,
    model: str | None,
    client: object | None = None,
) -> None:
    """就地注入思考参数到 create kwargs（幂等：先剥本模块管理的 key 再重算）。

    client 用于取 base_url 做端点反查（AsyncOpenAI 及其包装均暴露 .base_url；
    取不到时退回模型名/binding 反查）。换模型重调前无需手动清理——本函数先剥后加。
    注意：本函数持有整个 ``extra_body`` key，经此咽喉的调用方若自带 extra_body
    会被剥掉（当前无此调用方）。
    """
    for key in _MANAGED_KEYS:
        kwargs.pop(key, None)
    base_url = str(getattr(client, "base_url", "") or "")
    kwargs.update(reasoning_kwargs(resolve_provider(binding, model, base_url), model))


__all__ = ["resolve_provider", "reasoning_kwargs", "apply_reasoning_kwargs"]
