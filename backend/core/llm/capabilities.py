"""Provider/model 能力注册表。

核心用途：``supports_vision(binding, model)`` 驱动 Stage-2 失败降级——
LLM 调用失败时，若模型不支持 vision 且消息里带了图，就剥掉图片用**同一模型**
重试纯文本（而不是像早期实现那样「有图就硬切 VISION_MODEL」）。

chat 主路径始终用配置的 chat model（TEXT_MODEL），图片乐观注入，不预先 gate；
vision 能力只用在 RAG ingestion（image_extractor）等需要的地方。

查表优先级（get_capability，一致）：
1. MODEL_OVERRIDES —— model 前缀匹配，按 pattern 长度**降序**（最具体的先命中，
   故 "qwen-vl-plus" 命中 "qwen-vl"(True) 而非 "qwen"(False)）
2. PROVIDER_CAPABILITIES[binding]
3. DEFAULT_CAPABILITIES
"""
from __future__ import annotations

# provider 级默认能力（当前只消费 supports_vision）
PROVIDER_CAPABILITIES: dict[str, dict[str, object]] = {
    "openai": {"supports_vision": True},
    "azure_openai": {"supports_vision": True},
    "anthropic": {"supports_vision": True},     # Claude 3+
    "gemini": {"supports_vision": True},
    "openrouter": {"supports_vision": True},    # 网关，依赖底层模型
    "groq": {"supports_vision": False},
    "deepseek": {"supports_vision": False},
    "dashscope": {"supports_vision": False},    # per-model：qwen-vl-* 经 override 置 True
    "zhipuai": {"supports_vision": False},      # per-model：glm-4v
    "moonshot": {"supports_vision": False},     # per-model：*-vision
    "siliconflow": {"supports_vision": False},
    "ollama": {"supports_vision": False},
    "vllm": {"supports_vision": False},
    "lm_studio": {"supports_vision": False},
}

DEFAULT_CAPABILITIES: dict[str, object] = {
    "supports_vision": False,
}

# model 级 override（大小写不敏感前缀匹配；get_capability 按 pattern 长度降序）
MODEL_OVERRIDES: dict[str, dict[str, object]] = {
    # ── 支持 vision ───────────────────────────────────────────────────────
    "gpt-4o": {"supports_vision": True},
    "gpt-4.1": {"supports_vision": True},
    "gpt-4-turbo": {"supports_vision": True},
    "gpt-4-vision": {"supports_vision": True},
    "claude-3": {"supports_vision": True},
    "claude-4": {"supports_vision": True},
    "claude-sonnet": {"supports_vision": True},
    "claude-opus": {"supports_vision": True},
    "claude-haiku": {"supports_vision": True},
    "gemini": {"supports_vision": True},
    "qwen-vl": {"supports_vision": True},
    "qwen2-vl": {"supports_vision": True},
    "qwen2.5-vl": {"supports_vision": True},
    "qwen3-vl": {"supports_vision": True},
    "qwen/qwen-vl": {"supports_vision": True},
    "qwen/qwen2-vl": {"supports_vision": True},
    "qwen/qwen2.5-vl": {"supports_vision": True},
    "qwen/qwen3-vl": {"supports_vision": True},
    "glm-4v": {"supports_vision": True},
    "moonshot-v1-8k-vision": {"supports_vision": True},
    "moonshot-v1-32k-vision": {"supports_vision": True},
    "moonshot-v1-128k-vision": {"supports_vision": True},
    "kimi-k2": {"supports_vision": True},
    "llava": {"supports_vision": True},
    "minicpm-v": {"supports_vision": True},
    # ── 明确不支持 vision（覆盖 provider 默认 True）─────────────────────
    "gpt-3.5": {"supports_vision": False},
    "deepseek": {"supports_vision": False},
    # 纯文本 qwen-plus/turbo/max：前缀短于 qwen-vl，降序匹配时 qwen-vl 先命中
    "qwen": {"supports_vision": False},
    "qwq": {"supports_vision": False},
}


def get_capability(
    binding: str,
    capability: str,
    model: str | None = None,
    default: object = None,
) -> object:
    """查能力值。优先级：model override（前缀长度降序）→ provider → default。"""
    binding_lower = (binding or "openai").lower()

    # 1. model override（最具体前缀优先）
    if model:
        model_lower = model.lower()
        for pattern, overrides in sorted(MODEL_OVERRIDES.items(), key=lambda x: -len(x[0])):
            if model_lower.startswith(pattern) and capability in overrides:
                return overrides[capability]

    # 2. provider 级
    provider_caps = PROVIDER_CAPABILITIES.get(binding_lower, {})
    if capability in provider_caps:
        return provider_caps[capability]

    # 3. 默认
    if capability in DEFAULT_CAPABILITIES:
        return DEFAULT_CAPABILITIES[capability]

    return default


def supports_vision(binding: str | None, model: str | None = None) -> bool:
    """模型是否支持图片输入（Stage-2 降级决策依据）。"""
    return bool(get_capability(binding or "openai", "supports_vision", model, default=False))
