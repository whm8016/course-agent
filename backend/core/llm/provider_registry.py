"""LLM provider registry — single source of truth for all supported providers.

每条 ProviderSpec 描述一个供应商：名称、底层 backend 类型、默认 API 端点、
可用于从模型名推断供应商的关键字。绝大多数云厂商走 openai_compat backend，
只有 Anthropic（原生 SDK）和 Azure OpenAI 需要单独处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    """描述一个 LLM 供应商的元数据。"""

    name: str
    """Canonical 供应商 ID，如 "dashscope"。binding 字段应与此对齐。"""

    backend: str
    """底层实现类型：
    - "openai_compat"  — 用 AsyncOpenAI(base_url=...) 接入
    - "anthropic"      — 用原生 anthropic SDK + 适配器
    - "azure_openai"   — 用 AsyncAzureOpenAI
    """

    default_api_base: str | None = None
    """默认 API 端点；用户未填 base_url 时使用此值。None 表示用 SDK 默认。"""

    env_key: str = ""
    """推荐的 API key 环境变量名（仅用于文档/提示）。"""

    keywords: tuple[str, ...] = field(default_factory=tuple)
    """用于从模型名推断供应商的子串，全小写匹配。"""


# ---------------------------------------------------------------------------
# 全量供应商注册表
# ---------------------------------------------------------------------------

PROVIDERS: list[ProviderSpec] = [
    # ---- OpenAI 官方 -------------------------------------------------------
    ProviderSpec(
        name="openai",
        backend="openai_compat",
        default_api_base=None,  # SDK 默认 api.openai.com
        env_key="OPENAI_API_KEY",
        keywords=("gpt-", "o1", "o3", "o4", "text-davinci", "whisper"),
    ),
    # ---- DeepSeek ----------------------------------------------------------
    ProviderSpec(
        name="deepseek",
        backend="openai_compat",
        default_api_base="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        keywords=("deepseek",),
    ),
    # ---- 阿里云 DashScope（通义 Qwen）--------------------------------------
    ProviderSpec(
        name="dashscope",
        backend="openai_compat",
        default_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        keywords=("qwen",),
    ),
    # ---- 智谱 AI（GLM）-----------------------------------------------------
    ProviderSpec(
        name="zhipuai",
        backend="openai_compat",
        default_api_base="https://open.bigmodel.cn/api/paas/v4",
        env_key="ZHIPUAI_API_KEY",
        keywords=("glm-", "chatglm"),
    ),
    # ---- Moonshot（Kimi）--------------------------------------------------
    ProviderSpec(
        name="moonshot",
        backend="openai_compat",
        default_api_base="https://api.moonshot.cn/v1",
        env_key="MOONSHOT_API_KEY",
        keywords=("moonshot", "kimi"),
    ),
    # ---- Groq --------------------------------------------------------------
    ProviderSpec(
        name="groq",
        backend="openai_compat",
        default_api_base="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        keywords=("llama", "mixtral", "gemma", "groq"),
    ),
    # ---- Google Gemini（OpenAI 兼容端点）-----------------------------------
    ProviderSpec(
        name="gemini",
        backend="openai_compat",
        default_api_base="https://generativelanguage.googleapis.com/v1beta/openai",
        env_key="GEMINI_API_KEY",
        keywords=("gemini",),
    ),
    # ---- OpenRouter（多模型网关）-------------------------------------------
    ProviderSpec(
        name="openrouter",
        backend="openai_compat",
        default_api_base="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        keywords=(),
    ),
    # ---- SiliconFlow（硅基流动）--------------------------------------------
    ProviderSpec(
        name="siliconflow",
        backend="openai_compat",
        default_api_base="https://api.siliconflow.cn/v1",
        env_key="SILICONFLOW_API_KEY",
        keywords=(),
    ),
    # ---- Ollama（本地）-----------------------------------------------------
    ProviderSpec(
        name="ollama",
        backend="openai_compat",
        default_api_base="http://localhost:11434/v1",
        env_key="",
        keywords=(),
    ),
    # ---- vLLM（本地 / 私有部署）-------------------------------------------
    ProviderSpec(
        name="vllm",
        backend="openai_compat",
        default_api_base="http://localhost:8000/v1",
        env_key="",
        keywords=(),
    ),
    # ---- LM Studio（本地）-------------------------------------------------
    ProviderSpec(
        name="lm_studio",
        backend="openai_compat",
        default_api_base="http://localhost:1234/v1",
        env_key="",
        keywords=(),
    ),
    # ---- Anthropic（原生 SDK）----------------------------------------------
    ProviderSpec(
        name="anthropic",
        backend="anthropic",
        default_api_base=None,  # SDK 默认 api.anthropic.com
        env_key="ANTHROPIC_API_KEY",
        keywords=("claude",),
    ),
    # ---- Azure OpenAI ------------------------------------------------------
    ProviderSpec(
        name="azure_openai",
        backend="azure_openai",
        default_api_base=None,  # 必须由用户配置 azure_endpoint
        env_key="AZURE_OPENAI_API_KEY",
        keywords=(),
    ),
]

# binding 别名映射（用户可能填的非 canonical 名称）
_ALIASES: dict[str, str] = {
    "azure": "azure_openai",
    "claude": "anthropic",
    "google": "gemini",
    "qwen": "dashscope",
    "glm": "zhipuai",
    "kimi": "moonshot",
}

_BY_NAME: dict[str, ProviderSpec] = {p.name: p for p in PROVIDERS}


# ---------------------------------------------------------------------------
# 查找函数
# ---------------------------------------------------------------------------

def find_by_name(binding: str | None) -> ProviderSpec | None:
    """按 canonical name 或别名查找 ProviderSpec；未找到返回 None。"""
    if not binding:
        return None
    canonical = _ALIASES.get(binding.lower(), binding.lower())
    return _BY_NAME.get(canonical)


def find_by_model(model: str | None) -> ProviderSpec | None:
    """从模型名关键字推断 ProviderSpec；无法推断时返回 None。"""
    if not model:
        return None
    lower = model.lower()
    for spec in PROVIDERS:
        for kw in spec.keywords:
            if kw in lower:
                return spec
    return None


__all__ = ["ProviderSpec", "PROVIDERS", "find_by_name", "find_by_model"]
