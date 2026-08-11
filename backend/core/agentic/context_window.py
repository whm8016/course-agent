"""有效上下文窗口解析 + 双阈值预算计算（通用上下文管理重构核心）。

四级窗口解析（对标 DeepTutor ``resolve_effective_context_window`` + When Refusals Fail 论文
arXiv:2512.02445「标称窗口不可信」）：

  1. 显式配置：``settings.llm.context_window``（全局覆盖，catalog / .env 透传）
  2. 探测缓存：``data/context_window_cache.json`` 里由启动预热/admin 重探写入的真实值
     （``window_probe.read_probe_cache``，热路径同步读、零网络；详见 core.agentic.window_probe）
  3. 模型名模式：``_MODEL_WINDOWS`` 已知模型表（从 context_builder 迁入，探测拿不到时兜底）
  4. heuristic 兜底：``max(16384, max_tokens × 4)``；命中即 ``log_flow`` 告警，让运维能发现
     漏配，杜绝旧实现「未知模型静默按 32768 跑」的失效模式。

默认行为零变化：探测缓存层在「从未跑过探测」时恒返回 None，解析退到第 3/4 级--与改造前
逐字节一致。只有当启动预热或 admin 重探真的写入缓存后，第 2 级才生效（这正是探测的意义：
用供应商真实窗口替换过期的硬编码表）。

双阈值预算（硬天花板减法留白，对标 Claude Code ``contextWindow - min(maxOutputTokens, 20k) - safety``）：

  硬天花板 = effective_window - min(max_output, output_reserve_tokens) - safety_margin_tokens
  软阈值   = min(rot_threshold_tokens, effective_window * quality_ratio, 硬天花板)

两条线分工：软阈值驱动主动压缩追求质量（窗口比例线 quality_ratio，RULER 实测有效长度多为窗口
0.25-0.5，取 0.5；rot_threshold_tokens 作为绝对上限在比例线算出过大时钳制，rot 论文 64k 为成本
可接受上界），硬天花板驱动紧急降级保证可用（绝不让请求被 API 拒绝）。预留量是绝对量，不随窗口
缩放--这是减法留白的核心理由。

本模块放 ``core/agentic`` 而非 ``services/session``，因为它需要 import settings，而
``context_builder.py`` 目前不依赖 settings，放这里避免给 context_builder 引入 settings 依赖
（context_builder 仅委托本模块的解析能力，自身仍不直接 import settings）。
"""
from __future__ import annotations

import logging

from core.agentic.window_probe import read_probe_cache
from core.observability import log_flow
from settings import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# token 计数（tiktoken 精确，降级 len//4）。从 services/session/context_builder 迁入本模块，
# 治 core/agentic 反向 import services 的层级违规（context_budget/context_policy 都要用
# count_tokens）。context_builder 现反向 import 本函数（services -> core 正向，无循环）。
_encoding = None


def _get_encoding():
    global _encoding
    if _encoding is not None:
        return _encoding
    try:
        import tiktoken
        _encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoding = False  # 标记为不可用
    return _encoding


def count_tokens(text: str) -> int:
    """tiktoken 精确计数，不可用时降级到 len // 4。"""
    if not text:
        return 0
    enc = _get_encoding()
    if enc:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 模型名 -> 标称上下文窗口（token）。作为四级解析的第 3 级（精确已知，探测/显式配置之后）。
# 未知模型不走此表，走 heuristic 兜底并告警（见 resolve_effective_window）。
#
# 这是「兜底表」不是权威源——模型迭代远快于代码发版，任何硬编码表上线即开始过期
# （例如旧表把 deepseek-chat 记作 65536，实际它已 route 到 V4-Flash 的 1M 窗口，
# 误差 15 倍）。数值截至 2026-08 各厂商官方文档/发布页，需定期核对刷新；生产环境
# 优先用 settings.llm.context_window 显式配置，不要依赖本表的准确性。
# ---------------------------------------------------------------------------
_MODEL_WINDOWS: dict[str, int] = {
    # --- 阿里云百炼 / DashScope（Qwen 系列，https://help.aliyun.com/zh/model-studio） ---
    "qwen-max": 32_768,
    "qwen-max-latest": 32_768,
    "qwen-plus": 1_000_000,
    "qwen-plus-latest": 1_000_000,
    "qwen-turbo": 131_072,
    "qwen-turbo-latest": 131_072,
    "qwen-flash": 1_000_000,
    "qwen-long": 1_000_000,
    "qwen3-max": 262_144,
    "qwen3.7-max": 1_000_000,
    "qwen3.7-plus": 1_000_000,
    "qwen3-235b-a22b": 131_072,
    "qwen3-32b": 131_072,
    "qwen3-30b-a3b": 131_072,
    "qwen3.5-397b-a17b": 262_144,
    "qwen3.5-27b": 262_144,

    # --- DeepSeek（deepseek-chat/-reasoner 现路由至 V4-Flash，https://api-docs.deepseek.com） ---
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v3.2": 128_000,   # 旧快照，仍可能被 pin 调用
    "deepseek-v3": 64_000,

    # --- OpenAI（GPT 系列） ---
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "gpt-5.4": 1_000_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.5": 1_000_000,
    "o1": 200_000,
    "o3": 200_000,
    "o3-mini": 200_000,

    # --- Anthropic Claude（经 OpenAI 兼容网关/代理接入场景） ---
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-fable-5": 1_000_000,

    # --- Google Gemini ---
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-3.1-pro": 1_000_000,
    "gemini-3.5-flash": 1_000_000,

    # --- Moonshot / Kimi ---
    "moonshot-v1-128k": 128_000,
    "kimi-k2": 128_000,
    "kimi-k3": 1_048_576,

    # --- 智谱 GLM ---
    "glm-4": 128_000,
    "glm-4.5": 128_000,
    "glm-4.7": 200_000,

    # --- Meta Llama / Mistral（自托管 / 网关常见别名） ---
    "llama-3.1-70b": 128_000,
    "llama-3.3-70b": 128_000,
    "llama-4-scout": 10_000_000,   # Meta 官方标称 10M；实际有效窗口远低于此，
                                    # rot_threshold（软阈值）独立兜底质量风险，不依赖标称值
    "llama-4-maverick": 1_000_000,
    "mistral-large": 128_000,
}

# heuristic 兜底参数（对标 DeepTutor default_context_window_for_model）
_HEURISTIC_FLOOR = 16384        # 未知模型窗口下界（太小连 system prompt 都放不下）
_HEURISTIC_MAX_TOKENS_MULT = 4  # 输出上限的倍数估算窗口（max_tokens=8192 -> 32768，与旧 _DEFAULT_WINDOW 一致）


def resolve_effective_window_with_source(
    model: str | None, base_url: str | None = None
) -> tuple[int, str]:
    """四级解析模型有效上下文窗口（token），返回 (窗口, 来源)。

    优先级：显式配置(settings.llm.context_window) -> 探测缓存(window_probe.read_probe_cache)
    -> 模型名模式(_MODEL_WINDOWS) -> heuristic(max(16384, max_tokens×4))。命中 heuristic 时
    log_flow 告警（杜绝静默回退）。

    ``base_url`` 默认取 ``settings.llm.base_url``（启动 active profile）；admin 重探端点传当前
    active profile 的 base_url，使探测缓存的键与刚写入的探测结果对齐。热路径调用方不传
    base_url（用默认值），签名仍只要求 model。

    返回值即「标称有效窗口」--不做 MECW 折算。质量风险由软阈值(rot_threshold_tokens)独立
    控制（rot 论文实证标称窗口不可信，故用独立绝对阈值而非标称打折）。来源供 admin 端点展示。
    """
    cfg = get_settings()
    # 1. 显式配置（catalog / .env 透传到 settings.llm.context_window）
    explicit = cfg.llm.context_window
    if explicit and explicit > 0:
        return int(explicit), "explicit"
    # 2. 探测缓存（启动预热/admin 重探写入的真实窗口；热路径同步读，零网络）
    burl = base_url if base_url is not None else cfg.llm.base_url
    probed = read_probe_cache(burl, model or "")
    if probed and probed > 0:
        return int(probed), "probe"
    # 3. 模型名模式匹配
    key = (model or "").strip()
    if key in _MODEL_WINDOWS:
        return _MODEL_WINDOWS[key], "table"
    # 4. heuristic 兜底 + 告警
    max_tok = cfg.llm.max_tokens or 4096
    heuristic = max(_HEURISTIC_FLOOR, max_tok * _HEURISTIC_MAX_TOKENS_MULT)
    log_flow(
        "context.window_heuristic_fallback",
        logger=logger,
        level=logging.WARNING,
        model=key or "<none>",
        heuristic_window=heuristic,
        hint="模型未在 _MODEL_WINDOWS 登记、且未显式配 context_window，请补登或设 LLM__CONTEXT_WINDOW",
    )
    return heuristic, "heuristic"


def resolve_effective_window(model: str | None) -> int:
    """四级解析模型有效上下文窗口（token）。保留单参签名不破坏调用方。

    等价于 ``resolve_effective_window_with_source(model)[0]``；需要来源（admin 诊断）的调用方
    直接用 with_source 变体。
    """
    return resolve_effective_window_with_source(model)[0]


def compute_budgets(model: str | None) -> tuple[int, int]:
    """计算双阈值预算 (soft_trigger, hard_ceiling)（token，减法留白）。

    hard_ceiling = effective_window - min(max_output, output_reserve_tokens) - safety_margin_tokens
    soft_trigger = min(rot_threshold_tokens, int(effective_window * quality_ratio), hard_ceiling)

    hard_ceiling 是安全线（绝不让请求被 API 拒绝）：从窗口扣除输出预留与安全余量。
    soft_trigger 是质量线（超了才主动压缩）：窗口比例线（quality_ratio，RULER 实测有效长度多为窗口
    0.25-0.5，取 0.5）与绝对上限 rot_threshold_tokens 取 min，再被 hard_ceiling 钳制。soft <= hard
    由 min 保证。
    """
    cfg = get_settings()
    window = resolve_effective_window(model)
    reserve = min(cfg.llm.max_tokens or 4096, cfg.context_budget.output_reserve_tokens)
    hard_ceiling = window - reserve - cfg.context_budget.safety_margin_tokens
    soft_trigger = min(
        cfg.context_budget.rot_threshold_tokens,
        int(window * cfg.context_budget.quality_ratio),
        hard_ceiling,
    )
    # 兜底：极小窗口模型算出负/零预算时，给一个最小可用值，避免「预算=0 -> 永远触发压缩」死循环。
    if soft_trigger < 1:
        soft_trigger = max(1, hard_ceiling)
    if hard_ceiling < 1:
        hard_ceiling = soft_trigger
    return soft_trigger, hard_ceiling


__all__ = ["resolve_effective_window", "resolve_effective_window_with_source", "compute_budgets"]
