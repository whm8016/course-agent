"""LLM 成本核算：跨 provider usage 归一 + 价目表 + 成本估算 + 指标埋点。

成本采集是**横切关注点**：所有 LLM 出口（agentic/loop._one_round）经
``usage_from_response_chunk`` 把 provider 各异的 usage 归一为 ``TokenUsage``，
再按 ``data/model_pricing.json`` 价目表算成美元，最后埋进 Prometheus Counter。

【为什么是这里、为什么这么分层】
- OTel GenAI 语义约定（``gen_ai.usage.input_tokens`` 等）是事实标准；成本不由 OTel 存储，
  需应用层按价目表算。命名对齐该约定，零成本对接任意后端。
- ``usage_from_response_chunk`` 一套代码读两类 provider：OpenAI 末块（``choices=[]`` +
  ``chunk.usage``）与 Anthropic 适配器在 ``message_stop`` 合成的**同形态**等价块
  （见 anthropic_adapter._openai_usage_chunk）。loop 侧无需感知 provider 差异。
- 价目表与 ``model_catalog.json`` 同目录同风格，按 mtime 失效重载，热改不重启。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# backend/data/ —— 与 model_catalog.json 同目录
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_PRICING_PATH = _DATA_DIR / "model_pricing.json"


@dataclass
class TokenUsage:
    """一次 LLM 调用的 token 用量（跨 provider 归一后）。

    cache_read_tokens = prompt cache 命中量（OpenAI cached_tokens /
    Anthropic cache_read_input_tokens）。cache 命中的 input 同时计入 input_tokens
    （provider 习惯把命中部分也算在 prompt_tokens 里），成本侧按更便宜的 cache 价档算。
    samples = 累积时采到真实 usage 的次数（add 时 other 非 None 则 +1）——"是否采到过"
    属于累积测量值的属性而非调用方旁路状态；token 全 0 且 samples==0 即缺失（端点不支持
    stream_options 等），samples>0 且全 0 才是真实零成本轮。estimate_cost/record_llm_usage
    无视此字段（带默认值，向后兼容）。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    samples: int = 0

    def add(self, other: TokenUsage | None) -> TokenUsage:
        """返回 self + other 的新实例（other 为 None 时返回 self 的等价副本，samples 不变）。

        samples 纯加法：单次采集对象由 ``_normalize_usage`` 构造时置 samples=1（"我是
        一次采集"），累积器再被 sum() 折叠时不会重复计数。
        """
        if other is None:
            return TokenUsage(self.input_tokens, self.output_tokens, self.cache_read_tokens, self.samples)
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            samples=self.samples + other.samples,
        )

    @classmethod
    def sum(cls, usages) -> TokenUsage:
        """聚合一组 usage（None 元素跳过）。子 pipeline 多 loop 用量汇总的单一真相源。"""
        total = cls()
        for u in usages:
            total = total.add(u)
        return total


def usage_payload(usage: TokenUsage, cost_usd: float | None = None) -> dict[str, Any]:
    """TokenUsage → 前端事件 usage dict（done 事件 / quiz/research result 事件共用）。

    字段清单与前端 MessageUsage（frontend/src/types/index.ts）逐字段对应——三处消费点
    共用本函数，杜绝手写 dict 拷贝漂移。collected 派生自 samples（见 TokenUsage docstring）。
    """
    payload: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "collected": usage.samples > 0,
    }
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd
    return payload


def _normalize_usage(usage: Any) -> TokenUsage | None:
    """把 OpenAI 形态的 usage 对象归一为 TokenUsage；prompt/completion 均缺失返回 None。

    samples=1：归一产物代表一次真实采集（TokenUsage.samples 语义见 dataclass docstring）。
    """
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return TokenUsage(
        input_tokens=int(prompt or 0),
        output_tokens=int(completion or 0),
        cache_read_tokens=int(cached or 0),
        samples=1,
    )


def usage_from_response_chunk(chunk: Any) -> TokenUsage | None:
    """从流式 chunk 归一出 TokenUsage；无 usage 字段返回 None。

    OpenAI：开了 ``stream_options={"include_usage": True}`` 时，末块 ``choices=[]`` 且
    ``chunk.usage = {prompt_tokens, completion_tokens, prompt_tokens_details.cached_tokens}``。
    Anthropic：anthropic_adapter 在 message_stop 合成同形态 chunk。故此处读
    ``.prompt_tokens`` / ``.completion_tokens`` / ``.prompt_tokens_details.cached_tokens``
    即可覆盖两类 provider。

    注意：loop 原本 ``if not choices: continue`` 会把这个空-choices 的 usage 末块跳掉，
    导致 usage 从来抓不到——调用方须在 continue 之前先取 usage（见 loop._one_round）。
    """
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    return _normalize_usage(usage)


def usage_from_response(resp: Any) -> TokenUsage | None:
    """从非流式响应对象归一出 TokenUsage（``resp.usage`` 与流式末块同字段结构）。

    供不走 agent loop 的辅助 LLM 调用点读 usage（judge/L2 摘要/追问/视觉/图谱等，
    经 ``analytics.token_usage.track_llm_response`` 一站式埋点）；Anthropic binding 下
    ``resp.usage`` 由 anthropic_adapter._response_to_openai 合成（同 OpenAI 形态）。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    return _normalize_usage(usage)


# --------------------------------------------------------------------------
# 价目表：按 mtime 失效重载，热改不重启
# --------------------------------------------------------------------------

_pricing_cache: dict[str, Any] = {}
_pricing_mtime: float | None = None
_pricing_missing_warned = False


def _load_pricing() -> dict[str, Any]:
    """读 data/model_pricing.json；按 mtime 失效重载；缺失/损坏返回空表（不抛）。"""
    global _pricing_cache, _pricing_mtime, _pricing_missing_warned
    try:
        mtime = _PRICING_PATH.stat().st_mtime
    except OSError:
        if not _pricing_missing_warned:
            logger.info(
                "LLM 价目表 %s 不存在，成本估算将返回 0（成本指标仍记录 token 数）", _PRICING_PATH
            )
            _pricing_missing_warned = True
        return {}
    if _pricing_mtime == mtime and _pricing_cache:
        return _pricing_cache
    try:
        with _PRICING_PATH.open("r", encoding="utf-8") as f:
            _pricing_cache = json.load(f)
        _pricing_mtime = mtime
        _pricing_missing_warned = False
        logger.info("LLM 价目表已加载：%d 个模型条目", len(_pricing_cache))
    except (OSError, json.JSONDecodeError):
        logger.exception("LLM 价目表加载失败，成本估算返回 0")
        _pricing_cache = {}
    return _pricing_cache


def _lookup_price(model: str | None) -> dict[str, Any] | None:
    """查价：全等优先，其次最长家族前缀命中（deepseek-v4-pro 命中 deepseek-v4 而非 deepseek）。

    家族前缀匹配用于吸收版本号尾缀差异（deepseek-v4-pro / deepseek-v4.1 / deepseek-chat
    都可挂在 deepseek-v4 条目下）。未命中返回 None。
    """
    table = _load_pricing()
    if not table:
        return None
    key = (model or "").strip()
    if key in table:
        return table[key]
    best: dict[str, Any] | None = None
    best_len = 0
    for pk, pv in table.items():
        if pk.startswith("_"):
            continue  # 跳过 _comment 等元字段
        if key.startswith(pk) and len(pk) > best_len:
            best, best_len = pv, len(pk)
    return best


def estimate_cost(model: str | None, usage: TokenUsage | None) -> float:
    """按价目表估算单次调用成本（美元）。

    价目表单位：每 1M token 的美元价，结构
    ``{"<model>": {"input": 1.1, "output": 2.76, "cache_read": 0.28}}``。
    缺字段按 0 处理；模型未命中/无价目表/usage 为 None 均返回 0.0（可观测但不阻塞业务）。

    【扣减公式，勿改回加法】TokenUsage 是 OpenAI 子集语义：cache 命中量已计入
    input_tokens（provider 把命中部分算在 prompt_tokens 里，见 TokenUsage docstring）。
    故命中部分必须先从 input 里扣掉再乘 input 全价，否则同一批 token 会被收两遍钱
    （input 价 + cache 价）——LiteLLM #6215/#11364、Langfuse #12306 同款历史 bug。
    ``min`` 钳制防御语义异常的 provider（cache_read > input 时不让未命中量变负）。
    """
    if usage is None:
        return 0.0
    price = _lookup_price(model)
    if not price:
        return 0.0
    in_per_1m = float(price.get("input", 0) or 0)
    out_per_1m = float(price.get("output", 0) or 0)
    cache_per_1m = float(price.get("cache_read", 0) or 0)
    cache = min(usage.cache_read_tokens, usage.input_tokens)
    cost = (
        (usage.input_tokens - cache) * in_per_1m / 1_000_000
        + usage.output_tokens * out_per_1m / 1_000_000
        + cache * cache_per_1m / 1_000_000
    )
    return round(cost, 6)


# --------------------------------------------------------------------------
# 指标埋点：惰性 import metrics（cost 是 metrics 的消费方，反向依赖用函数内 import 避免
# 模块加载顺序问题；metrics 叶子模块不感知 cost 类型）
# --------------------------------------------------------------------------

def observe_usage(model: str, usage: TokenUsage | None) -> None:
    """逐轮埋 token 维度 Counter（ca_llm_tokens_total{model,token_type}）。usage 为 None 跳过。"""
    if usage is None:
        return
    from core.observability.metrics import LLM_TOKENS_TOTAL
    if usage.input_tokens:
        LLM_TOKENS_TOTAL.labels(model=model, token_type="input").inc(usage.input_tokens)
    if usage.output_tokens:
        LLM_TOKENS_TOTAL.labels(model=model, token_type="output").inc(usage.output_tokens)
    if usage.cache_read_tokens:
        LLM_TOKENS_TOTAL.labels(model=model, token_type="cache_read").inc(usage.cache_read_tokens)


def observe_cost(model: str, course_id: Any, mode: Any, cost_usd: float) -> None:
    """按 loop 汇总埋 cost 维度 Counter（ca_llm_cost_usd_total{model,course_id,mode}）。cost<=0 跳过。"""
    if cost_usd <= 0:
        return
    from core.observability.metrics import LLM_COST_USD_TOTAL
    LLM_COST_USD_TOTAL.labels(
        model=model,
        course_id=str(course_id or ""),
        mode=str(mode or ""),
    ).inc(cost_usd)


def observe_usage_missing(model: str | None, reason: str) -> None:
    """真实 usage 缺失埋 Counter（ca_llm_usage_missing_total{model,reason}）。

    缺失轮 token 照记 0（账目 append-only，缺失不补估算——估算值会污染价目快照账本与
    quota 决策），但缺失本身必须可观测：缺失率突增 = 端点行为变化信号。

    reason 目前只有 stream_options_unsupported（端点不支持被剥参重试，llm.py 与
    question/agent_base.py 两个剥参点打）。原因不明的缺失（流尾无 usage 块）**不打本
    Counter**——同一轮若已被剥参打点会双计；它经 TokenUsage.samples=0（loop done 的
    collected=False / log_flow usage_collected）可观测。新增确知原因的缺失点时在此登记。
    """
    from core.observability.metrics import LLM_USAGE_MISSING_TOTAL
    LLM_USAGE_MISSING_TOTAL.labels(model=str(model or ""), reason=reason).inc()
