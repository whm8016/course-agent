"""context_window 四级解析 + compute_budgets 双阈值测试。

覆盖：
- resolve_effective_window / resolve_effective_window_with_source：
  显式配置 > 探测缓存 > 模型名模式 > heuristic 兜底（+告警）
- compute_budgets：减法留白公式、soft<=hard 不变式、小窗口模型 soft 被硬天花板钳住

探测缓存层用 autouse fixture 隔离到 tmp_path，绝不读真实 data/context_window_cache.json，
保证 table/heuristic/explicit 用例不受探测层干扰。
"""
from __future__ import annotations

import json
import logging

import pytest

from core.agentic import context_window, window_probe
from core.agentic.context_window import (
    compute_budgets,
    count_tokens_messages,
    resolve_effective_window,
    resolve_effective_window_with_source,
)


@pytest.fixture(autouse=True)
def _isolate_probe_cache(monkeypatch, tmp_path):
    """把探测缓存指向 tmp_path（空文件），清进程内缓存。

    使真实 read_probe_cache 恒返回 None，让下方 table/heuristic/explicit 用例只测各自层级。
    需要测探测层的用例显式调 write_probe_cache 写入此 tmp 文件。
    """
    cache_file = tmp_path / "context_window_cache.json"
    monkeypatch.setattr(window_probe, "_cache_path", lambda: str(cache_file))
    window_probe._reset_mem_cache()


# ---------------------------------------------------------------------------
# resolve_effective_window 四级解析
# ---------------------------------------------------------------------------

def test_explicit_config_overrides_pattern(monkeypatch):
    """显式 settings.llm.context_window 优先于模型名模式匹配。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", 200000)
    assert resolve_effective_window("qwen-plus") == 200000  # 不是表里的 1_000_000


def test_model_name_pattern_for_known_models():
    """已知模型走 _MODEL_WINDOWS 精确窗口（非 heuristic，无告警）。"""
    assert resolve_effective_window("qwen-plus") == 1_000_000
    assert resolve_effective_window("deepseek-chat") == 1_000_000
    assert resolve_effective_window("qwen-max") == 32768


def test_heuristic_fallback_for_unknown_model_warns(monkeypatch, caplog):
    """未知模型走 heuristic（max(16384, max_tokens×4)）并 log_flow 告警，杜绝静默回退。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    with caplog.at_level(logging.WARNING, logger="core.agentic.context_window"):
        win = resolve_effective_window("some-brand-new-model")
    assert win == 32768  # max(16384, 8192*4)
    assert any("heuristic" in r.message.lower() or "window" in r.message.lower()
               for r in caplog.records if r.levelno >= logging.WARNING) or \
        any("heuristic" in str(r).lower() for r in caplog.records)


def test_heuristic_floor_when_tiny_max_tokens(monkeypatch):
    """max_tokens 极小时 heuristic 不低于 16384 下界。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 1024)
    assert resolve_effective_window("unknown-tiny") == 16384  # max(16384, 1024*4=4096)


def test_none_model_falls_to_heuristic(monkeypatch):
    """model=None（未传模型名）走 heuristic。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    # 钉住 max_tokens（默认已提至 16384，heuristic=max(16384, mt*4)）——测公式不测默认值
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    assert resolve_effective_window(None) == 32768


# ---------------------------------------------------------------------------
# 探测缓存层（第 2 级）
# ---------------------------------------------------------------------------

def test_probe_cache_used_when_no_explicit(monkeypatch):
    """探测缓存命中时，优先于模型名表（显式配置未设的前提下）。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    base_url = context_window.get_settings().llm.base_url
    # qwen-plus 表里是 1_000_000；探测写入 999_999 应胜出
    window_probe.write_probe_cache(base_url, "qwen-plus", 999_999)
    window_probe._reset_mem_cache()  # 强制下次读盘
    assert resolve_effective_window("qwen-plus") == 999_999


def test_explicit_beats_probe_cache(monkeypatch):
    """显式配置（第 1 级）优先于探测缓存（第 2 级）。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", 200000)
    base_url = context_window.get_settings().llm.base_url
    window_probe.write_probe_cache(base_url, "qwen-plus", 999_999)
    window_probe._reset_mem_cache()
    assert resolve_effective_window("qwen-plus") == 200000


def test_table_used_when_probe_cache_miss(monkeypatch):
    """探测缓存未命中（无条目）时退回模型名表（第 3 级）。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    # 不写探测缓存 -> read_probe_cache 返回 None -> 走表
    assert resolve_effective_window("qwen-plus") == 1_000_000


# ---------------------------------------------------------------------------
# resolve_effective_window_with_source（来源诊断，供 admin 端点）
# ---------------------------------------------------------------------------

def test_with_source_explicit(monkeypatch):
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", 200000)
    val, src = resolve_effective_window_with_source("qwen-plus")
    assert (val, src) == (200000, "explicit")


def test_with_source_probe(monkeypatch):
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    base_url = context_window.get_settings().llm.base_url
    window_probe.write_probe_cache(base_url, "qwen-plus", 999_999)
    window_probe._reset_mem_cache()
    val, src = resolve_effective_window_with_source("qwen-plus")
    assert (val, src) == (999_999, "probe")


def test_with_source_table(monkeypatch):
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    val, src = resolve_effective_window_with_source("qwen-plus")
    assert (val, src) == (1_000_000, "table")


def test_with_source_heuristic(monkeypatch):
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", None)
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    val, src = resolve_effective_window_with_source("some-brand-new-model")
    assert (val, src) == (32768, "heuristic")


def test_with_source_explicit_beats_probe(monkeypatch):
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", 200000)
    base_url = context_window.get_settings().llm.base_url
    window_probe.write_probe_cache(base_url, "qwen-plus", 999_999)
    window_probe._reset_mem_cache()
    val, src = resolve_effective_window_with_source("qwen-plus")
    assert src == "explicit"  # 显式优先


# ---------------------------------------------------------------------------
# compute_budgets 双阈值（减法留白）
# ---------------------------------------------------------------------------

def test_compute_budgets_qwen_plus(monkeypatch):
    """大窗口模型：soft=rot上限(128k)（比例线 500k > 128k 被上限钳住），hard=窗口-reserve-safety。"""
    # 钉住 max_tokens（默认已提至 16384）——测公式与 settings 的关系，不锁默认值
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    soft, hard = compute_budgets("qwen-plus")
    # window=1_000_000, reserve=min(8192,20000)=8192, safety=4096 -> hard=987712
    assert hard == 1_000_000 - 8192 - 4096
    assert soft == 128000  # min(128000, 1_000_000*0.5=500000, 987712)
    assert soft <= hard


def test_compute_budgets_small_window_soft_bound_by_ratio(monkeypatch):
    """小窗口模型：比例线(window*0.5) < 硬天花板 -> soft 被比例线钳住（更保守，符合 RULER 小窗口有效长度低）。"""
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    # qwen-max window=32768 -> hard=32768-8192-4096=20480, 比例线=16384, soft=min(128000,16384,20480)=16384
    soft, hard = compute_budgets("qwen-max")
    assert hard == 20480
    assert soft == 16384  # 比例线生效（< 硬天花板）
    assert soft <= hard


def test_compute_budgets_soft_never_exceeds_hard():
    """不变式：任意已知模型 soft <= hard。"""
    for model in ("qwen-plus", "qwen-max", "deepseek-chat", "qwen-long", "gpt-4o"):
        soft, hard = compute_budgets(model)
        assert soft <= hard, f"{model}: soft={soft} > hard={hard}"
        assert soft >= 1 and hard >= 1


def test_compute_budgets_explicit_window(monkeypatch):
    """显式 context_window 覆盖后，预算基于该窗口计算。"""
    monkeypatch.setattr(context_window.get_settings().llm, "context_window", 200000)
    monkeypatch.setattr(context_window.get_settings().llm, "max_tokens", 8192)
    soft, hard = compute_budgets("anything")
    assert hard == 200000 - 8192 - 4096
    assert soft == 100000  # min(128000, 200000*0.5=100000, 187712) 比例线生效


def test_compute_budgets_128k_model_soft_identity_64000():
    """128k 级模型：比例线=64000 恰好与旧默认值逐字相同（回归保护，主力模型零行为变化）。"""
    soft, hard = compute_budgets("gpt-4o")  # window=128000
    # ratio line = 128000*0.5 = 64000, rot=128000, hard=115712 -> soft=64000
    assert soft == 64000
    assert soft <= hard


def test_compute_budgets_quality_ratio_one_degrades_to_two_way_min(monkeypatch):
    """quality_ratio=1.0：比例线=满窗口恒 >= hard_ceiling，三取 min 退化为旧 min(rot, hard)。"""
    monkeypatch.setattr(context_window.get_settings().context_budget, "quality_ratio", 1.0)
    soft, hard = compute_budgets("qwen-plus")
    # window=1_000_000, hard=987712, 比例线=1_000_000（最大，不约束）, rot=128000
    assert soft == min(128000, hard)  # 退化为两取 min（比例线不约束）
    assert soft == 128000


# ---------------------------------------------------------------------------
# count_tokens_messages（messages 级统一估算器，对标 LiteLLM token_counter 主循环）
# ---------------------------------------------------------------------------

def test_msg_overhead_tokens_added():
    """每条消息 +3 结构开销：10 条短消息总数 = 内容 token + 30（锚定 _MSG_OVERHEAD_TOKENS）。"""
    msgs = [{"role": "user", "content": "hi"} for _ in range(10)]
    body = context_window.count_tokens("hi") * 10
    assert count_tokens_messages(msgs) == body + 10 * 3


def test_name_field_overhead():
    """有 name 字段的消息比无 name 多 1（对标 LiteLLM tokens_per_name）。"""
    plain = count_tokens_messages([{"role": "user", "content": "hi"}])
    named = count_tokens_messages([{"role": "user", "name": " searcher", "content": "hi"}])
    assert named == plain + 1


def test_tool_calls_counted_as_json():
    """tool_calls 按发给 API 的 JSON 序列化计入（保留旧口径，比 LiteLLM 仅数 arguments 更保守）。"""
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "search", "arguments": "{\"q\": \"牛顿第二定律\"}"}}]
    msgs = [{"role": "assistant", "content": "", "tool_calls": tcs}]
    assert count_tokens_messages(msgs) == context_window.count_tokens(
        json.dumps(tcs, ensure_ascii=False)) + 3


def test_multimodal_image_counted_as_constant_not_base64():
    """带图消息：image part 记常数而非 base64 全文编码。

    修 str(content) 爆表 bug：旧 _messages_tokens 对 list content 做 str() 会把
    1MB base64 编出数十万 token，带图轮必然误判超阈值、触发过度裁剪。
    """
    big_b64 = "A" * 1_000_000
    msgs_big = [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
    ]}]
    total = count_tokens_messages(msgs_big)
    assert total < 5_000  # 旧口径此消息 > 25 万 token

    small_b64 = "B" * 1_000
    msgs_small = [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{small_b64}"}},
    ]}]
    # 常数口径：两张不同大小的图计数相同
    assert count_tokens_messages(msgs_small) == total
    # Anthropic 线上形态 {"type":"image","source":...} 同常数（prepare_multimodal_messages
    # 在 _build_messages 阶段按 binding 注入，enforce 估算时可能是任一形态）
    msgs_anthropic = [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": big_b64}},
    ]}]
    assert count_tokens_messages(msgs_anthropic) == total
    # 图片常数本身可叠加：两张图 = 两份常数
    msgs_two = [{"role": "user", "content": [
        {"type": "text", "text": "描述这张图"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big_b64}"}},
    ]}]
    assert count_tokens_messages(msgs_two) == total + context_window._IMAGE_TOKEN_ESTIMATE


def test_tools_definition_counted_with_extra_constants():
    """tools 定义计入（对标 LiteLLM _count_extra）：+序列化本体 +9 +3；有 system 消息再 -4。"""
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]
    serialized = context_window.count_tokens(json.dumps(tools, ensure_ascii=False))

    base = count_tokens_messages([{"role": "user", "content": "hi"}])
    assert count_tokens_messages([{"role": "user", "content": "hi"}], tools=tools) \
        == base + serialized + 9 + 3  # 无 system 消息

    base_sys = count_tokens_messages(
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}])
    assert count_tokens_messages(
        [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}],
        tools=tools) == base_sys + serialized + 9 + 3 - 4  # 有 system -4


def test_legacy_messages_tokens_plus_overhead_equivalence():
    """收编锚点：_messages_tokens 与 token_estimate 均委托 count_tokens_messages，三函数完全等价。

    Step1 时旧口径 = 新口径 - 3×条数（补结构开销而非换算法）；Step2 收编后为恒等。
    """
    msgs = [
        {"role": "system", "content": "你是物理辅导老师。"},
        {"role": "user", "content": "什么是牛顿第二定律？"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search", "arguments": "{\"q\": \"F=ma\"}"}}]},
        {"role": "tool", "content": "牛顿第二运动定律：F = ma ……", "tool_call_id": "c1"},
    ]
    from core.agentic.context_budget import _messages_tokens
    from core.agentic.context_policy import token_estimate
    assert count_tokens_messages(msgs) == _messages_tokens(msgs) == token_estimate(msgs)


def test_empty_and_none_inputs():
    """空消息列表 / content=None / tools=None 边界。"""
    assert count_tokens_messages([]) == 0
    assert count_tokens_messages([{"role": "assistant", "content": None}]) == 3
    assert count_tokens_messages([{"role": "user", "content": "hi"}], tools=None) \
        == count_tokens_messages([{"role": "user", "content": "hi"}])
    assert count_tokens_messages([{"role": "user", "content": "hi"}], tools=[]) \
        == count_tokens_messages([{"role": "user", "content": "hi"}])
