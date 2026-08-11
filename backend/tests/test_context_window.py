"""context_window 四级解析 + compute_budgets 双阈值测试。

覆盖：
- resolve_effective_window / resolve_effective_window_with_source：
  显式配置 > 探测缓存 > 模型名模式 > heuristic 兜底（+告警）
- compute_budgets：减法留白公式、soft<=hard 不变式、小窗口模型 soft 被硬天花板钳住

探测缓存层用 autouse fixture 隔离到 tmp_path，绝不读真实 data/context_window_cache.json，
保证 table/heuristic/explicit 用例不受探测层干扰。
"""
from __future__ import annotations

import logging

import pytest

from core.agentic import context_window, window_probe
from core.agentic.context_window import (
    compute_budgets,
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

def test_compute_budgets_qwen_plus():
    """大窗口模型：soft=rot上限(128k)（比例线 500k > 128k 被上限钳住），hard=窗口-reserve-safety。"""
    soft, hard = compute_budgets("qwen-plus")
    # window=1_000_000, reserve=min(8192,20000)=8192, safety=4096 -> hard=987712
    assert hard == 1_000_000 - 8192 - 4096
    assert soft == 128000  # min(128000, 1_000_000*0.5=500000, 987712)
    assert soft <= hard


def test_compute_budgets_small_window_soft_bound_by_ratio():
    """小窗口模型：比例线(window*0.5) < 硬天花板 -> soft 被比例线钳住（更保守，符合 RULER 小窗口有效长度低）。"""
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
