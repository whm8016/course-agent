"""metrics.py 模块级指标幂等性回归测试。

根因：core/observability/metrics.py 在 import 时构造全局 Prometheus 指标，注册进全进程共享的
默认 REGISTRY。长跑单进程（ARQ worker / 离线评测 inspect）若二次执行模块体——部分导入被
sys.modules 驱逐后 ARQ 重试（max_tries）、第二模块身份、或 reload——第二次 register 同名指标即抛
``Duplicated timeseries``，连累索引任务全 ERROR。_metric helper 让构造幂等（同名已注册则复用）。

本测试用 importlib.reload 确定性复现"模块体二次执行"：首次 import 已注册全部指标，reload 强制
再跑一次模块体，等价于同一全局 REGISTRY 上的二次 register。修复前 reload 会抛 Duplicated，
修复后幂等复用、无异常。
"""
from __future__ import annotations

import importlib


def test_metrics_module_survives_reload(monkeypatch):
    """reload（= 模块体二次执行）不得抛 Duplicated timeseries，且符号仍可用。"""
    # 确保单进程模式：multiproc env + 缺目录会让指标构造走 MmapedDict 路径，干扰本测试判定。
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    metrics = importlib.import_module("core.observability.metrics")
    # 首次 import 已注册全部指标；reload 强制再跑一次模块体（等价二次 register 同一 REGISTRY）
    importlib.reload(metrics)

    # reload 后符号仍是有效指标对象（_metric 复用了既有 collector），observe 不抛
    metrics.observe_turn("chat", "ok", 1200)
    metrics.observe_worker_job("indexing", "ok", 5000)
    assert metrics.TURN_DURATION is not None
    assert metrics.LLM_TOKENS_TOTAL is not None


def test_metric_reuses_existing_collector(monkeypatch):
    """同名已注册时 _metric 返回的是同一个 collector 对象（复用而非新建/崩溃）。"""
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    from prometheus_client import REGISTRY

    metrics = importlib.import_module("core.observability.metrics")
    # 模拟"已注册"：REGISTRY 里已有 ca_turn_duration_seconds → 再次构造应直接返回既有对象
    existing = REGISTRY._names_to_collectors["ca_turn_duration_seconds"]
    again = metrics._metric(
        __import__("prometheus_client").Histogram,
        "ca_turn_duration_seconds",
        "dup",
        labelnames=["mode", "status"],
    )
    assert again is existing
