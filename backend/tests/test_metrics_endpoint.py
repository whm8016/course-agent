"""plan 第三批-3：Prometheus 多进程 /metrics 端点回归测试。

验证自定义 /metrics 端点（替换 Instrumentator.expose）：
- 单进程（PROMETHEUS_MULTIPROC_DIR 未设）：用默认 REGISTRY，返回 200 + 有效 prometheus 格式 +
  ca_ 业务指标（与旧 Instrumentator.expose 行为等价）—— 这是核心回归护栏。
- multiprocess env 设置：端点走 MultiProcessCollector 路径不崩（空目录返回空 exposition）；
  完整多 worker 聚合交 loadtest 真机验证（测试进程指标在 import 时已建在默认 registry，env 后置
  不会切到 multiproc 写文件，故此处只验证端点代码路径不报错）。
- _mark_prometheus_process_dead 在 env 未设时是安全 no-op。
"""
from __future__ import annotations

import os

import pytest


async def test_metrics_endpoint_single_process(client):
    """单进程：/metrics 返回 200 + prometheus 格式 + ca_ 业务指标（默认 REGISTRY）。"""
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    body = r.text
    # prometheus exposition 格式
    assert "# HELP" in body or "# TYPE" in body
    # 业务指标以 ca_ 前缀（http 层由 instrumentator 埋，业务层由 metrics.py 埋）
    assert "ca_turn_duration_seconds" in body or "ca_llm_tokens_total" in body


async def test_metrics_endpoint_handles_multiprocess_env(tmp_path, monkeypatch, client):
    """PROMETHEUS_MULTIPROC_DIR 设置：/metrics 走 MultiProcessCollector 路径，不崩（200）。"""
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    r = await client.get("/metrics")
    assert r.status_code == 200  # MultiProcessCollector 读空目录不报错
    assert "text/plain" in r.headers.get("content-type", "")


def test_mark_process_dead_noop_without_env(monkeypatch):
    """env 未设时 _mark_prometheus_process_dead 是安全 no-op（不调 mark_process_dead）。"""
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    from main import _mark_prometheus_process_dead
    import prometheus_client.multiprocess as mp

    called = {"v": False}
    orig = mp.mark_process_dead

    def _spy(pid, path=None):  # noqa: ARG001
        called["v"] = True

    monkeypatch.setattr(mp, "mark_process_dead", _spy)
    try:
        _mark_prometheus_process_dead()  # env 未设 → 直接 return，不进 try
    finally:
        monkeypatch.setattr(mp, "mark_process_dead", orig)
    assert called["v"] is False


def test_lightrag_gauges_declare_livesum():
    """两个无 label 池 Gauge 声明 multiprocess_mode='livesum'（多 worker 求和不碰撞）。"""
    from core.observability.metrics import LIGHTRAG_IN_USE, LIGHTRAG_INSTANCES

    # prometheus_client Gauge 把 multiprocess_mode 存在 _multiprocess_mode
    assert LIGHTRAG_INSTANCES._multiprocess_mode == "livesum"
    assert LIGHTRAG_IN_USE._multiprocess_mode == "livesum"


@pytest.mark.parametrize("env_set", [False, True])
def test_dead_handler_runs_mark_when_env_set(monkeypatch, env_set):
    """env 设了 → _mark_prometheus_process_dead 调 mark_process_dead(os.getpid())。"""
    import prometheus_client.multiprocess as mp
    from main import _mark_prometheus_process_dead

    if env_set:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/x")
    else:
        monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)

    seen = {}
    orig = mp.mark_process_dead

    def _spy(pid, path=None):  # noqa: ARG001
        seen["pid"] = pid

    monkeypatch.setattr(mp, "mark_process_dead", _spy)
    try:
        _mark_prometheus_process_dead()
    finally:
        monkeypatch.setattr(mp, "mark_process_dead", orig)

    if env_set:
        assert seen.get("pid") == os.getpid()
    else:
        assert seen == {}
