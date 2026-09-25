"""沙箱 service 模式客户端测试：httpx uds 调用映射 / 降级语义 / 判分反馈文案。

Mock 边界：transport 注入 httpx.MockTransport（替换的是进程外 HTTP 依赖）；
run_tests/grade_answer 本体真跑（service 分派经 settings.sandbox.mode 注入）。
攻击面验证（网络隔离/密钥不可达）在 test_sandbox_service_attacks.py（需 docker）。
"""
from __future__ import annotations

import httpx
import pytest


def _enable_service_mode(monkeypatch):
    """把 settings 切到 service 模式（socket/token 为测试值）。"""
    from pydantic import SecretStr
    from settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s.sandbox, "mode", "service")
    monkeypatch.setattr(s.sandbox, "socket", "/run/sandbox/sandbox.sock")
    monkeypatch.setattr(s.sandbox, "token", SecretStr("t0ken"))


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ── run_code：响应映射 ──────────────────────────────────────────────────────


def test_service_ok_maps_to_true(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    transport = _mock_transport(
        lambda req: httpx.Response(200, json={
            "ok": True, "exit_code": 0, "stdout": "hi\n", "stderr": "",
            "timed_out": False, "detail": "",
        })
    )
    assert sb._run_code_service("print(1)", 5, transport=transport) == (True, "")


def test_service_timeout_maps_to_detail(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    transport = _mock_transport(
        lambda req: httpx.Response(200, json={
            "ok": False, "exit_code": None, "stdout": "", "stderr": "",
            "timed_out": True, "detail": "timeout>5s",
        })
    )
    assert sb._run_code_service("while True: pass", 5, transport=transport) == (
        False, "timeout>5s",
    )


def test_service_nonzero_exit_carries_stderr(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    transport = _mock_transport(
        lambda req: httpx.Response(200, json={
            "ok": False, "exit_code": 3, "stdout": "", "stderr": "ZeroDivisionError",
            "timed_out": False, "detail": "",
        })
    )
    ok, detail = sb._run_code_service("1/0", 5, transport=transport)
    assert ok is False and detail.startswith("exit=3:") and "ZeroDivisionError" in detail


def test_service_5xx_raises_unavailable(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    transport = _mock_transport(lambda req: httpx.Response(500, text="boom"))
    with pytest.raises(sb.SandboxUnavailableError):
        sb._run_code_service("print(1)", 5, transport=transport)


def test_service_connect_error_raises_unavailable(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)

    def handler(req):
        raise httpx.ConnectError("no route", request=req)

    with pytest.raises(sb.SandboxUnavailableError):
        sb._run_code_service("print(1)", 5, transport=_mock_transport(handler))


def test_service_403_rejected_token_raises(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    transport = _mock_transport(lambda req: httpx.Response(403, text="forbidden"))
    with pytest.raises(sb.SandboxUnavailableError, match="token"):
        sb._run_code_service("print(1)", 5, transport=transport)


def test_service_sends_token_header_and_payload(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["token"] = req.headers.get("X-Sandbox-Token")
        captured["json"] = __import__("json").loads(req.content)
        return httpx.Response(200, json={
            "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
            "timed_out": False, "detail": "",
        })

    sb._run_code_service("print(1)", 7, transport=_mock_transport(handler))
    assert captured["token"] == "t0ken"
    assert captured["json"] == {"code": "print(1)", "timeout_s": 7}


def test_service_oversize_code_rejected_before_transport(monkeypatch):
    """超过服务端 64KB 上限的代码在客户端直接拒，不打服务。"""
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)

    def handler(req):  # 若被调用说明没做前置校验
        raise AssertionError("transport should not be hit for oversize code")

    ok, detail = sb._run_code_service("x = 1\n" + "#" * (70 * 1024), 5,
                                      transport=_mock_transport(handler))
    assert ok is False and "too long" in detail


def test_run_code_dispatches_by_mode(monkeypatch):
    """mode=local 走本地子进程（默认路径不受 service 改造影响）。"""
    from settings import get_settings

    s = get_settings()
    monkeypatch.setattr(s.sandbox, "mode", "local")
    from core.question import sandbox as sb

    assert sb.run_code("print(6*7)") == (True, "")


# ── run_tests：降级语义（沙箱不可用 → skipped + degraded，不硬失败） ────────


def _ok_resp() -> httpx.Response:
    return httpx.Response(200, json={
        "ok": True, "exit_code": 0, "stdout": "", "stderr": "",
        "timed_out": False, "detail": "",
    })


def test_run_tests_degrades_all_when_service_down(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)

    def handler(req):
        raise httpx.ConnectError("down", request=req)

    monkeypatch.setattr(
        sb, "_service_transport_factory", lambda: _mock_transport(handler)
    )
    r = sb.run_tests("x=1", ["assert x==1", "assert x==2"])
    assert r.degraded is True
    assert r.total == 0 and r.passed == [] and r.failed == []
    assert sorted(r.skipped) == [0, 1]


def test_run_tests_degrades_midway_keeps_earlier_results(monkeypatch):
    from core.question import sandbox as sb

    _enable_service_mode(monkeypatch)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise httpx.ConnectError("down mid-way", request=req)
        return _ok_resp()

    monkeypatch.setattr(
        sb, "_service_transport_factory", lambda: _mock_transport(handler)
    )
    r = sb.run_tests("x=1", ["assert x==1", "assert x==2", "assert x==3"])
    assert r.degraded is True
    assert r.passed == [0]                      # 首条已执行且通过
    assert r.skipped == [1, 2] and r.total == 1  # 剩余全部跳过，不计分母


# ── grading：降级反馈文案 ──────────────────────────────────────────────────


async def test_grade_coding_degraded_feedback(monkeypatch):
    from core.question.grading import grade_answer
    from core.question.sandbox import TestRunResult

    monkeypatch.setattr(
        "core.question.grading.run_tests",
        lambda *a, **k: TestRunResult(total=0, degraded=True),
    )
    r = await grade_answer({
        "question_id": "q1", "question_type": "coding",
        "question": "q", "correct_answer": "a",
        "user_answer": "print(1)", "test_cases": ["assert True"],
    })
    assert r.grade == "ungraded"
    assert "沙箱" in r.feedback and "不可用" in r.feedback


async def test_grade_coding_budget_exhausted_feedback_distinguishes(monkeypatch):
    """预算耗尽（非降级）保持原文案，两个口径不串。"""
    from core.question.grading import grade_answer
    from core.question.sandbox import TestRunResult

    monkeypatch.setattr(
        "core.question.grading.run_tests",
        lambda *a, **k: TestRunResult(total=0),  # degraded=False
    )
    r = await grade_answer({
        "question_id": "q1", "question_type": "coding",
        "question": "q", "correct_answer": "a",
        "user_answer": "print(1)", "test_cases": ["assert True"],
    })
    assert r.grade == "ungraded"
    assert "预算" in r.feedback and "沙箱不可用" not in r.feedback
