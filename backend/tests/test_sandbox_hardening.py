"""沙箱加固（A1）测试：rlimit preamble + 并发闸 + run_tests 逐条判分。

覆盖（plan §A1）：
  1. _inject_preamble：win32 原样返回 / posix 注入 setrlimit（纯函数，跨平台可测）
  2. _sandbox_env：线程数环境变量三家（OMP/OPENBLAS/MKL）都置 1（防 BLAS 绕过 CPU 限制）
  3. run_code 正常路径不被 preamble 破坏（有 preamble 时 print/assert 照常跑）
  4. 内存上限（Linux only）：bytearray(10**10) 非零退出而不是拖垮 worker
  5. _EXEC_GATE 并发闸：4 槽占满后第 5 个执行等待，释放后放行
  6. run_tests：全过 / 部分过（2/3，逐条独立执行不互相遮蔽）/ 用例自身语法错只记该条失败
  7. sanitize：清洗超长（>500字）/非 assert 前缀/空用例 + 截断到 10 条，保留原始下标
  8. starter_code 拼接在学生代码之前
"""
from __future__ import annotations

import sys
import threading
import time

import pytest

from core.question import sandbox
from core.question.sandbox import (
    TestRunResult,
    _EXEC_GATE,
    _inject_preamble,
    _sandbox_env,
    run_code,
    run_tests,
    sanitize_test_cases,
)


# ── 1. preamble 注入（纯函数） ────────────────────────────────────────────────


def test_inject_preamble_platform_branch():
    code = "print(1)"
    injected = _inject_preamble(code)
    if sys.platform == "win32":
        assert injected == code  # Windows 无 resource 模块，不注入
    else:
        assert "resource.setrlimit" in injected
        assert injected.endswith(code)  # 学生代码在 preamble 之后，原样保留
        # soft=hard 写死（hard 上限后子代码无法调高，调高需特权）
        for lim in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NPROC"):
            assert lim in injected


# ── 2. 环境变量：线程数封 1 ──────────────────────────────────────────────────


def test_sandbox_env_pins_thread_counts():
    env = _sandbox_env()
    assert env.get("OMP_NUM_THREADS") == "1"
    assert env.get("OPENBLAS_NUM_THREADS") == "1"
    assert env.get("MKL_NUM_THREADS") == "1"


# ── 3. preamble 不破坏正常执行 ───────────────────────────────────────────────


def test_run_code_with_preamble_still_runs_normal_code():
    ok, detail = run_code("x = 1 + 1\nassert x == 2")
    assert ok is True, detail


# ── 4. 内存上限（Linux） ─────────────────────────────────────────────────────


@pytest.mark.skipif(sys.platform == "win32", reason="rlimit 仅 POSIX 生效")
def test_memory_limit_kills_oversized_allocation():
    ok, detail = run_code("bytearray(10 ** 10)")  # 10GB 分配 → MemoryError 非零退出
    assert ok is False
    assert detail  # 带 MemoryError 摘要


# ── 5. 并发闸：4 槽 ──────────────────────────────────────────────────────────


def test_exec_gate_blocks_fifth_concurrent_run():
    held = threading.Semaphore(0)
    # 占满 4 个槽（模拟 4 个并发沙箱执行）
    for _ in range(4):
        _EXEC_GATE.acquire()
    try:
        t = threading.Thread(
            target=lambda: (run_code("print('gated')"), held.release())
        )
        t.start()
        time.sleep(0.3)
        assert t.is_alive(), "闸内无空槽时执行应阻塞等待"
        assert not held.acquire(timeout=0.1), "被阻塞的执行不应已完成"
    finally:
        for _ in range(4):
            _EXEC_GATE.release()
        t.join(timeout=5)
    assert not t.is_alive()
    assert held.acquire(timeout=2), "释放空槽后被阻塞的执行应完成"


def test_exec_gate_capacity_is_four():
    # 并发上限 4（100 人同时交卷的排队闸门，plan §A1-3）
    for _ in range(4):
        assert _EXEC_GATE.acquire(blocking=False)
    assert not _EXEC_GATE.acquire(blocking=False), "第 5 个槽不应可用"
    for _ in range(4):
        _EXEC_GATE.release()


# ── 6. run_tests：逐条独立执行 ───────────────────────────────────────────────


def test_run_tests_all_pass():
    code = "def add(a, b):\n    return a + b\n"
    cases = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
    r = run_tests(code, cases)
    assert r.total == 3
    assert r.passed == [0, 1, 2]
    assert r.failed == []
    assert r.pass_ratio == 1.0


def test_run_tests_partial_credit():
    """部分正确：一条挂不遮蔽其余（逐条独立子进程）。"""
    code = "def div(a, b):\n    return a / b\n"  # 没处理除零
    cases = ["assert div(6, 3) == 2", "assert div(5, 1) == 5", "assert div(1, 0) == float('inf')"]
    r = run_tests(code, cases)
    assert r.total == 3
    assert r.passed == [0, 1]
    assert [i for i, _ in r.failed] == [2]
    assert "ZeroDivisionError" in r.failed[0][1]
    assert r.pass_ratio == pytest.approx(2 / 3)


def test_run_tests_syntax_error_case_records_single_failure():
    """用例自身语法错 → 只记该条 failed，不炸整批。"""
    code = "x = 1"
    cases = ["assert x == 1", "assert x == = 2"]
    r = run_tests(code, cases)
    assert r.total == 2
    assert r.passed == [0]
    assert len(r.failed) == 1 and r.failed[0][0] == 1


def test_run_tests_prepends_starter_code():
    """starter_code 拼在学生代码之前：签名骨架 + 学生实现 + assert 三段同进程。"""
    starter = "def add(a, b):\n"
    student = "    return a + b\n"
    r = run_tests(student, ["assert add(1, 1) == 2"], starter_code=starter)
    assert r.passed == [0] and r.failed == []


def test_run_tests_timeout_per_case():
    code = "def f(n):\n    while n:\n        pass\n"
    r = run_tests(code, ["assert f(1) is None"], timeout_per_test=1)
    assert r.passed == []
    assert "timeout" in r.failed[0][1]


# ── 7. sanitize：清洗与钳制 ──────────────────────────────────────────────────


def test_sanitize_drops_invalid_and_caps_count():
    cases = (
        ["assert a == 1", "", "   ", "print('not assert')", "x" * 501]
        + [f"assert b == {i}" for i in range(12)]  # 共 17 条 → 截断
    )
    kept = sanitize_test_cases(cases)
    idxs = [i for i, _ in kept]
    texts = [t for _, t in kept]
    assert all(t.startswith("assert") for t in texts)  # 空串/非 assert 前缀被丢
    assert all(len(t) <= 500 for t in texts)  # 超长被丢
    assert len(kept) == 10  # 钳制上限
    assert idxs == sorted(idxs)  # 保留原始下标且有序


def test_sanitize_preserves_original_indices():
    kept = sanitize_test_cases(["print('skip me')", "assert ok"])
    assert kept == [(1, "assert ok")]


# ── 模块契约 ─────────────────────────────────────────────────────────────────


def test_test_run_result_shape():
    r = TestRunResult(passed=[0], failed=[(1, "err")], total=2)
    assert r.pass_ratio == 0.5
    assert sandbox.__all__  # 导出契约存在
    for name in ("run_tests", "sanitize_test_cases", "TestRunResult"):
        assert name in sandbox.__all__
