"""沙箱容器攻击面自检（由 test_sandbox_service_attacks.py 经 docker compose exec 从 stdin 注入执行）。

容器内无 pytest，本脚本用纯 stdlib（http.client over AF_UNIX）直连本容器 uds，
逐项验证隔离性质并打印 PASS/FAIL；任何 FAIL → 非零退出。token 从容器自身 env 读
（宿主无须知道）。新增攻击面时在此加 case，pytest 侧不用改。
"""
from __future__ import annotations

import http.client
import json
import os
import socket
import sys
import threading
import time

SOCKET = os.environ.get("SANDBOX_SOCKET", "/run/sandbox/sandbox.sock")
TOKEN = os.environ.get("SANDBOX_TOKEN", "")

# LC_CTYPE 非 leak：子进程以 C locale 启动时 CPython PEP 538 locale 强制会自我
# 注入 LC_CTYPE=C.UTF-8（setlocale + 更新自身 environ），除此之外只允许 3 个
# 线程数变量——集合差集非空即泄密。
_ALLOWED_ENV = {
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "LC_CTYPE",
}
_OUTPUT_MAX = 64 * 1024  # 与服务端 OUTPUT_MAX_BYTES 同值
_MEM_BOMB_BYTES = 512 * 1024 * 1024  # 超过服务端 256MB RLIMIT_AS


class _UnixHTTP(http.client.HTTPConnection):
    """http.client over Unix domain socket（容器内无 httpx，stdlib 直连）。"""

    def __init__(self, timeout: float = 60):
        super().__init__("localhost", timeout=timeout)
        self._unix_path = SOCKET

    def connect(self):
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(self.timeout)
        s.connect(self._unix_path)
        self.sock = s


def _post_raw(payload: dict, token: str) -> int:
    """绕过 pydantic 校验语义的裸 POST，返回 HTTP 状态码。"""
    conn = _UnixHTTP()
    try:
        conn.request(
            "POST", "/execute",
            body=json.dumps(payload),
            headers={"Content-Type": "application/json", "X-Sandbox-Token": token},
        )
        return conn.getresponse().status
    finally:
        conn.close()


def execute(code: str, timeout_s: int = 5, token: str | None = None) -> tuple[int, dict]:
    """返回 (http_status, body_dict)；连接层异常直接 raise（判 FAIL）。"""
    conn = _UnixHTTP(timeout=timeout_s + 30)
    try:
        conn.request(
            "POST", "/execute",
            body=json.dumps({"code": code, "timeout_s": timeout_s}),
            headers={
                "Content-Type": "application/json",
                "X-Sandbox-Token": TOKEN if token is None else token,
            },
        )
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, json.loads(raw) if raw else {}
    finally:
        conn.close()


_CASES: list[tuple[str, callable]] = []


def case(fn):
    _CASES.append(fn)
    return fn


# ── 密钥不可达 ──────────────────────────────────────────────────────────────


@case
def env_has_no_credentials():
    status, body = execute("import os, json; print(json.dumps(sorted(os.environ)))")
    assert status == 200, body
    env = set(json.loads(body["stdout"].strip()))
    leaked = env - _ALLOWED_ENV
    assert not leaked, f"子进程 env 泄漏: {leaked}"


@case
def no_env_file_or_secrets_in_fs():
    status, body = execute(
        "import os\n"
        "print(os.path.exists('/app/.env'), os.path.exists('.env'), "
        "os.path.exists('/run/secrets'))"
    )
    assert status == 200, body
    assert body["stdout"].strip() == "False False False", body["stdout"]
    assert "sk-" not in body["stdout"] and body["exit_code"] == 0


@case
def runs_as_non_root():
    status, body = execute("import os; print(os.getuid())")
    assert status == 200 and body["exit_code"] == 0, body
    assert int(body["stdout"].strip()) != 0, "沙箱以 root 运行（RLIMIT_NPROC 失效）"


# ── 无网络 ──────────────────────────────────────────────────────────────────


@case
def no_outbound_tcp_or_dns():
    code = (
        "import socket\n"
        "tcp = dns = 'ok'\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=3).close()\n"
        "except OSError:\n"
        "    tcp = 'fail'\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 80)\n"
        "except OSError:\n"
        "    dns = 'fail'\n"
        "print(tcp, dns)\n"
    )
    status, body = execute(code, timeout_s=12)
    if body.get("timed_out"):
        return  # DNS 解析在无网栈里可能一直等到超时——同样证明无出网
    assert status == 200 and body["exit_code"] == 0, body
    assert body["stdout"].strip() == "fail fail", (
        f"外连/解析未被拒绝: {body['stdout']!r}"
    )


# ── 资源上限 ────────────────────────────────────────────────────────────────


@case
def fork_bomb_capped_and_service_survives():
    code = "import os\nwhile True:\n    try:\n        os.fork()\n    except OSError:\n        pass"
    t0 = time.monotonic()
    status, body = execute(code, timeout_s=3)
    wall = time.monotonic() - t0
    assert wall < 20, f"fork 炸弹拖垮执行: {wall:.1f}s"
    assert body.get("timed_out") or (body.get("exit_code") or 0) != 0, body
    # 兽群成员在 RLIMIT_CPU=10s 内被内核收割；期间同 uid 配额可能短暂占满 → 有界重试
    for _ in range(10):
        time.sleep(2)
        s2, b2 = execute("print(1)")
        if s2 == 200 and b2.get("ok"):
            return
        assert time.monotonic() - t0 < 30, f"fork 炸弹后服务未恢复: {s2} {b2}"
    raise AssertionError("fork 炸弹后服务在 20s 内未恢复可用")


@case
def memory_bomb_killed_by_rlimit():
    status, body = execute(f"bytearray({_MEM_BOMB_BYTES})")
    assert status == 200, body
    assert not body.get("ok") and not body.get("timed_out"), body
    assert "MemoryError" in body.get("stderr", ""), body


@case
def cpu_spin_hits_wall_timeout():
    t0 = time.monotonic()
    status, body = execute("while True: pass", timeout_s=3)
    wall = time.monotonic() - t0
    assert status == 200 and body.get("timed_out"), body
    assert wall < 10, f"死循环未被墙钟超时及时收割: {wall:.1f}s"


@case
def stdout_truncated():
    status, body = execute("print('A' * (200 * 1024))")
    assert status == 200 and body["exit_code"] == 0, body
    assert len(body["stdout"]) == _OUTPUT_MAX, len(body["stdout"])


# ── 认证与限流 ──────────────────────────────────────────────────────────────


@case
def token_required():
    s1, _ = execute("print(1)", token="")
    s2, _ = execute("print(1)", token="wrong-token")
    assert s1 == 403 and s2 == 403, (s1, s2)


@case
def invalid_input_rejected():
    # timeout_s 超上界/缺 code → 422（pydantic 校验）；code 超 64KB → 400（显式拒）
    assert _post_raw({"code": "print(1)", "timeout_s": 999}, TOKEN) == 422
    assert _post_raw({"timeout_s": 5}, TOKEN) == 422
    assert _post_raw({"code": "x = 1  #" + "A" * (70 * 1024), "timeout_s": 5}, TOKEN) == 400


@case
def concurrent_requests_queue_not_reject():
    results: list[tuple[int, dict]] = [None] * 5  # type: ignore[list-item]

    def worker(i):
        try:
            results[i] = execute("import time; time.sleep(0.5); print(1)")
        except Exception as exc:  # noqa: BLE001 —— 记录后由断言统一判
            results[i] = (0, {"detail": str(exc)})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    assert all(s == 200 and b.get("ok") for s, b in results), results
    assert wall < 8, f"5 并发未排队消化: {wall:.1f}s"


def main() -> int:
    if not TOKEN:
        print("FAIL 容器未配置 SANDBOX_TOKEN（服务应 fail-closed，无法自检）")
        return 1
    failures = 0
    for fn in _CASES:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 —— 自检脚本须吞异常汇总退出码
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"{'ALL PASS' if not failures else f'{failures} FAILED'} ({len(_CASES)} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
