"""coding 题执行器：local 子进程（尽力而为，dev/CI）或 service 沙箱容器（上线口径）。

两种模式（settings.sandbox.mode）：
- **local**（默认）：本进程子进程——独立临时目录 + 超时 + `-I` 隔离模式 + 最小
  环境变量 + rlimit 资源上限（POSIX）。⚠️ 不是安全边界：无网络隔离；RLIMIT_NPROC
  按 UID 计，容器以 root 跑时对 root 不生效；Windows 开发机无 resource 模块、
  preamble 不注入（仅超时+最小 env）。仅供开发与 CI（settings `_check_prod`
  在 production 拒绝 local 模式）。
- **service**：经 Unix domain socket 调独立沙箱容器（sandbox/，compose 服务）：
  `network_mode: none`（无出网、无内网可达）+ 非 root + 零凭据零业务卷 + 容器级
  mem/cpus/pids 限额。学生代码攻击面从「backend 容器 root + .env + 出网」坍缩为
  「一个什么都没有的空容器」。服务内部沿用同一套子进程纪律（超时+`-I`+最小 env+
  rlimit preamble），执行数据流对齐 Piston api/src/job.js 的 prime→execute→cleanup。
  不可用时抛 SandboxUnavailableError → 判分降级 ungraded，**绝不回退本地执行**
  （fail 降可用性，不降安全性）。攻击面回归见 tests/test_sandbox_service_attacks.py。

用法：①quiz pipeline critic 轮：question_type=coding 时执行 correct_answer 里的
围栏代码块（correct_answer 才是"要能跑的参考答案"；explanation 里的代码常是
故意的错误示范，跑了必误杀，不吃），跑不过（非零退出/超时）计入 critic issues
触发重试。②判分期：`run_tests` 把「starter_code + 学生代码 + 单条 assert」逐条
独立子进程执行（human-eval check_correctness 同构），逐条=部分正确可计数。
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from settings import get_settings

logger = logging.getLogger(__name__)

# service 模式：沙箱服务 /execute 的 code 上限（与服务端常量同值）；超限客户端
# 直接拒（不打服务）——判分口径与「代码超长」本地编译失败一致，属无效输入。
_SERVICE_CODE_MAX_BYTES = 64 * 1024


class SandboxUnavailableError(RuntimeError):
    """沙箱服务不可用（连接失败/超时/5xx/token 拒绝）。

    判分侧捕获后降级 ungraded（TestRunResult.degraded），critic 侧落入既有
    except Exception → issue 重试口径。绝不回退本地子进程执行。
    """

DEFAULT_TIMEOUT_S = 5

# ```python / ```py / ```python3 / ```Python 围栏块（围栏头大小写不敏感），按出现顺序
_CODE_BLOCK = re.compile(
    r"```(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)

# ---- A1 资源上限常量（human-eval reliability_guard 的子进程版）----------------
# 课程 coding 题预期纯 stdlib 函数式；256MB 对重库（numpy 全家）可能误杀——上限
# 做成常量可调，误杀时进 issue 反馈而非静默。
_RLIMIT_AS_MB = 256
_RLIMIT_CPU_S = 10
# 按 UID 计的进程/线程数上限：取 32 给同 uid 的 worker 自身留头寸（gunicorn -w4
# + arq ≈ 6-10 任务），太低会误伤 worker 自己的 fork/thread。
_RLIMIT_NPROC = 32

# 用例防御钳制：数量/单条长度/总执行预算（防 LLM 生成失控的 test_cases 拖垮判分）
_MAX_TEST_CASES = 10
_MAX_TEST_CASE_LEN = 500
_TESTS_TOTAL_BUDGET_S = 40.0

_RLIMIT_PREAMBLE = (
    "import resource\n"
    f"resource.setrlimit(resource.RLIMIT_AS, "
    f"({_RLIMIT_AS_MB} * 1024 * 1024, {_RLIMIT_AS_MB} * 1024 * 1024))\n"
    f"resource.setrlimit(resource.RLIMIT_CPU, ({_RLIMIT_CPU_S}, {_RLIMIT_CPU_S}))\n"
    f"resource.setrlimit(resource.RLIMIT_NPROC, ({_RLIMIT_NPROC}, {_RLIMIT_NPROC}))\n"
)

# 并发闸：同时最多 4 个沙箱子进程（100 人同时交卷时排队消化，峰值 ~4×60MB 瞬时）。
# threading 而非 asyncio：run_code 走 asyncio.to_thread 调用、天然多线程，闸必须
# 挡住所有线程（critic 路径 + 判分路径 + 未来调用方）。
_EXEC_GATE = threading.Semaphore(4)


def _inject_preamble(code: str) -> str:
    """POSIX 下把 setrlimit preamble 拼在代码头部（soft=hard，子代码改不高 hard）。

    Windows 开发机无 resource 模块，原样返回（无内存/CPU 上限，靠超时兜底）。
    不用 preexec_fn：多线程进程里 fork+exec 间跑 Python 回调有死锁风险。
    """
    if sys.platform == "win32":
        return code
    return _RLIMIT_PREAMBLE + code


@dataclass
class TestRunResult:
    """run_tests 结果：passed/failed 带原始用例下标（部分正确计数用）。

    skipped：总预算耗尽未执行的用例——不计 total/ratio（基础设施限速不是学生
    能力问题，复审 P2-2）。
    """

    __test__ = False  # 防 pytest 把 Test 开头的 dataclass 误当测试类收集

    passed: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    total: int = 0
    degraded: bool = False  # 沙箱服务不可用导致用例未执行（区别于预算耗尽的 skipped）

    @property
    def pass_ratio(self) -> float:
        return len(self.passed) / self.total if self.total else 0.0


def extract_code_blocks(text: str) -> list[str]:
    """抽 Markdown 围栏代码块（python 系语言标注或裸 ```），按出现顺序去空块。"""
    return [
        m.group(1).strip()
        for m in _CODE_BLOCK.finditer(text or "")
        if m.group(1).strip()
    ]


def _sandbox_env() -> dict[str, str]:
    """最小子进程环境：剥掉 DB URL / API key 等全部父进程凭据（防生成代码读走外传）。

    Windows 下 SYSTEMROOT 必须保留（缺了基础 import 链都起不来）；POSIX 给空环境
    即可。注意 `-I` 只忽略 PYTHON* 变量，不剥其余 env——这里显式整env 覆盖。
    线程数三家封 1：human-eval reliability_guard 同款，防 BLAS/OpenMP 线程池
    在 CPU 限制外并行烧核。
    """
    base = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if sys.platform == "win32":
        base["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return base


def sanitize_test_cases(test_cases: list[str]) -> list[tuple[int, str]]:
    """清洗 assert 用例：strip 去空、丢超长（>500 字）、丢非 assert 前缀、截断到 10 条。

    返回 (原始下标, 用例文本)——下标保留原位，判分反馈能对回题目里的第几条。
    丢弃而非报错：test_cases 是 LLM 生成的，坏条目按无效输入处理，不算学生失败。
    """
    kept: list[tuple[int, str]] = []
    for i, raw in enumerate(test_cases or []):
        t = str(raw).strip()
        if not t or not t.startswith("assert") or len(t) > _MAX_TEST_CASE_LEN:
            continue
        kept.append((i, t))
        if len(kept) >= _MAX_TEST_CASES:
            break
    return kept


def run_tests(
    code: str,
    test_cases: list[str],
    *,
    timeout_per_test: int = DEFAULT_TIMEOUT_S,
    starter_code: str = "",
) -> TestRunResult:
    """「starter_code + 学生代码 + 单条 assert」逐条独立子进程执行判分。

    human-eval check_correctness 同构：每条用例独立跑一遍完整代码（exit 0=过），
    逐条而非合并——一条挂不遮蔽其余，部分正确可计数（ratio → good/fair/poor）。
    总预算 40s：超预算的剩余用例进 skipped（不计分母），防失控用例串拖垮判分
    端点的 60s 超时。
    """
    cases = sanitize_test_cases(test_cases)
    full_code = (starter_code + "\n" if starter_code.strip() else "") + (code or "")
    result = TestRunResult(total=len(cases))
    deadline = time.monotonic() + _TESTS_TOTAL_BUDGET_S
    for pos, (idx, tc) in enumerate(cases):
        if time.monotonic() > deadline:
            # 预算耗尽=基础设施限速，不计入 failed/total（不拉低学生 ratio）
            result.skipped.append(idx)
            result.total -= 1
            continue
        try:
            ok, detail = run_code(full_code + "\n\n" + tc, timeout=timeout_per_test)
        except SandboxUnavailableError as exc:
            # 沙箱服务不可用：剩余用例（含当前）全部跳过、不计分母，degraded 供
            # 判分反馈区分「沙箱暂不可用」vs「预算耗尽」。不重试、不回退本地执行。
            logger.warning("沙箱服务不可用，剩余用例跳过：%s", exc, exc_info=True)
            result.degraded = True
            for rem_idx, _ in cases[pos:]:
                result.skipped.append(rem_idx)
                result.total -= 1
            break
        if ok:
            result.passed.append(idx)
        else:
            result.failed.append((idx, detail))
    return result


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """跑一段 python 代码，返回 (ok, detail)。按 settings.sandbox.mode 分派执行域。

    ok=False 的情形：编译/运行期异常（非零退出）或超时。ok=True 仅表示退出码 0
    ——不保证语义正确（那是 critic LLM 的职责）。失败 detail 截 500 字进 issues。
    service 模式不可用抛 SandboxUnavailableError（见类 docstring）。
    """
    # 闸在分派层统一包两模式（服务端另有自己的并发闸）：backend 侧同时最多 4 个
    # 沙箱执行在途，判分 to_thread 与 critic 并发调用都被挡住
    with _EXEC_GATE:
        if get_settings().sandbox.mode == "service":
            return _run_code_service(code, timeout)
        return _run_code_local(code, timeout)


_service_client_cache: httpx.Client | None = None


def _service_client_factory() -> httpx.Client:
    """service 模式共享 Client（懒加载单例）。独立工厂便于测试注入 MockTransport。

    不每次调用新建：httpx 0.28 的 HTTPTransport 构造无条件建 SSL 上下文（实测
    ~10ms），而 uds 上 http:// 根本不用 TLS——判分每题最多 10 次调用、critic 轮
    再加围栏块执行，每次新建是纯浪费。keepalive 关闭：每请求新建 uds 连接
    （~0.1ms 量级），沙箱容器重启后池内不留死连接。
    """
    global _service_client_cache
    if _service_client_cache is None:
        _service_client_cache = httpx.Client(
            transport=httpx.HTTPTransport(
                uds=get_settings().sandbox.socket,
                retries=0,
                limits=httpx.Limits(max_keepalive_connections=0),
            ),
        )
    return _service_client_cache


def _run_code_service(code: str, timeout: int) -> tuple[bool, str]:
    """service 模式：经 uds 调沙箱容器 /execute，响应映射回 (ok, detail)。

    词表与本地路径同口径（timed_out → timeout>Ns；非零退出 → exit=N: stderr）。
    rlimit preamble 由服务端注入（本端发原始代码）。URL host 无意义（uds transport
    只看路径），用 sandbox.local 占位。共享 Client 不在此关闭（进程级单例）。
    """
    if len(code.encode("utf-8", errors="ignore")) > _SERVICE_CODE_MAX_BYTES:
        return False, "code too long for sandbox service"
    cfg = get_settings().sandbox
    try:
        resp = _service_client_factory().post(
            "http://sandbox.local/execute",
            json={"code": code, "timeout_s": int(timeout)},
            headers={"X-Sandbox-Token": cfg.token.get_secret_value()},
            timeout=httpx.Timeout(
                connect=cfg.connect_timeout_s,
                read=timeout + cfg.margin_s,
                write=10.0,
                pool=cfg.connect_timeout_s,
            ),
        )
    except httpx.HTTPError as exc:
        raise SandboxUnavailableError(f"sandbox service unreachable: {exc}") from exc
    if resp.status_code == 403:
        raise SandboxUnavailableError("sandbox service rejected token")
    if resp.status_code != 200:
        raise SandboxUnavailableError(
            f"sandbox service error HTTP {resp.status_code}"
        )
    data = resp.json()
    if data.get("timed_out"):
        return False, f"timeout>{timeout}s"
    exit_code = data.get("exit_code")
    if exit_code != 0:
        err = (data.get("stderr") or data.get("stdout") or "").strip()
        return False, f"exit={exit_code}: {err[:500]}"
    return True, ""


def _run_code_local(code: str, timeout: int) -> tuple[bool, str]:
    """local 模式：在独立临时目录起子进程跑一段 python 代码，返回 (ok, detail)。

    加固口径：`-I`（忽略 PYTHONPATH / 用户 site / PYTHON* env）+ `env=_sandbox_env()`
    （剥凭据 + 线程数封 1）+ `stdin=DEVNULL`（input() 挂不住）+ `-X utf8` + utf-8
    解码（防子进程输出按 locale 编码炸 UnicodeDecodeError——异常属调用方 issue
    而非崩溃）+ POSIX 下 rlimit preamble（内存/CPU/进程数 hard 上限）。并发闸在
    run_code 分派层（两模式同一处）。仅 dev/CI（见模块 docstring 的模式说明）。
    """
    with tempfile.TemporaryDirectory(prefix="quiz_sandbox_") as td:
        path = Path(td) / "snippet.py"
        path.write_text(_inject_preamble(code), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(path)],
                capture_output=True,
                timeout=timeout,
                cwd=td,
                env=_sandbox_env(),
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout>{timeout}s"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"exit={proc.returncode}: {err[:500]}"
    return True, ""


__all__ = [
    "TestRunResult",
    "DEFAULT_TIMEOUT_S",
    "SandboxUnavailableError",
    "extract_code_blocks",
    "run_code",
    "run_tests",
    "sanitize_test_cases",
]
