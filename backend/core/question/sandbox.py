"""coding 题最小子进程执行器（尽力而为沙箱）。

⚠️ 风险披露（不是安全边界）：本模块做「独立临时目录 + 超时 + `-I` 隔离模式
子进程 + 最小环境变量 + rlimit 资源上限（POSIX）」的尽力而为防护。自判分期起
执行对象包含**不可信学生代码**（信任域低于 critic 轮的 LLM 生成代码），防线为：
①RLIMIT_AS/CPU/NPROC 以 soft=hard 写死注入 preamble——子代码无法调高 hard 上限
（调高需特权），爆内存/死循环/fork 炸弹在内核层被杀；②线程数环境变量封 1（防
BLAS/OpenMP 线程池绕过 CPU 限制）；③`_EXEC_GATE` 并发闸防瞬时资源堆积。
已知剩余风险（无相应基建，显式接受）：①无网络隔离（socket 可外传，课程场景
接受）；②RLIMIT_NPROC 按 UID 计——容器以 root 跑时对 root 不生效，仅防非特权
uid 下的 fork 炸弹（运维建议 worker 用独立低权用户）；③Windows 开发机无
resource 模块、preamble 不注入（无内存/CPU 上限，仅超时+最小 env）。
🚫 红线：开放公网高利害考试（成绩直接挂钩）前必须换容器级隔离
（gVisor/Firecracker/独立容器+netns），不得继续复用本文件。

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

logger = logging.getLogger(__name__)

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
    for idx, tc in cases:
        if time.monotonic() > deadline:
            # 预算耗尽=基础设施限速，不计入 failed/total（不拉低学生 ratio）
            result.skipped.append(idx)
            result.total -= 1
            continue
        ok, detail = run_code(full_code + "\n\n" + tc, timeout=timeout_per_test)
        if ok:
            result.passed.append(idx)
        else:
            result.failed.append((idx, detail))
    return result


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """在独立临时目录跑一段 python 代码，返回 (ok, detail)。

    ok=False 的情形：编译/运行期异常（非零退出）或超时。ok=True 仅表示退出码 0
    ——不保证语义正确（那是 critic LLM 的职责）。失败 detail 截 500 字进 issues。
    加固口径：`-I`（忽略 PYTHONPATH / 用户 site / PYTHON* env）+ `env=_sandbox_env()`
    （剥凭据 + 线程数封 1）+ `stdin=DEVNULL`（input() 挂不住）+ `-X utf8` + utf-8
    解码（防子进程输出按 locale 编码炸 UnicodeDecodeError——异常属调用方 issue
    而非崩溃）+ POSIX 下 rlimit preamble（内存/CPU/进程数 hard 上限）+ `_EXEC_GATE`
    并发闸（同时最多 4 个子进程）。
    """
    with tempfile.TemporaryDirectory(prefix="quiz_sandbox_") as td:
        path = Path(td) / "snippet.py"
        path.write_text(_inject_preamble(code), encoding="utf-8")
        with _EXEC_GATE:
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
    "extract_code_blocks",
    "run_code",
    "run_tests",
    "sanitize_test_cases",
]
