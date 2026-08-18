"""coding 题最小子进程执行器（尽力而为沙箱）。

⚠️ 风险披露（不是安全边界）：本模块只做「独立临时目录 + 超时 + `-I` 隔离模式
子进程 + 最小环境变量」的尽力而为校验，防的是 LLM 生成的代码片段本身跑不过
（语法错/死循环），**不是**多租户级容器隔离（gVisor/Firecracker/namespace 级）。
已知剩余风险（无相应基建，显式接受）：①无内存上限——`bytearray(10**10)` 类
分配可在超时触发前 OOM worker；②超时只杀直接子进程，代码自行 spawn 的孙进程
存活；③无网络/文件系统限制。且被校验代码的生成条件含不可信 RAG 检索文本
（课程 KB 文档可注入出题上下文）——若未来开放用户自带代码执行或 KB 信任域
放宽，必须换容器级隔离，不得继续复用本文件。

用法（quiz pipeline critic 轮）：question_type=coding 时执行 correct_answer 里
的围栏代码块（correct_answer 才是"要能跑的参考答案"；explanation 里的代码常是
故意的错误示范，跑了必误杀，不吃），跑不过（非零退出/超时）计入 critic issues
触发重试。
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5

# ```python / ```py / ```python3 / ```Python 围栏块（围栏头大小写不敏感），按出现顺序
_CODE_BLOCK = re.compile(
    r"```(?:python3?|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)


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
    """
    if sys.platform == "win32":
        return {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
    return {}


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """在独立临时目录跑一段 python 代码，返回 (ok, detail)。

    ok=False 的情形：编译/运行期异常（非零退出）或超时。ok=True 仅表示退出码 0
    ——不保证语义正确（那是 critic LLM 的职责）。失败 detail 截 500 字进 issues。
    加固口径：`-I`（忽略 PYTHONPATH / 用户 site / PYTHON* env）+ `env=_sandbox_env()`
    （剥凭据）+ `stdin=DEVNULL`（input() 挂不住）+ `-X utf8` + utf-8 解码（防
    子进程输出按 locale 编码炸 UnicodeDecodeError——异常属调用方 issue 而非崩溃）。
    """
    with tempfile.TemporaryDirectory(prefix="quiz_sandbox_") as td:
        path = Path(td) / "snippet.py"
        path.write_text(code, encoding="utf-8")
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


__all__ = ["extract_code_blocks", "run_code", "DEFAULT_TIMEOUT_S"]
