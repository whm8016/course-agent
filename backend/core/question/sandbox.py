"""coding 题最小子进程执行器（尽力而为沙箱）。

⚠️ 风险披露（不是安全边界）：本模块只做「独立临时目录 + 超时 + `-I` 隔离模式
子进程」的尽力而为校验，防的是 LLM 生成的代码片段本身跑不过（语法错/死循环），
**不是**多租户级容器隔离（gVisor/Firecracker/namespace 级）。当前仓库无相应
基建，本模块不冒充安全边界——被校验的代码来自本仓库 LLM 生成的题目参考答案，
不是任意用户代码执行服务。若未来开放用户自带代码执行，必须换容器级隔离，
不得继续复用本文件。

用法（quiz pipeline critic 轮）：question_type=coding 且 correct_answer/explanation
含围栏代码块时逐块执行，跑不过（非零退出/超时）计入 critic issues 触发重试。
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 5

# ```python / ```py / 裸 ``` 围栏块，按出现顺序
_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[str]:
    """抽 Markdown 围栏代码块（```python / ```py / 裸 ```），按出现顺序去空块。"""
    return [
        m.group(1).strip()
        for m in _CODE_BLOCK.finditer(text or "")
        if m.group(1).strip()
    ]


def run_code(code: str, timeout: int = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """在独立临时目录跑一段 python 代码，返回 (ok, detail)。

    ok=False 的情形：编译/运行期异常（非零退出）或超时。ok=True 仅表示退出码 0
    ——不保证语义正确（那是 critic LLM 的职责）。失败 detail 截 500 字进 issues。
    `-I`（isolated mode）：忽略 PYTHONPATH / 用户 site-packages / 环境变量注入，
    尽力缩小子进程与宿主的环境耦合。
    """
    with tempfile.TemporaryDirectory(prefix="quiz_sandbox_") as td:
        path = Path(td) / "snippet.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout>{timeout}s"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, f"exit={proc.returncode}: {err[:500]}"
    return True, ""


__all__ = ["extract_code_blocks", "run_code", "DEFAULT_TIMEOUT_S"]
