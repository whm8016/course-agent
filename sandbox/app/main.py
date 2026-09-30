"""编程题沙箱执行服务（sidecar 容器，仓库根 sandbox/ + docker-compose.yml）。

部署形态（隔离即架构）：``network_mode: none``（学生代码无出网、无内网可达）+
非 root（RLIMIT_NPROC 按 UID 计，root 下失效）+ 唯一挂载为 uds 共享卷 + 容器内
**零凭据零业务卷** + 容器级 mem/cpus/pids 限额。攻击面回归：
backend/tests/test_sandbox_service_attacks.py（pytest -m sandbox_service）。

执行纪律与 backend local 模式同源（core/question/sandbox.py）：超时 + `-I` +
最小 env + rlimit preamble（soft=hard）。执行顺序对齐 Piston api/src/job.js 的
prime（并发闸→临时目录→写 snippet）→ execute（spawn→墙钟超时→kill）→
cleanup（TemporaryDirectory 退出即清）；meta 等价物 = returncode/stdout/stderr。

认证：X-Sandbox-Token 与 backend settings.sandbox.token 同值（hmac 恒时比较）。
未配 SANDBOX_TOKEN 时全部 /execute 拒绝（fail-closed，防裸奔部署）。
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("sandbox")

_TOKEN = os.environ.get("SANDBOX_TOKEN", "")
_SOCKET_PATH = os.environ.get("SANDBOX_SOCKET", "/run/sandbox/sandbox.sock")
_CONCURRENCY = int(os.environ.get("SANDBOX_CONCURRENCY", "4"))

# 与 backend/core/question/sandbox.py 的 _SERVICE_CODE_MAX_BYTES 同值；两端成对改
CODE_MAX_BYTES = 64 * 1024
OUTPUT_MAX_BYTES = 64 * 1024
TIMEOUT_MAX_S = 30

# ── 子进程资源上限（与 backend local 模式同值，两端成对改）──────────────────
# NPROC 按 UID 计：本服务自身与全部执行子进程同 uid（65532），闸=32 时 fork 炸弹
# 兽群连同服务进程共享该上限；兽群成员各自在 RLIMIT_CPU=10s 内被内核收割，期间
# 新 execute 可能短暂 EAGAIN——判分侧按用例失败重试，属已接受的 R2 残留。
_RLIMIT_AS_MB = 256
_RLIMIT_CPU_S = 10
_RLIMIT_NPROC = 32

_RLIMIT_PREAMBLE = (
    "import resource\n"
    f"resource.setrlimit(resource.RLIMIT_AS, "
    f"({_RLIMIT_AS_MB} * 1024 * 1024, {_RLIMIT_AS_MB} * 1024 * 1024))\n"
    f"resource.setrlimit(resource.RLIMIT_CPU, ({_RLIMIT_CPU_S}, {_RLIMIT_CPU_S}))\n"
    f"resource.setrlimit(resource.RLIMIT_NPROC, ({_RLIMIT_NPROC}, {_RLIMIT_NPROC}))\n"
)

# 最小子进程环境：不透传容器 env（SANDBOX_TOKEN 绝不进学生代码进程）；
# 线程数封 1 防 BLAS/OpenMP 线程池绕过 CPU 限制（human-eval reliability_guard 同款）。
_MINIMAL_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

_GATE = asyncio.Semaphore(_CONCURRENCY)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class ExecuteRequest(BaseModel):
    code: str
    timeout_s: int = Field(default=5, gt=0, le=TIMEOUT_MAX_S)


class ExecuteResponse(BaseModel):
    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    detail: str = ""


def _truncate(raw: bytes) -> str:
    return raw[:OUTPUT_MAX_BYTES].decode("utf-8", errors="replace")


async def _execute(code: str, timeout_s: int) -> ExecuteResponse:
    # prime：临时目录即 box（POSIX TemporaryDirectory 默认 0700），snippet 0600
    with tempfile.TemporaryDirectory(prefix="exec_") as td:
        path = Path(td) / "snippet.py"
        path.write_text(_RLIMIT_PREAMBLE + code, encoding="utf-8")
        path.chmod(0o600)
        # execute：墙钟超时（对齐 isolate --wall-time），超时 kill 直 child 并回收
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-X", "utf8", str(path),
            cwd=td,
            env=_MINIMAL_ENV,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_s + 1)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecuteResponse(
                ok=False, timed_out=True, detail=f"timeout>{timeout_s}s"
            )
    # cleanup：TemporaryDirectory 上下文退出即清（对齐 isolate --cleanup 逐 box 清）
    exit_code = proc.returncode
    return ExecuteResponse(
        ok=exit_code == 0,
        exit_code=exit_code,
        stdout=_truncate(out),
        stderr=_truncate(err),
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(
    req: ExecuteRequest, x_sandbox_token: str = Header(default="")
) -> ExecuteResponse:
    if not _TOKEN or not hmac.compare_digest(x_sandbox_token, _TOKEN):
        raise HTTPException(status_code=403, detail="invalid sandbox token")
    if len(req.code.encode("utf-8")) > CODE_MAX_BYTES:
        raise HTTPException(status_code=400, detail="code too large")
    async with _GATE:
        return await _execute(req.code, req.timeout_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if not _TOKEN:
        logger.error("SANDBOX_TOKEN 未设置：全部 /execute 将拒绝（fail-closed）")
    sock = Path(_SOCKET_PATH)
    if sock.exists():
        sock.unlink()  # 上次退出残留的 socket 文件会让 bind 失败
    uvicorn.run(app, uds=str(sock), loop="asyncio", log_level="info", access_log=False)


if __name__ == "__main__":
    main()
