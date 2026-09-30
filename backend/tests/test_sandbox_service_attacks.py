"""沙箱容器攻击面回归（pytest -m sandbox_service；需先 docker compose up -d sandbox）。

宿主（Windows）连不进容器卷里的 uds，故经 `docker compose exec` 把
sandbox_selftest.py 从 stdin 注入沙箱容器内执行（纯 stdlib，自检脚本读容器自身
的 SANDBOX_TOKEN）。服务未启动时 skip 并给出启动命令；默认被 pyproject
addopts 排除，CI（无 docker）不受影响。
"""
from __future__ import annotations

import pathlib
import subprocess

import pytest

pytestmark = pytest.mark.sandbox_service

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SELFTEST = pathlib.Path(__file__).parent / "sandbox_selftest.py"


def _compose(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )


@pytest.fixture(scope="module")
def sandbox_running() -> None:
    probe = _compose("exec", "-T", "sandbox", "true")
    if probe.returncode != 0:
        pytest.skip(
            "sandbox 服务未运行；先 `docker compose up -d --build sandbox` 后"
            " `pytest -m sandbox_service`"
        )


def test_sandbox_container_attack_suite(sandbox_running):
    """攻击面自检整体作为一条 pytest 断言：非零退出 = 有隔离性质失守。"""
    result = _compose(
        "exec", "-T", "sandbox", ".venv/bin/python", "-",
        input=_SELFTEST.read_text(encoding="utf-8"),
        timeout=300,
    )
    assert result.returncode == 0, (
        f"沙箱攻击面自检失败:\n{result.stdout}\n{result.stderr}"
    )
    # 逐 case 的 PASS 行至少覆盖已知的隔离维度，防自检脚本被掏空后假绿
    for name in (
        "env_has_no_credentials",
        "no_env_file_or_secrets_in_fs",
        "runs_as_non_root",
        "no_outbound_tcp_or_dns",
        "fork_bomb_capped_and_service_survives",
        "memory_bomb_killed_by_rlimit",
        "cpu_spin_hits_wall_timeout",
        "stdout_truncated",
        "token_required",
        "invalid_input_rejected",
        "concurrent_requests_queue_not_reject",
    ):
        assert f"PASS {name}" in result.stdout, f"缺少 case: {name}"


def test_sandbox_healthcheck_passing(sandbox_running):
    """compose healthcheck 视角：uds 可连（服务活着）且容器状态 healthy。"""
    result = _compose("ps", "--format", "json", "sandbox")
    assert result.returncode == 0, result.stderr
    # docker compose ps --format json 在单服务时输出一个 JSON 对象（v2 近版）
    # 或 JSON Lines；两种形态都取 health 字段
    import json as _json

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    states = [_json.loads(ln).get("Health", "") for ln in lines]
    assert any(s == "healthy" for s in states), result.stdout
