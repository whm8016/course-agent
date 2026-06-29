"""请求级上下文变量（contextvars）。

asyncio.create_task() 会在创建任务时复制当前 context，因此在 TurnRuntimeManager
调用 create_task 之前执行 bind_context()，后台任务就能自动继承 turn_id 等字段。
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_REQUEST_ID: ContextVar[str] = ContextVar("obs_request_id", default="")
_TURN_ID: ContextVar[str] = ContextVar("obs_turn_id", default="")
_USER_ID: ContextVar[str] = ContextVar("obs_user_id", default="")
_COURSE_ID: ContextVar[str] = ContextVar("obs_course_id", default="")
_MODE: ContextVar[str] = ContextVar("obs_mode", default="")
_JOB_ID: ContextVar[str] = ContextVar("obs_job_id", default="")

_ALL_VARS: dict[str, ContextVar[str]] = {
    "request_id": _REQUEST_ID,
    "turn_id": _TURN_ID,
    "user_id": _USER_ID,
    "course_id": _COURSE_ID,
    "mode": _MODE,
    "job_id": _JOB_ID,
}


def bind_context(**kwargs: str) -> None:
    """在当前 async 上下文中绑定字段。

    只设置传入的 key，不清除其他字段，方便分层调用：
      - middleware 设 request_id
      - TRM.start_turn 追加 turn_id / user_id / course_id / mode
      - worker job 追加 job_id
    """
    for key, value in kwargs.items():
        var = _ALL_VARS.get(key)
        if var is not None:
            var.set(str(value) if value is not None else "")


def get_context_fields() -> dict[str, Any]:
    """返回当前上下文中所有非空字段（供 ContextFilter 注入 LogRecord）。"""
    return {key: var.get() for key, var in _ALL_VARS.items() if var.get()}
