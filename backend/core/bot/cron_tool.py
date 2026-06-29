"""cron 工具 — agent 在对话里设定时提醒（对标 DeepTutor ``tools/cron_tool.py``）。

agent 调 ``cron`` 工具（action=schedule/list/cancel），owner 取**当前会话上下文**
（partner_id=owner:bot_id + channel/chat_id/session_key/user_id），经 contextvar
注入（``ChatPipeline.run`` set，与 solve session_id 同款注入模式 —— execute_tool 不
收 context，靠 contextvar 传）。

到点 ``CronService`` → ``services/cron/executor.execute_job`` → bot
``AgentLoop.process_direct`` → 按 channel 发回 IM / web 通知（executor.py 已实现）。

挂载范围：仅 **bot 对话**（``AgentLoop._enabled_tools`` 含 ``cron``）。web 对话
（/api/chat）不挂载 —— web 用前端 BotReminderPanel 配置定时。
"""
from __future__ import annotations

import contextvars
import logging
import re
import time
from datetime import datetime
from typing import Any

from core.agent.tool_protocol import ToolResult
from services.cron.service import CronOwner, CronSchedule, get_cron_service

logger = logging.getLogger(__name__)

# 当前会话的 cron owner（dict：partner_id/channel/chat_id/session_key/user_id）
_current_owner: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cron_current_owner", default=None
)


def set_cron_owner(owner: dict[str, Any] | None) -> contextvars.Token:
    return _current_owner.set(owner)


def reset_cron_owner(token: contextvars.Token) -> None:
    _current_owner.reset(token)


def current_cron_owner() -> dict[str, Any] | None:
    return _current_owner.get()


# ── 调度描述 / 解析 ────────────────────────────────────────────────────────

def _fmt_ms(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M")


def _describe_schedule(schedule: CronSchedule) -> str:
    if schedule.kind == "at":
        return f"一次性 {_fmt_ms(schedule.at_ms)}"
    if schedule.kind == "every":
        return f"每 {schedule.every_seconds} 秒"
    tz_part = f" ({schedule.tz})" if schedule.tz else ""
    return f"cron `{schedule.expr}`{tz_part}"


_REL_RE = re.compile(r"^(?:in\s+|\+)?\s*(\d+)\s*([smhd])$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_at(value: str) -> int:
    """解析 at：支持相对（in 30s / +5m / 2h / 1d）与绝对 ISO 8601（2026-06-12T09:00）。

    相对时间从**服务器当前时间**起算（agent 不需要知道「现在」）。
    """
    v = value.strip()
    rel = _REL_RE.match(v)
    if rel:
        n = int(rel.group(1))
        return int((time.time() + n * _UNIT_SECONDS[rel.group(2).lower()]) * 1000)
    try:
        parsed = datetime.fromisoformat(v)
    except ValueError:
        raise ValueError(
            f"无法解析时间 {value!r}——支持相对（in 30s / +5m / 2h / 1d）"
            f"或绝对 ISO 8601（2026-06-12T09:00）"
        ) from None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()  # naive → 服务器本地时区
    return int(parsed.timestamp() * 1000)


def _build_schedule(kwargs: dict[str, Any]) -> CronSchedule:
    at_raw = str(kwargs.get("at") or "").strip()
    every_raw = kwargs.get("every_seconds")
    expr = str(kwargs.get("cron_expr") or "").strip()
    chosen = [bool(at_raw), every_raw is not None, bool(expr)]
    if sum(chosen) != 1:
        raise ValueError("at / every_seconds / cron_expr 三选一（且只能选一个）")
    if at_raw:
        return CronSchedule(kind="at", at_ms=_parse_at(at_raw))
    if every_raw is not None:
        return CronSchedule(kind="every", every_seconds=int(every_raw))
    return CronSchedule(
        kind="cron", expr=expr, tz=str(kwargs.get("tz") or "").strip() or None
    )


# ── 核心动作 ────────────────────────────────────────────────────────────────

def run_cron_action(kwargs: dict[str, Any]) -> tuple[bool, str]:
    """执行 cron 动作；返回 (ok, text)。owner 从 contextvar 取（无则不可用）。"""
    owner_raw = current_cron_owner()
    if not isinstance(owner_raw, dict) or not owner_raw.get("partner_id"):
        return False, "当前会话不支持设定时（仅 IM bot 对话可用）。"
    owner = CronOwner(
        kind="partner",
        partner_id=owner_raw["partner_id"],
        channel=owner_raw.get("channel", ""),
        chat_id=owner_raw.get("chat_id", ""),
        session_key=owner_raw.get("session_key", ""),
        user_id=owner_raw.get("user_id", ""),
    )
    service = get_cron_service()
    action = str(kwargs.get("action") or "").strip().lower()
    if action == "add":
        action = "schedule"
    elif action == "remove":
        action = "cancel"

    if action == "list":
        jobs = service.list_jobs(owner_key=owner.key)
        if not jobs:
            return True, "当前会话没有定时任务。"
        lines = [f"共 {len(jobs)} 个定时任务："] + [_render_job(j) for j in jobs]
        return True, "\n".join(lines)

    if action == "cancel":
        job_id = str(kwargs.get("job_id") or "").strip()
        if not job_id:
            return False, "取消需要 job_id（先 action=list 查看）。"
        if service.cancel_job(job_id, owner_key=owner.key):
            return True, f"已取消任务 `{job_id}`。"
        return False, f"未找到任务 `{job_id}`。"

    if action == "schedule":
        message = str(kwargs.get("message") or "").strip()
        if not message:
            return False, "设定时需要 message（到点提醒的内容）。"
        try:
            schedule = _build_schedule(kwargs)
            job = service.add_job(
                name=str(kwargs.get("name") or "").strip(),
                message=message,
                schedule=schedule,
                owner=owner,
            )
        except (ValueError, TypeError) as exc:
            return False, f"设定时失败：{exc}"
        return True, (
            f"已设定 **{job.name or '提醒'}**（`{job.id}`）— {_describe_schedule(schedule)}；"
            f"首次执行 {_fmt_ms(job.state.next_run_at_ms)}。到点会发到本会话。"
        )

    return False, f"未知动作 {action!r}（支持 schedule / list / cancel）。"


def _render_job(job) -> str:
    status = job.state.last_status or "pending"
    return (
        f"- `{job.id}` **{job.name or '提醒'}** — {_describe_schedule(job.schedule)}；"
        f"下次 {_fmt_ms(job.state.next_run_at_ms)}；上次：{status}"
    )


async def execute_cron(*, course_id: str = "", user_id: str = "", **kwargs: Any) -> ToolResult:
    """cron 工具 executor：读 contextvar owner → run_cron_action → ToolResult。"""
    ok, text = run_cron_action(kwargs)
    return ToolResult(content=text, success=ok)


CRON_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "cron",
        "description": (
            "设定 / 查看 / 取消定时提醒。到点会把提醒内容发到当前会话（QQ / 飞书群或私聊）。"
            "用户让你「每天 X 点提醒」「N 分钟后提醒」时用此工具。"
            "action=schedule 需 message + at/every_seconds/cron_expr 三选一；"
            "action=list 列出本会话定时；action=cancel 需 job_id。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["schedule", "list", "cancel"],
                    "description": "动作",
                },
                "message": {"type": "string", "description": "到点提醒的内容（schedule 必填）"},
                "name": {"type": "string", "description": "任务名（可选）"},
                "at": {
                    "type": "string",
                    "description": "一次性：相对（in 30s / +5m / 2h / 1d，从当前时间起算）或绝对 ISO 8601（2026-06-12T09:00）",
                },
                "every_seconds": {"type": "integer", "description": "周期：间隔秒数"},
                "cron_expr": {
                    "type": "string",
                    "description": "cron 表达式，如 0 9 * * *（每天 9 点）",
                },
                "tz": {"type": "string", "description": "cron 时区（如 Asia/Shanghai），可选"},
                "job_id": {"type": "string", "description": "cancel 时必填"},
            },
            "required": ["action"],
        },
    },
}


__all__ = [
    "set_cron_owner",
    "reset_cron_owner",
    "current_cron_owner",
    "run_cron_action",
    "execute_cron",
    "CRON_SCHEMA",
]
