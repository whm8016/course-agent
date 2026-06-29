"""进程级 Cron 服务 — 定时任务调度。

精确对齐 DeepTutor services/cron/service.py。

单进程、asyncio.Event 驱动（精确睡眠，非固定轮询），JSON 持久化。

Job 归属：
  owner.kind == "partner"  →  由 bot 的 AgentLoop 执行，结果推向 QQ/飞书
  owner.kind == "chat"     →  （预留，当前课程 agent 暂不使用）

调用方：
  lifespan 中 await get_cron_service().start()
  api/bot.py 中 get_cron_service().add_job / list_jobs / cancel_job
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_SLEEP_SECONDS = 60.0
_MAX_RUN_HISTORY = 10


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── 数据模型 ─────────────────────────────────────────────────────────────


@dataclass
class CronSchedule:
    """When a job runs: one-shot, fixed interval, or cron expression."""

    kind: str  # "at" | "every" | "cron"
    at_ms: int | None = None          # "at": epoch ms
    every_seconds: int | None = None  # "every": interval in seconds
    expr: str | None = None           # "cron": e.g. "0 9 * * *"
    tz: str | None = None             # "cron": IANA timezone


@dataclass
class CronOwner:
    """Who scheduled the job and where its output goes."""

    kind: str           # "partner" | "chat"
    partner_id: str = ""      # partner: owning bot id
    channel: str = ""         # partner: originating channel ("qq" | "feishu" | "web")
    chat_id: str = ""         # partner: originating chat id
    session_key: str = ""     # partner: conversation key (optional override)
    channel_meta: dict[str, Any] = field(default_factory=dict)
    user_id: str = ""         # 创建提醒的学生 user_id（web 渠道落库 BotNotification 用）

    @property
    def key(self) -> str:
        if self.kind == "partner":
            return f"partner:{self.partner_id}"
        return "chat:local"


@dataclass
class CronRunRecord:
    run_at_ms: int
    status: str          # "ok" | "error" | "skipped"
    duration_ms: int = 0
    error: str | None = None


@dataclass
class CronJobState:
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: str | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass
class CronJob:
    id: str
    name: str
    message: str
    schedule: CronSchedule
    owner: CronOwner
    enabled: bool = True
    delete_after_run: bool = False
    created_at_ms: int = 0
    state: CronJobState = field(default_factory=CronJobState)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CronJob":
        state_raw = dict(data.get("state") or {})
        state_raw["run_history"] = [
            CronRunRecord(**r) for r in state_raw.get("run_history", [])
        ]
        return cls(
            id=str(data["id"]),
            name=str(data.get("name") or ""),
            message=str(data.get("message") or ""),
            schedule=CronSchedule(**(data.get("schedule") or {"kind": "every"})),
            owner=CronOwner(**(data.get("owner") or {"kind": "partner"})),
            enabled=bool(data.get("enabled", True)),
            delete_after_run=bool(data.get("delete_after_run", False)),
            created_at_ms=int(data.get("created_at_ms", 0)),
            state=CronJobState(**state_raw),
        )


# ── 调度逻辑 ─────────────────────────────────────────────────────────────


def compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_seconds or schedule.every_seconds <= 0:
            return None
        return now_ms + schedule.every_seconds * 1000

    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo
            from croniter import croniter

            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base = datetime.fromtimestamp(now_ms / 1000, tz=tz)
            next_dt = croniter(schedule.expr, base).get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except ImportError:
            raise ValueError(
                "cron 表达式需要 croniter 包 — 请改用 'every' 或 'at' 调度"
            ) from None
        except Exception as exc:
            raise ValueError(f"无效 cron 表达式 {schedule.expr!r}: {exc}") from None

    return None


def validate_schedule(schedule: CronSchedule) -> None:
    if schedule.kind == "at":
        if not schedule.at_ms:
            raise ValueError("'at' 调度需要 at_ms 时间戳")
        if schedule.at_ms <= _now_ms():
            raise ValueError("'at' 时间已过期")
        return
    if schedule.kind == "every":
        if not schedule.every_seconds or schedule.every_seconds < 30:
            raise ValueError("'every' 间隔至少 30 秒")
        return
    if schedule.kind == "cron":
        if compute_next_run(schedule, _now_ms()) is None:
            raise ValueError(f"cron 表达式 {schedule.expr!r} 永远不会触发")
        return
    raise ValueError(f"未知调度类型 {schedule.kind!r}")


# ── CronService ────────────────────────────────────────────────────────────


class CronService:
    """Single-process job store + scheduler.

    精确对齐 DeepTutor CronService API。
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Awaitable[tuple[str, str | None]]] | None = None,
    ) -> None:
        self.store_path = store_path
        self.on_job = on_job
        self._jobs: dict[str, CronJob] = {}
        self._loaded = False
        self._timer_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._running = False

    # ── 持久化 ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for raw in data.get("jobs", []):
                job = CronJob.from_dict(raw)
                self._jobs[job.id] = job
        except Exception:
            backup = self.store_path.with_suffix(f".corrupt-{int(time.time())}")
            try:
                self.store_path.rename(backup)
            except OSError:
                pass
            logger.exception("Cron store 损坏，已备份到 %s", backup)

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "jobs": [asdict(job) for job in self._jobs.values()]}
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    # ── Job 管理 ──────────────────────────────────────────────────────────

    def add_job(
        self,
        *,
        name: str,
        message: str,
        schedule: CronSchedule,
        owner: CronOwner,
        delete_after_run: bool | None = None,
    ) -> CronJob:
        self._load()
        validate_schedule(schedule)
        if not message.strip():
            raise ValueError("message 不能为空")
        job = CronJob(
            id=uuid.uuid4().hex[:10],
            name=name.strip() or message.strip()[:48],
            message=message.strip(),
            schedule=schedule,
            owner=owner,
            delete_after_run=(
                delete_after_run if delete_after_run is not None else schedule.kind == "at"
            ),
            created_at_ms=_now_ms(),
        )
        job.state.next_run_at_ms = compute_next_run(schedule, _now_ms())
        self._jobs[job.id] = job
        self._save()
        self._wake.set()
        return job

    def list_jobs(self, owner_key: str | None = None) -> list[CronJob]:
        self._load()
        jobs = list(self._jobs.values())
        if owner_key is not None:
            jobs = [j for j in jobs if j.owner.key == owner_key]
        return sorted(jobs, key=lambda j: j.state.next_run_at_ms or 0)

    def get_job(self, job_id: str) -> CronJob | None:
        self._load()
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str, *, owner_key: str | None = None) -> bool:
        self._load()
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if owner_key is not None and job.owner.key != owner_key:
            return False
        del self._jobs[job_id]
        self._save()
        self._wake.set()
        return True

    def remove_owner_jobs(self, owner_key: str) -> int:
        """删除属于指定 owner 的所有 job（如 bot 被销毁时）。"""
        self._load()
        doomed = [jid for jid, j in self._jobs.items() if j.owner.key == owner_key]
        for jid in doomed:
            del self._jobs[jid]
        if doomed:
            self._save()
            self._wake.set()
        return len(doomed)

    # ── 调度循环 ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._load()
        now = _now_ms()
        changed = False
        for job in list(self._jobs.values()):
            if job.schedule.kind == "at" and (job.schedule.at_ms or 0) <= now:
                del self._jobs[job.id]
                changed = True
        if changed:
            self._save()
        self._running = True
        self._timer_task = asyncio.create_task(self._loop(), name="cron:scheduler")
        logger.info("Cron service 已启动（%d 个任务）", len(self._jobs))

    async def stop(self) -> None:
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
            self._timer_task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cron tick 异常")
            sleep_s = self._seconds_until_next_due()
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=sleep_s)
            except asyncio.TimeoutError:
                pass

    def _seconds_until_next_due(self) -> float:
        due = [
            j.state.next_run_at_ms
            for j in self._jobs.values()
            if j.enabled and j.state.next_run_at_ms
        ]
        if not due:
            return _MAX_SLEEP_SECONDS
        delta = (min(due) - _now_ms()) / 1000
        return max(0.05, min(delta, _MAX_SLEEP_SECONDS))

    async def _tick(self) -> None:
        now = _now_ms()
        for job in list(self._jobs.values()):
            if not job.enabled or not job.state.next_run_at_ms:
                continue
            if job.state.next_run_at_ms > now:
                continue
            await self._run_job(job)

    async def _run_job(self, job: CronJob) -> None:
        started = _now_ms()
        status, error = "skipped", None
        if self.on_job is not None:
            try:
                status, error = await self.on_job(job)
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
                logger.exception("Cron job %s (%s) 崩溃", job.id, job.name)

        job.state.last_run_at_ms = started
        job.state.last_status = status
        job.state.last_error = error
        job.state.run_history.append(
            CronRunRecord(
                run_at_ms=started,
                status=status,
                duration_ms=_now_ms() - started,
                error=error,
            )
        )
        job.state.run_history = job.state.run_history[-_MAX_RUN_HISTORY:]

        if job.delete_after_run or job.schedule.kind == "at":
            self._jobs.pop(job.id, None)
        else:
            job.state.next_run_at_ms = compute_next_run(job.schedule, _now_ms())
            if job.state.next_run_at_ms is None:
                self._jobs.pop(job.id, None)
        self._save()


# ── 进程单例 ───────────────────────────────────────────────────────────────

_service: CronService | None = None


def get_cron_service() -> CronService:
    """进程级 CronService 单例（懒加载）。store 路径：data/cron/jobs.json。"""
    global _service
    if _service is None:
        from services.cron.executor import execute_job

        store = Path("data/cron/jobs.json")
        _service = CronService(store_path=store, on_job=execute_job)
    return _service


__all__ = [
    "CronJob",
    "CronOwner",
    "CronSchedule",
    "CronService",
    "compute_next_run",
    "get_cron_service",
    "validate_schedule",
]
