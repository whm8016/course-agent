"""Cron service for scheduling agent tasks."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Coroutine
import uuid

from .types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter
            from datetime import datetime

            base_time = now_ms / 1000
            base_dt = datetime.fromtimestamp(base_time)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None

    return None


class CronService:
    """Manages scheduled tasks for the bot."""

    def __init__(
        self,
        workspace: Path,
        on_execute: Callable[[str, str, str], Coroutine[Any, Any, str]] | None = None,
    ):
        self.workspace = workspace
        self.cron_dir = workspace / "cron"
        self.cron_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = self.cron_dir / "jobs.json"
        self.on_execute = on_execute
        self._store = self._load_store()
        self._task: asyncio.Task | None = None
        self._running = False

    def _load_store(self) -> CronStore:
        if not self._store_path.exists():
            return CronStore()
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            jobs = []
            for j in data.get("jobs", []):
                schedule = CronSchedule(**j.get("schedule", {}))
                payload = CronPayload(**j.get("payload", {}))
                jobs.append(CronJob(
                    id=j["id"],
                    name=j.get("name", ""),
                    schedule=schedule,
                    payload=payload,
                    state=CronJobState(j.get("state", "pending")),
                    next_run_ms=j.get("next_run_ms"),
                    last_run_ms=j.get("last_run_ms"),
                    created_at_ms=j.get("created_at_ms", 0),
                    run_count=j.get("run_count", 0),
                ))
            return CronStore(jobs=jobs)
        except Exception:
            return CronStore()

    def _save_store(self) -> None:
        data = {"jobs": []}
        for j in self._store.jobs:
            data["jobs"].append({
                "id": j.id,
                "name": j.name,
                "schedule": {"kind": j.schedule.kind, "at_ms": j.schedule.at_ms, "every_ms": j.schedule.every_ms, "expr": j.schedule.expr, "tz": j.schedule.tz},
                "payload": {"task": j.payload.task, "channel": j.payload.channel, "chat_id": j.payload.chat_id},
                "state": j.state.value,
                "next_run_ms": j.next_run_ms,
                "last_run_ms": j.last_run_ms,
                "created_at_ms": j.created_at_ms,
                "run_count": j.run_count,
            })
        self._store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Cron service started (%d jobs)", len(self._store.jobs))

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(30)
                if not self._running:
                    break
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Cron tick error")

    async def _tick(self) -> None:
        now = _now_ms()
        for job in self._store.jobs:
            if job.state in (CronJobState.DONE, CronJobState.CANCELLED):
                continue
            if job.next_run_ms is None:
                job.next_run_ms = _compute_next_run(job.schedule, now)
                continue
            if now >= job.next_run_ms:
                await self._run_job(job, now)

    async def _run_job(self, job: CronJob, now: int) -> None:
        job.state = CronJobState.RUNNING
        job.last_run_ms = now
        job.run_count += 1

        if self.on_execute:
            try:
                await self.on_execute(job.payload.task, job.payload.channel, job.payload.chat_id)
            except Exception:
                logger.exception("Cron job %s failed", job.id)
                job.state = CronJobState.FAILED

        if job.schedule.kind == "at":
            job.state = CronJobState.DONE
        else:
            job.state = CronJobState.PENDING
            job.next_run_ms = _compute_next_run(job.schedule, now)

        self._save_store()

    def add_job(self, name: str, schedule: CronSchedule, payload: CronPayload) -> CronJob:
        job = CronJob(
            id=uuid.uuid4().hex[:12],
            name=name,
            schedule=schedule,
            payload=payload,
            created_at_ms=_now_ms(),
            next_run_ms=_compute_next_run(schedule, _now_ms()),
        )
        self._store.jobs.append(job)
        self._save_store()
        return job

    def list_jobs(self) -> list[CronJob]:
        return self._store.jobs

    def cancel_job(self, job_id: str) -> bool:
        for j in self._store.jobs:
            if j.id == job_id:
                j.state = CronJobState.CANCELLED
                self._save_store()
                return True
        return False
