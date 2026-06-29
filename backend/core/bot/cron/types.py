"""Cron job types."""

from dataclasses import dataclass, field
from enum import Enum


class CronJobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CronSchedule:
    kind: str = "every"  # "at" | "every" | "cron"
    at_ms: int | None = None
    every_ms: int | None = None
    expr: str | None = None
    tz: str | None = None


@dataclass
class CronPayload:
    task: str = ""
    channel: str = "web"
    chat_id: str = "web"


@dataclass
class CronJob:
    id: str
    name: str
    schedule: CronSchedule
    payload: CronPayload
    state: CronJobState = CronJobState.PENDING
    next_run_ms: int | None = None
    last_run_ms: int | None = None
    created_at_ms: int = 0
    run_count: int = 0


@dataclass
class CronStore:
    jobs: list[CronJob] = field(default_factory=list)
