"""INFO-level helpers for observing question-flow context and prompt shapes in backend logs."""

from __future__ import annotations

import logging
from typing import Any

_HEAD = 400


def _squash_ws(s: str) -> str:
    return " ".join((s or "").split())


def clip_text(s: str, n: int = _HEAD) -> str:
    t = _squash_ws(s)
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def log_question_flow(logger: logging.Logger, stage: str, **fields: Any) -> None:
    """Emit one line: stage + scalar fields; string fields also get _chars and _head."""
    parts: list[str] = [f"stage={stage}"]
    for key, val in fields.items():
        if val is None:
            continue
        if isinstance(val, (int, float, bool)):
            parts.append(f"{key}={val}")
            continue
        s = str(val)
        parts.append(f"{key}_chars={len(s)}")
        parts.append(f"{key}_head={clip_text(s)}")
    logger.info("[question-flow] %s", " | ".join(parts))
