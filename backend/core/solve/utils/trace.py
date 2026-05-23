"""Lightweight trace metadata helpers (DeepTutor-compatible stubs)."""

from __future__ import annotations

import uuid
from typing import Any


def new_call_id(prefix: str = "call") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def build_trace_metadata(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def derive_trace_metadata(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    out = dict(base)
    out.update(updates)
    return out


__all__ = ["new_call_id", "build_trace_metadata", "derive_trace_metadata"]
