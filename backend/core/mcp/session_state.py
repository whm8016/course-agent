"""per-session deferred 工具持久化（对标 DeepTutor ``services/mcp/session_state.py``）。

记录每个 chat session 已 load 的 deferred 工具名，使后续 turn 从一开始就含这些
schema，而不必重新 load。落 ``data/sessions/<session_id>/loaded_tools.json``。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from config import MCP_SESSIONS_DIR

logger = logging.getLogger(__name__)

_STATE_FILENAME = "loaded_tools.json"
_SESSIONS_BASE = Path(MCP_SESSIONS_DIR)


def _state_file(session_id: str) -> Path:
    sid = (session_id or "").strip() or "_anon"
    # 防 path 穿越：只保留字母数字、-、_
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in sid)
    return _SESSIONS_BASE / safe / _STATE_FILENAME


def load_loaded_tools(session_id: str) -> set[str]:
    if not session_id:
        return set()
    try:
        path = _state_file(session_id)
        if not path.exists():
            return set()
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("loaded-tools state unreadable for %s", session_id, exc_info=True)
        return set()
    names = data.get("loaded_tools") if isinstance(data, dict) else None
    if not isinstance(names, list):
        return set()
    return {str(n) for n in names if str(n).strip()}


def record_loaded_tools(session_id: str, names: set[str]) -> None:
    if not session_id:
        return
    try:
        path = _state_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"loaded_tools": sorted(names)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("failed to persist loaded-tools state for %s", session_id, exc_info=True)


__all__ = ["load_loaded_tools", "record_loaded_tools"]
