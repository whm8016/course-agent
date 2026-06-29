"""Shared rate-limiter instance used by all routers."""
from __future__ import annotations

from starlette.requests import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import REDIS_URL, TESTING

# TESTING=1 disables rate-limiting so tests don't hit 429 on rapid registrations.
_TESTING = TESTING


def _key_func(request: Request) -> str:
    """Rate-limit by user_id when authenticated, fall back to IP."""
    if _TESTING:
        return "testing-unlimited"
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        try:
            from core.db.auth import decode_token
            payload = decode_token(auth[7:])
            if payload and payload.get("sub"):
                return f"user:{payload['sub']}"
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(
    key_func=_key_func,
    storage_uri=REDIS_URL,
    enabled=not _TESTING,
)
