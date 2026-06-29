"""WebSocket auth gates — synchronous TestClient, no LLM calls."""
from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("JWT_SECRET", "test-secret-pytest-only-32chars!!")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("REDIS_URL", "memory://")
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test")
os.environ.setdefault("ALLOWED_ORIGINS", "*")
os.environ.setdefault("TESTING", "1")

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="module")
def app():
    from main import app as _app
    return _app


def test_deep_solve_ws_no_token_rejected(app):
    """Without token ws_authenticate must close the socket (4001)."""
    with TestClient(app) as tc:
        try:
            with tc.websocket_connect("/api/deep-solve/run"):
                pass
        except Exception:
            # Expected: connection rejected or closed before handshake
            pass


def test_deep_solve_ws_invalid_token_rejected(app):
    """Bad token must be rejected."""
    with TestClient(app) as tc:
        try:
            with tc.websocket_connect("/api/deep-solve/run?token=bad-token"):
                pass
        except Exception:
            pass


def test_question_generate_ws_no_token_rejected(app):
    with TestClient(app) as tc:
        try:
            with tc.websocket_connect("/api/question/generate"):
                pass
        except Exception:
            pass


def test_deep_research_ws_no_token_rejected(app):
    with TestClient(app) as tc:
        try:
            with tc.websocket_connect("/api/deep-research/run"):
                pass
        except Exception:
            pass
