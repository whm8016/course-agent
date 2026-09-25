"""followup WS：鉴权用户 id 透传 FollowupAgent（usage 埋点归因维度）。

回归背景：ws_authenticate 返回 dict（键 "id"），旧代码 getattr(user, "user_id", "")
作用在 dict 上恒取默认 ""，track.user_id 永远为空——用量埋点无法归因到人。
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient


@pytest.fixture()
def app():
    from main import app as _app
    return _app


async def test_followup_ws_passes_authenticated_user_id(app, db, monkeypatch):
    from core.db.auth import create_token
    from core.db.database import AsyncSessionLocal, User

    async with AsyncSessionLocal() as s:
        s.add(User(id="u_follow", username="stu_follow", password_hash="x",
                   role="student"))
        await s.commit()

    captured: dict = {}

    class _Agent:
        def __init__(self, language="zh"):
            pass

        async def stream_process(self, *, user_message, question_context,
                                 history_context, user_id):
            captured["user_id"] = user_id
            yield "答案"

    monkeypatch.setattr("api.question.FollowupAgent", _Agent)
    token = create_token("u_follow", "stu_follow")
    tc = TestClient(app)
    with tc.websocket_connect(f"/api/question/followup?token={token}") as ws:
        ws.send_json({"user_message": "为什么", "question_context": {"q": "1+1"}})
        msgs = []
        while True:
            msg = ws.receive_json()
            msgs.append(msg)
            if msg.get("type") in ("done", "error"):
                break

    assert captured["user_id"] == "u_follow"  # 修复前恒为 ""
    assert msgs[-1]["type"] == "done"
    assert any(m["type"] == "answer" and m["content"] == "答案" for m in msgs)
