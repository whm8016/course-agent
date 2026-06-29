"""cron_tool 单测：schedule 解析（纯函数）+ owner 缺失保护 + 未知动作。

不触碰 CronService 持久化（无 owner / 未知动作分支早返回，不调 service.add_job/list）。
"""
import pytest

from core.bot.cron_tool import _build_schedule, reset_cron_owner, run_cron_action, set_cron_owner


def test_build_schedule_at():
    s = _build_schedule({"at": "2026-06-12T09:00"})
    assert s.kind == "at"
    assert s.at_ms is not None


def test_build_schedule_every():
    s = _build_schedule({"every_seconds": 60})
    assert s.kind == "every"
    assert s.every_seconds == 60


def test_build_schedule_cron():
    s = _build_schedule({"cron_expr": "0 9 * * *"})
    assert s.kind == "cron"
    assert s.expr == "0 9 * * *"


def test_build_schedule_requires_exactly_one():
    with pytest.raises(ValueError):
        _build_schedule({})
    with pytest.raises(ValueError):
        _build_schedule({"at": "2026-06-12T09:00", "every_seconds": 60})


def test_no_owner_unavailable():
    """无 cron owner（web /api/chat 不挂载 cron）→ 不可用。"""
    ok, text = run_cron_action({"action": "list"})
    assert ok is False
    assert "不支持" in text


def test_unknown_action_with_owner():
    """有 owner 但动作非法 → 未知动作（不调 service）。"""
    token = set_cron_owner({
        "partner_id": "owner1:bot1", "channel": "qq", "chat_id": "grp1",
        "session_key": "qq:grp1", "user_id": "u1",
    })
    try:
        ok, text = run_cron_action({"action": "bogus"})
        assert ok is False
        assert "未知动作" in text
    finally:
        reset_cron_owner(token)


def test_schedule_missing_message():
    """schedule 但无 message → 失败提示（不调 service.add_job）。"""
    token = set_cron_owner({
        "partner_id": "owner1:bot1", "channel": "qq", "chat_id": "grp1",
        "session_key": "qq:grp1", "user_id": "u1",
    })
    try:
        ok, text = run_cron_action({"action": "schedule", "every_seconds": 60})
        assert ok is False
        assert "message" in text
    finally:
        reset_cron_owner(token)


def test_build_schedule_at_relative():
    """相对时间 at（in 30s）从服务器当前时间起算，agent 无需知道「现在」。"""
    import time as _time
    before = _time.time()
    s = _build_schedule({"at": "in 30s"})
    after = _time.time()
    assert s.kind == "at"
    assert s.at_ms is not None
    assert before * 1000 + 29000 <= s.at_ms <= after * 1000 + 31000


def test_build_schedule_at_plus_and_unit_forms():
    """+5m / 2h / 1d 都能解析。"""
    for expr in ("+5m", "2h", "1d", "in 45s"):
        s = _build_schedule({"at": expr})
        assert s.kind == "at"
        assert s.at_ms is not None
