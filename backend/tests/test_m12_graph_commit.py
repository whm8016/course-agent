"""M-12: graph_memory flush 后必须有 commit，否则 graph 更新全部静默丢失。

根因：_flush_turns 第 142 行 `async with AsyncSessionLocal() as db:` 块内循环调
update_graphs_from_conversation（内部 save_graphs 只 execute(update)，不 commit）。
而 AsyncSessionLocal 的 __aexit__ 在没有显式 commit 时会 rollback → graph 更新被回滚。

修复：循环结束后补 `await db.commit()`。
"""
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.flush_manager import _flush_turns


async def test_flush_turns_commits_graph_changes():
    """M-12: _flush_turns 结束时必须调用 db.commit()。"""
    commit_mock = AsyncMock()
    rollback_mock = AsyncMock()

    # 构造一个 async context manager，退出时返回自身（模拟 AsyncSessionLocal）
    fake_db = MagicMock()
    fake_db.commit = commit_mock
    fake_db.rollback = rollback_mock

    class _FakeSession:
        def __init__(self):
            self._db = fake_db

        async def __aenter__(self):
            return self._db

        async def __aexit__(self, *exc):
            return False

    turns = [{"u": "什么是导数", "a": "导数是变化率..."}]

    with patch("core.db.database.AsyncSessionLocal", _FakeSession), \
         patch("core.memory.mem0_client.get_memory") as gm, \
         patch(
             "core.memory.graph_memory.update_graphs_from_conversation",
             AsyncMock(return_value=True),
         ):
        # mem0 get_memory().add() 返回空结果即可，避免真实 LLM/PG 调用
        gm.return_value.add = AsyncMock(return_value={"results": []})

        await _flush_turns("u1", "c1", turns)

    commit_mock.assert_awaited_once(), "graph 更新后必须 commit，否则被 __aexit__ rollback 丢失"


async def test_flush_turns_empty_turns_no_commit():
    """M-12: 空 turns 时 _flush_turns 提前返回，不应创建 session/commit（边界）。"""
    with patch("core.db.database.AsyncSessionLocal") as session_cls, \
         patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock()
        await _flush_turns("u1", "c1", [])

    # 空 turns 早退（_flush_turns 开头 if not turns: return），不应触碰 DB
    session_cls.assert_not_called()
