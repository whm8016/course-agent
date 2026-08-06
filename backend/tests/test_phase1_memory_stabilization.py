"""Phase 1 止血：flush 失败不再丢数据 + graph 节流计数改 Redis INCR。

覆盖三处修复：
1. _flush_turns 返回 bool——mem0 关键写失败返回 False（不再静默吞），_flush_one 据此
   保留 Redis key 等下次重试（修「写失败即永久丢数据」，H-7 被架空的根因）。
2. _flush_one 在 _flush_turns 返回 False 时显式保留 key（区别于抛异常路径）。
3. graph_memory 节流计数从模块级全局 dict 改 Redis INCR（跨 gunicorn/ARQ worker 共享）。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory import flush_manager, graph_memory


# ── 1. _flush_turns 返回值契约 ──────────────────────────────────────────────


class _FakeSession:
    """模拟 AsyncSessionLocal：async context manager + no-op commit/rollback。"""

    def __init__(self):
        self._db = MagicMock()
        self._db.commit = AsyncMock()
        self._db.rollback = AsyncMock()

    async def __aenter__(self):
        return self._db

    async def __aexit__(self, *exc):
        return False


async def test_flush_turns_true_on_mem0_success():
    """mem0.add 正常返回 → _flush_turns 返回 True（可安全删 key）。"""
    with patch("core.db.database.AsyncSessionLocal", _FakeSession), \
         patch("core.memory.mem0_client.get_memory") as gm, \
         patch(
             "core.memory.graph_memory.update_graphs_from_conversation",
             AsyncMock(return_value=False),
         ):
        gm.return_value.add = AsyncMock(return_value={"results": []})
        ok = await flush_manager._flush_turns("u1", "c1", [{"u": "q", "a": "a"}])
    assert ok is True


async def test_flush_turns_false_on_mem0_failure():
    """mem0.add 抛异常 → _flush_turns 返回 False（_flush_one 须保留 key 重试）。"""
    with patch("core.memory.mem0_client.get_memory") as gm:
        gm.return_value.add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        ok = await flush_manager._flush_turns("u1", "c1", [{"u": "q", "a": "a"}])
    assert ok is False


async def test_flush_turns_skips_graph_when_mem0_fails():
    """mem0 失败时跳过 graph（避免整批重试时白烧一次 LLM 提取）。"""
    with patch("core.memory.mem0_client.get_memory") as gm, \
         patch(
             "core.memory.graph_memory.update_graphs_from_conversation",
             AsyncMock(),
         ) as graph_fn:
        gm.return_value.add = AsyncMock(side_effect=RuntimeError("mem0 down"))
        await flush_manager._flush_turns("u1", "c1", [{"u": "q", "a": "a"}])
    graph_fn.assert_not_called()


async def test_flush_turns_empty_returns_true():
    """空批早退返回 True（无数据可丢，安全删 key，与旧「无条件删」语义一致）。"""
    ok = await flush_manager._flush_turns("u1", "c1", [])
    assert ok is True


# ── 2. _flush_one 保留 key（False 路径，区别于抛异常路径）────────────────────


def _make_redis(store: dict) -> SimpleNamespace:
    """H-7/M-13 同款的内存版 mock redis（scan/set/get/delete/lrange/llen 子集）。"""

    async def _scan(cursor=0, match=None, count=100):  # noqa: ARG001
        import fnmatch
        return (0, [k for k in store if fnmatch.fnmatch(k, match)])

    async def _set(k, v, ex=None, nx=False):  # noqa: ARG001
        if nx and k in store:
            return False
        store[k] = v
        return True

    async def _get(k):
        return store.get(k)

    async def _delete(*keys):
        return sum(1 for k in keys if store.pop(k, None) is not None)

    async def _lrange(k, a, b):
        return list(store.get(k, []))[a : (b + 1 if b != -1 else None)]

    async def _llen(k):
        return len(store.get(k, []))

    return SimpleNamespace(
        scan=AsyncMock(side_effect=_scan), set=AsyncMock(side_effect=_set),
        get=AsyncMock(side_effect=_get), delete=AsyncMock(side_effect=_delete),
        lrange=AsyncMock(side_effect=_lrange), llen=AsyncMock(side_effect=_llen),
    )


async def test_flush_one_keeps_key_when_critical_write_fails():
    """Phase 1 新路径：_flush_turns 返回 False（非抛异常）→ key 必须保留待重试。"""
    store = {
        "mem_flush:u1:s1": ['{"u":"q","a":"a"}'] * 3,
        "mem_flush:u1:s1:ts": "0",
        "mem_flush:u1:s1:meta": '{"user_id":"u1","course_id":"c1"}',
    }
    r = _make_redis(store)
    data_key = "mem_flush:u1:s1"

    with patch.object(flush_manager, "_flush_turns", AsyncMock(return_value=False)):
        n = await flush_manager.scan_and_flush(r, max_turns=1, idle_timeout=1.0)

    assert n == 0  # 未成功 flush
    assert data_key in store, "关键写失败返回 False 时 key 必须保留待重试（Phase 1）"
    assert len(store[data_key]) == 3
    assert f"{data_key}:ts" in store
    assert f"{data_key}:meta" in store


# ── 3. graph_memory 节流计数 → Redis INCR（跨进程共享）──────────────────────


async def test_incr_turn_count_uses_redis_pipeline():
    """_incr_turn_count 对 graph_turn_count:{user_id} 做 INCR + EXPIRE，返回 INCR 结果。"""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[7, True])
    fake_redis = MagicMock()
    fake_redis.pipeline.return_value = pipe

    with patch("core.memory.flush_manager._get_redis", return_value=fake_redis):
        count = await graph_memory._incr_turn_count("u1")

    assert count == 7
    pipe.incr.assert_called_once_with("graph_turn_count:u1")
    pipe.expire.assert_called_once_with("graph_turn_count:u1", graph_memory._TURN_COUNT_TTL)
    pipe.execute.assert_awaited_once()


async def test_update_graphs_throttled_when_count_not_multiple():
    """count % 6 != 0 → 节流短路，不调 LLM 提取。"""
    with patch.object(graph_memory, "_incr_turn_count", AsyncMock(return_value=3)), \
         patch.object(graph_memory, "_extract_from_conversation", AsyncMock(return_value=None)) as ext:
        ret = await graph_memory.update_graphs_from_conversation(
            db=MagicMock(), user_id="u1", course_id="c1",
            user_message="什么是导数", assistant_answer="导数是变化率",
        )
    assert ret is False
    ext.assert_not_called()


async def test_update_graphs_extracts_when_count_multiple():
    """count % 6 == 0 → 节流放行，触发提取（提取返回空故最终 False，但提取确被调用）。"""
    with patch.object(graph_memory, "_incr_turn_count", AsyncMock(return_value=6)), \
         patch.object(graph_memory, "_extract_from_conversation", AsyncMock(return_value=None)) as ext:
        ret = await graph_memory.update_graphs_from_conversation(
            db=MagicMock(), user_id="u1", course_id="c1",
            user_message="什么是导数", assistant_answer="导数是变化率",
        )
    assert ret is False  # 提取返回 None → 无 kp/err → False
    ext.assert_awaited_once()
