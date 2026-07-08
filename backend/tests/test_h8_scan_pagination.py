"""H-8: SCAN 必须循环 cursor 直到归 0，否则超 ~count 的 key 被漏扫。

根因：原 scan_and_flush 第 178 行 `cursor, keys = await r.scan(match=..., count=200)`
只调一次。Redis SCAN 在大 keyspace 下会分批，单次只返回第一批（~count 个），
后续页的 buffer key 永远不被 flush。

修复：_scan_all_keys 用 `while cursor:` 翻页直到游标归 0。
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.memory.flush_manager import _scan_all_keys


def _paged_redis(pages: list[tuple[int, list[str]]]):
    """按预设的 (cursor, keys) 序列模拟 SCAN 分页。

    pages 形如 [(7, ["k0".."k199"]), (3, ["k200".."k299"]), (0, ["k300".."k399"])]：
    第一页返回 cursor=7（还有），第二页 cursor=3，第三页 cursor=0（结束）。
    """

    calls = {"i": 0}

    async def _scan(cursor=0, match=None, count=100):  # noqa: ARG001
        idx = calls["i"]
        calls["i"] += 1
        assert idx < len(pages), f"SCAN 被调了第 {idx + 1} 次，超出预设 {len(pages)} 页"
        # 校验上次返回的 cursor 被正确回传
        returned_cursor = pages[idx][0]
        if idx > 0:
            assert cursor == pages[idx - 1][0], (
                f"第 {idx} 次 SCAN 应带上一次的 cursor={pages[idx-1][0]}，实际={cursor}"
            )
        return returned_cursor, list(pages[idx][1])

    return SimpleNamespace(scan=AsyncMock(side_effect=_scan)), calls


async def test_scan_paginates_until_cursor_zero():
    """H-8: 多页 cursor 时，_scan_all_keys 必须翻完所有页。"""
    all_keys = [f"mem_flush:k{i}" for i in range(400)]
    page1 = all_keys[:200]
    page2 = all_keys[200:300]
    page3 = all_keys[300:]

    r, _ = _paged_redis([(7, page1), (3, page2), (0, page3)])
    got = await _scan_all_keys(r, match="mem_flush:*", count=200)

    assert r.scan.await_count == 3, "应循环 3 次（直到 cursor 归 0），原 bug 只调 1 次"
    assert sorted(got) == sorted(all_keys), "400 个 key 必须全部回收，漏页即丢数据"


async def test_scan_single_page_still_works():
    """H-8: 单页（cursor 立即归 0）正常返回。"""
    keys = ["mem_flush:a", "mem_flush:b"]
    r, _ = _paged_redis([(0, keys)])
    got = await _scan_all_keys(r, match="mem_flush:*", count=200)

    assert r.scan.await_count == 1
    assert sorted(got) == sorted(keys)


async def test_scan_empty():
    """H-8: 空 keyspace 也只调一次且返回空。"""
    r, _ = _paged_redis([(0, [])])
    got = await _scan_all_keys(r, match="mem_flush:*", count=200)

    assert got == []
    assert r.scan.await_count == 1
