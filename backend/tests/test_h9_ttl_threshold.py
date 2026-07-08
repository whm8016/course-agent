"""H-9: TTL 不应过短（原 600s/10min，worker 宕机 >10min 静默丢数据）。

修复：_TTL 延长到 86400s（24h），覆盖夜间故障窗口；数据量小不会撑爆 Redis。

断言：_TTL >= 86400（24h 兜底）。
"""
import core.memory.flush_manager as fm


def test_ttl_at_least_24_hours():
    """H-9: 兜底 TTL 必须 >= 86400s（24h），防止 worker 宕机丢数据。"""
    assert fm._TTL >= 86400, (
        f"_TTL={fm._TTL}s 过短，worker 宕机超过该时长即静默丢待落盘数据；要求 >= 86400s(24h)"
    )


def test_ttl_not_infinite():
    """H-9: TTL 仍应有上限（防泄漏兜底，不能是 None/0=永久）。"""
    assert fm._TTL is not None
    assert fm._TTL > 0
    assert fm._TTL <= 7 * 86400, "_TTL 过大反而失去兜底防泄漏意义（应 <= 7 天）"
