"""H-14：IM 消息去重接入 Redis（channels/dedup.py）。

根因：``QQChannel._processed_ids``（deque）、``FeishuChannel._processed_message_ids``
（OrderedDict）都是**进程内存**——多 worker 部署下 leader failover 后，新 leader 的
内存里没有旧 leader 已处理的 message_id；IM 平台（QQ/飞书）在网络抖动/重连时会**重发**
近期消息，导致同一消息被处理两次（重复回复 + 重复触发工具/cron）。

修法：删除内存去重，统一接入 ``channels/dedup.claim_processed``（原子 ``SET NX EX``）。
去重状态存 Redis（所有 worker 共享），leader 切换后命中同一 key，平台重发被拦截。

failover + 重发时序推演（claim_processed 原子，无 TOCTOU）：
  T0: leader A 收到 QQ 消息 msg#1，claim_processed("qq","1") → SET NX 成功 → 处理 + 回复。
  T1: A 的 leader lease TTL 过期（A 卡死），follower C 竞选接管 → on_gain 拉起 bot。
  T2: QQ 平台因 A 重连断开，**重发** msg#1 给 C。
  T3: C 调 claim_processed("qq","1") → SET NX 失败（key 已存在，TTL 1h 未过）→ 返回 False
      → _on_message 早退，不重复处理。✓ 无重复回复。
  T4: C 收到新消息 msg#2 → claim_processed("qq","2") → 成功 → 正常处理。✓ 不丢消息。

并发回调（同一 message_id 两次 _on_message 几乎同时）：
  SET NX 是 Redis 单命令原子，两次调用只可能一次 True 一次 False → 只处理一次。✓

Redis 不可用降级：claim_processed 捕获异常返回 True（允许处理）——最坏 failover 窗口内
重复回一条消息，不损坏数据，符合「不丢消息」优先原则。
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.bot.bus.queue import MessageBus  # noqa: E402
from core.bot.channels.feishu import FeishuChannel, FeishuConfig  # noqa: E402
from core.bot.channels.qq import QQChannel, QQConfig  # noqa: E402


class _FakeAuthor:
    def __init__(self, *, openid: str = "u1", member_openid: str = "u1", uid: str = ""):
        self.id = openid or None
        self.user_openid = openid
        self.member_openid = member_openid


class _FakeQQMsg:
    """模拟 botpy GroupMessage / C2CMessage 的最小字段。"""

    def __init__(self, msg_id: str, content: str = "hi", is_group: bool = True):
        self.id = msg_id
        self.content = content
        self.group_openid = "g1" if is_group else None
        self.author = _FakeAuthor()


async def _qq_consume(bus: MessageBus):
    """非阻塞取一条 inbound（若有），返回 None 表示队列空。"""
    try:
        return bus.inbound.get_nowait()
    except Exception:
        return None


async def test_qq_uses_redis_dedup_first_call_processed():
    """QQ 第一次收到 msg → claim_processed 返回 True → 消息进入 bus（被处理）。"""
    ch = QQChannel(QQConfig(allow_from=["*"]), MessageBus())
    data = _FakeQQMsg("m1", content="hello")

    with patch(
        "core.bot.channels.dedup.claim_processed", new=AsyncMock(return_value=True)
    ) as spy:
        await ch._on_message(data, is_group=True)

    spy.assert_awaited_once_with("qq", "m1")
    msg = await _qq_consume(ch.bus)
    assert msg is not None and msg.content == "hello"


async def test_qq_redis_dedup_skips_already_processed():
    """QQ 同 msg 第二次 → claim_processed 返回 False → 早退，不进 bus（不重复处理）。"""
    ch = QQChannel(QQConfig(allow_from=["*"]), MessageBus())
    data = _FakeQQMsg("m2", content="hello")

    with patch(
        "core.bot.channels.dedup.claim_processed", new=AsyncMock(return_value=False)
    ):
        await ch._on_message(data, is_group=True)

    # bus 应为空（消息被去重拦截，未 publish）
    assert await _qq_consume(ch.bus) is None


# ── Feishu ──────────────────────────────────────────────────────────────────


class _FakeFeishuSender:
    def __init__(self):
        self.sender_type = "user"
        self.sender_id = type("SID", (), {"open_id": "ou_u1"})()


class _FakeFeishuMessage:
    def __init__(self, msg_id: str, text: str = "hi", chat_type: str = "p2p"):
        self.message_id = msg_id
        self.chat_id = "oc_c1"
        self.chat_type = chat_type
        self.message_type = "text"
        # 飞书 text 消息的 content 是 JSON 字符串：{"text": "..."}
        import json as _json
        self.content = _json.dumps({"text": text}, ensure_ascii=False)
        self.mentions = []
        self.sender = _FakeFeishuSender()


class _FakeFeishuEvent:
    def __init__(self, msg):
        self.message = msg
        self.sender = msg.sender


class _FakeFeishuData:
    def __init__(self, msg):
        self.event = _FakeFeishuEvent(msg)


async def test_feishu_uses_redis_dedup_first_call_processed():
    """飞书第一次收到 msg → claim_processed 返回 True → 消息进入 bus。"""
    ch = FeishuChannel(FeishuConfig(allow_from=["*"]), MessageBus())
    data = _FakeFeishuData(_FakeFeishuMessage("fm1", text="你好"))

    with patch(
        "core.bot.channels.dedup.claim_processed", new=AsyncMock(return_value=True)
    ) as spy:
        await ch._on_message(data)

    spy.assert_awaited_once_with("feishu", "fm1")
    msg = await _qq_consume(ch.bus)
    assert msg is not None and msg.content == "你好"


async def test_feishu_redis_dedup_skips_already_processed():
    """飞书同 msg 第二次 → claim_processed 返回 False → 早退，不进 bus。"""
    ch = FeishuChannel(FeishuConfig(allow_from=["*"]), MessageBus())
    data = _FakeFeishuData(_FakeFeishuMessage("fm2", text="你好"))

    with patch(
        "core.bot.channels.dedup.claim_processed", new=AsyncMock(return_value=False)
    ):
        await ch._on_message(data)

    assert await _qq_consume(ch.bus) is None


async def test_dedup_claim_atomic_set_nx():
    """claim_processed 用 SET NX：同一 key 两次调用只成功一次（无 TOCTOU）。"""
    from core.bot.channels import dedup

    calls = {"set_args": []}

    class _FakeRedis:
        async def set(self, key, value, *, nx=None, ex=None):
            calls["set_args"].append((key, value, nx, ex))
            # 第一次 SET NX 成功，第二次失败（模拟 key 已存在）
            return True if len(calls["set_args"]) == 1 else None

    with patch("core.db.cache._get_pool", return_value=_FakeRedis()):
        first = await dedup.claim_processed("qq", "dup1")
        second = await dedup.claim_processed("qq", "dup1")

    assert first is True  # 首次抢到
    assert second is False  # 已被占用
    # 两次都用 NX（原子抢锁语义）
    assert all(args[2] is True for args in calls["set_args"])
