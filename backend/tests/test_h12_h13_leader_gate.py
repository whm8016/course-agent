"""H-12 / H-13：bot 启停的多 worker leader 门控。

根因：``TutorBotManager.start_bot`` 无 ``is_leader()`` 门控，follower worker 收到
bot 启停请求（多 worker + LB 常见）也会拉起一份 bot 实例 → 同一 bot 在两个 worker
各持一份（split-brain）：同一 IM 消息被双方各处理一遍、心跳双跑、cron 双触发。

修法：``start_bot`` 入口校验 ``is_leader()``，非 leader 抛 ``NotLeaderError``；
``api/bot.py`` 的 create/start/send_message 端点先 ``_require_leader_or_409()`` →
非 leader 返回 409（让前端/LB 重试到 leader）。

failover 时序推演：
  1. leader A 持有 bot B；A 的 leader lease TTL=30s。
  2. A 卡死 → 续约 loop 一起卡死 → TTL 过期（A 仍以为自己 leader，但其 _is_leader
     在下次 CAS 返回 0 时才翻 False；A 卡死期间不会再发请求，无害）。
  3. follower C 竞选 loop（每 10s）抢到锁 → ``_become_leader`` → ``_is_leader=True``
     → ``on_gain`` 拉 auto_start_bots → bot B 在 C 上启动。
  4. 此时 A 若苏醒：其 ``start_bot`` 入口读 ``is_leader()``——A 的 _is_leader 仍为
     True（CAS 还没跑），但 A 没有新的启动请求路径（LB 已把流量打到 C）；即便有，
     A 的内存里已有 bot B，``start_bot`` 命中 ``if running: return`` 早退，不会双跑。
     关键：A 一旦 CAS 返回 0 → ``_lose_leader`` → ``on_lose`` 停掉本地 bot B 实例，
     释放 WS 连接、取消 task。最终唯一 leader C 持有 bot B。
  5. 任意时刻 ``is_leader()`` 全局唯一真（CAS 原子 + worker_id 唯一保证无脑裂）。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import core.leader as leader  # noqa: E402
from core.bot.manager import NotLeaderError, TutorBotManager  # noqa: E402


@pytest.fixture(autouse=True)
async def _reset_leader_state():
    """每个测试前后重置 leader 全局标志，避免跨测试污染。"""
    leader._is_leader = False
    yield
    leader._is_leader = False


async def test_follower_start_bot_rejected():
    """非 leader 调 start_bot → 抛 NotLeaderError（follower 不能持有 bot 实例）。"""
    leader._is_leader = False
    mgr = TutorBotManager()
    with pytest.raises(NotLeaderError):
        await mgr.start_bot("owner1", "bot1")


async def test_auto_start_bots_skipped_when_not_leader():
    """非 leader 调 auto_start_bots → 直接跳过（不扫描磁盘、不启动任何 bot）。"""
    leader._is_leader = False
    mgr = TutorBotManager()
    # 即便磁盘上有 auto_start: true 的 bot，follower 也不该启动
    await mgr.auto_start_bots()
    # 没有 bot 被加入内存（_bots 为空证明未启动）
    assert mgr._bots == {}


async def test_leader_start_bot_passes_gate(monkeypatch, tmp_path):
    """leader 调 start_bot → 门控放行，进入实际启动（mock 掉重依赖，验证未被门控拦下）。"""
    leader._is_leader = True
    mgr = TutorBotManager()

    # 把工作区根指到临时目录，避免污染真实 data/tutorbot
    monkeypatch.setattr("core.bot.manager.get_bot_workspace_root", lambda: tmp_path)

    # mock 掉 start_bot 内部读 settings 的属性（llm 凭证 / tutorbot 心跳）

    class _FakeSettings:
        class _llm:
            api_key = type("S", (), {"get_secret_value": lambda self: "sk-x"})()
            base_url = "http://x"
            text_model = "m"

        class _tutorbot:
            heartbeat_enabled = False
            heartbeat_interval_sec = 60

        llm = _llm()
        tutorbot = _tutorbot()

    # start_bot 内部是 ``from settings import get_settings`` —— patch settings 模块
    import settings as settings_mod

    monkeypatch.setattr(settings_mod, "get_settings", lambda: _FakeSettings())

    # mock AgentLoop / provider / SessionManager（start_bot 内部局部 import）
    import sys as _sys

    class _FakeLoop:
        def __init__(self, *a, **k):
            pass

        async def run(self):
            return

        async def stop(self):
            return

        async def process_direct(self, *a, **k):
            return ""

    # 在 core.bot.agent 上挂假的 loop 模块，供 ``from core.bot.agent.loop import AgentLoop`` 命中
    fake_agent_loop_mod = type(_sys)("core.bot.agent.loop")
    fake_agent_loop_mod.AgentLoop = _FakeLoop
    monkeypatch.setitem(_sys.modules, "core.bot.agent.loop", fake_agent_loop_mod)

    class _FakeProvider:
        def __init__(self, *a, **k):
            pass

    fake_provider_mod = type(_sys)("core.bot.providers.openai_compat")
    fake_provider_mod.OpenAICompatProvider = _FakeProvider
    monkeypatch.setitem(_sys.modules, "core.bot.providers.openai_compat", fake_provider_mod)

    # 关键断言：leader 状态下不被门控拦下（不抛 NotLeaderError），能跑到返回 instance。
    # ChannelManager 默认空（config.channels={}），heartbeat 关闭，所以不会触达真实 WS。
    from core.bot.manager import BotConfig

    inst = await mgr.start_bot("owner1", "bot1", BotConfig(name="bot1", owner_id="owner1"))
    assert inst.bot_id == "bot1"
    # 证明门控确实放行（实例已注册到内存）
    assert mgr.get_bot("owner1", "bot1") is inst
    # 清理后台 task
    await mgr.stop_bot("owner1", "bot1")
