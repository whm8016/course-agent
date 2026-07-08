"""LLM 可靠性隔离回归测试（H-16 + M-14/16/19/20/21）。

H-16：自配/profile client 失败不接入全局 default 熔断器——用户选了个坏 profile 不应
连累所有走全局 client 的用户。判定信号 = _one_round 的 llm_client 参数（None=全局
默认 client 才熔断）。

M-14：worker scaling 应用到「显式传入的 configured」熔断器（此前仅 None 默认路径缩放）。
M-16：BACKEND_WORKERS env 与 settings.backend_workers 统一同一来源。
M-19：profile base_url 空值回退 .env（DASHSCOPE_BASE_URL），与 api_key 对称。
M-20：catalog 读-改-写原子（_atomic_update），杜绝 TOCTOU 丢更新 + 原子文件替换。
M-21：vision 模型描述图片的调用接入 reliability 层（retry + 熔断），失败退化为空描述、
不阻断主链路；此前是裸 await vision_client...create，瞬时错误（429/超时）直接丢图。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from core.agentic.loop import run_agent_loop
from core.agentic.types import LoopOutcome
from core.context import UnifiedContext
from core.llm.llm import _llm_circuit_breaker
from core.llm.reliability import CircuitOpenError, RetryConfig
from core.stream_bus import StreamBus


def _make_chunk(content: str = ""):
    delta = MagicMock()
    delta.content = content or None
    delta.tool_calls = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


async def _async_iter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# H-16：自配 client 隔离全局熔断器
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h16_user_supplied_client_uses_no_circuit_breaker():
    """run_agent_loop(client=<自配>) → _create_with_image_fallback 收到 circuit_breaker=None。

    自配/profile client 是用户私有资源：失败仍重试，但不计全局熔断器，避免坏 profile
    把全局 default 熔断器打 OPEN 误伤他人。
    """
    ctx = UnifiedContext(user_message="hi")
    bus = StreamBus()
    _llm_circuit_breaker.reset()
    captured: dict = {}

    async def _fake_create(llm_client, create_kwargs, binding, model, circuit_breaker=None):
        captured["circuit_breaker"] = circuit_breaker
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter([_make_chunk("ok")])
        return mock

    mock_client = MagicMock()
    try:
        with patch("core.agentic.loop._create_with_image_fallback", side_effect=_fake_create):
            outcome = await run_agent_loop(
                context=ctx, stream=bus, system_prompt="SYS",
                tool_schemas=None, client=mock_client,  # 自配：非 None
            )
        assert isinstance(outcome, LoopOutcome)
        # 核心断言：自配 client → circuit_breaker=None（不熔断）
        assert captured["circuit_breaker"] is None
    finally:
        _llm_circuit_breaker.reset()
        await bus.close()


@pytest.mark.asyncio
async def test_h16_default_client_uses_global_circuit_breaker():
    """run_agent_loop(client=None) → _create_with_image_fallback 收到全局 default 熔断器。

    未自配用户走全局 client，共享下游，失败要计入全局熔断器防雪崩。
    """
    ctx = UnifiedContext(user_message="hi")
    bus = StreamBus()
    _llm_circuit_breaker.reset()
    captured: dict = {}

    async def _fake_create(llm_client, create_kwargs, binding, model, circuit_breaker=None):
        captured["circuit_breaker"] = circuit_breaker
        mock = MagicMock()
        mock.__aiter__ = lambda self: _async_iter([_make_chunk("ok")])
        return mock

    try:
        with patch("core.agentic.loop._create_with_image_fallback", side_effect=_fake_create):
            await run_agent_loop(
                context=ctx, stream=bus, system_prompt="SYS", tool_schemas=None,
                # 不传 client → 全局默认
            )
        # 核心断言：默认路径 → 全局熔断器实例
        assert captured["circuit_breaker"] is _llm_circuit_breaker
    finally:
        _llm_circuit_breaker.reset()
        await bus.close()


@pytest.mark.asyncio
async def test_h16_user_supplied_client_failure_does_not_open_global_circuit():
    """自配 client 真实失败（重试耗尽）→ 全局 default 熔断器零污染（仍 CLOSED）。

    loop 首轮失败会向上抛 LLMRetryError（由 capability 处理错误响应），这是设计预期；
    本测试只关心：自配失败不能把全局 default 熔断器打 OPEN（核心隔离保证）。
    """
    from core.llm.reliability import LLMRetryError

    ctx = UnifiedContext(user_message="hi")
    bus = StreamBus()
    _llm_circuit_breaker.reset()

    # 直接走真实 _create_with_image_fallback（不 patch），用自配 client 注入失败 + 快重试
    async def _always_fail(**kwargs):
        raise RuntimeError("401 unauthorized")  # 不可重试，立即抛 LLMRetryError

    mock_client = MagicMock()
    mock_client.chat.completions.create = _always_fail
    fast_retry = RetryConfig(max_retries=0, base_delay=0.001, max_delay=0.001)
    try:
        with patch("core.llm.llm._retry_config", fast_retry):
            # 首轮失败 loop 向上抛（capability 接管错误响应）——这是预期行为
            with pytest.raises(LLMRetryError):
                await run_agent_loop(
                    context=ctx, stream=bus, system_prompt="SYS",
                    tool_schemas=None, client=mock_client,  # 自配
                )
        # 核心断言：全局 default 熔断器零污染——自配失败不误伤他人
        assert _llm_circuit_breaker.get_state().value == "closed"
        assert _llm_circuit_breaker.failure_count == 0
    finally:
        _llm_circuit_breaker.reset()
        await bus.close()


# ---------------------------------------------------------------------------
# M-21：vision 描述接入 reliability 层（retry + 熔断保护）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m21_vision_describe_retries_on_transient_error():
    """vision 描述瞬时错误（429）→ 经 reliability 重试后成功，不丢图。"""
    from core.llm.vision_describe import describe_image_attachments

    att = MagicMock()
    att.base64 = "AAAA"
    att.file_path = None
    att.mime_type = "image/png"
    att.is_image.return_value = True

    call_count = 0

    async def _create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("429 Too Many Requests")
        return MagicMock(choices=[MagicMock(message=MagicMock(content="这是一张猫的图片"))])

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    fast_retry = RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.005)
    descs = await describe_image_attachments(
        mock_client, "qwen-vl-plus", [att], retry_config=fast_retry
    )
    assert call_count == 3  # 2 次 429 重试 + 第 3 次成功
    assert descs == ["这是一张猫的图片"]


@pytest.mark.asyncio
async def test_m21_vision_describe_circuit_open_degrades_to_empty():
    """vision 下游熔断 OPEN → describe 退化为空描述（不抛 CircuitOpenError 打断主链路）。"""
    from core.llm.vision_describe import describe_image_attachments

    att = MagicMock()
    att.base64 = "AAAA"
    att.file_path = None
    att.mime_type = "image/png"
    att.is_image.return_value = True

    async def _create(**kwargs):
        raise CircuitOpenError("vision circuit OPEN")

    mock_client = MagicMock()
    mock_client.chat.completions.create = _create

    # 熔断 OPEN 时直接降级为空串，不向上抛
    descs = await describe_image_attachments(
        mock_client, "qwen-vl-plus", [att],
        retry_config=RetryConfig(max_retries=0),
    )
    assert descs == [""]


# ---------------------------------------------------------------------------
# M-14：worker scaling 应用到 configured circuit breaker
# ---------------------------------------------------------------------------


def test_m14_configured_circuit_breaker_scales_failure_threshold(monkeypatch):
    """显式传入 config（get_llm_circuit_breaker 路径）的 failure_threshold 也按 worker 缩放。

    此前只有 config=None 走 worker 缩放；get_llm_circuit_breaker 显式传 CircuitBreakerConfig，
    failure_threshold 保持原值，多 worker 下阈值偏大、熔断迟钝。修复后统一缩放。
    """
    from core.llm.reliability import CircuitBreaker, CircuitBreakerConfig

    monkeypatch.setenv("BACKEND_WORKERS", "4")
    cb = CircuitBreaker(
        "m14_cfg",
        config=CircuitBreakerConfig(
            failure_threshold=8, success_threshold=2, open_timeout=30.0,
        ),
    )
    # failure_threshold(8) // workers(4) = 2（>= 下限 2）。success_threshold 不缩放。
    assert cb.config.failure_threshold == 2
    assert cb.config.success_threshold == 2


def test_m14_failure_threshold_floor_is_two(monkeypatch):
    """workers 过大导致 //workers < 2 时，下限抬到 2，避免单次抖动即熔断。"""
    from core.llm.reliability import CircuitBreaker, CircuitBreakerConfig

    monkeypatch.setenv("BACKEND_WORKERS", "16")
    cb = CircuitBreaker(
        "m14_floor",
        config=CircuitBreakerConfig(
            failure_threshold=5, success_threshold=2, open_timeout=30.0,
        ),
    )
    # 5 // 16 = 0 → 抬到下限 2
    assert cb.config.failure_threshold == 2


# ---------------------------------------------------------------------------
# M-16：BACKEND_WORKERS env 与 settings.backend_workers 统一
# ---------------------------------------------------------------------------


def test_m16_resolve_backend_workers_env_priority(monkeypatch):
    """env BACKEND_WORKERS 优先于 settings.backend_workers。"""
    from core.llm.reliability import _resolve_backend_workers

    monkeypatch.setenv("BACKEND_WORKERS", "8")
    assert _resolve_backend_workers() == 8


def test_m16_resolve_backend_workers_falls_back_to_settings(monkeypatch):
    """env 缺失时回退 settings.backend_workers（不再硬编码只读 env）。"""
    from core.llm.reliability import _resolve_backend_workers
    from settings.base import get_settings

    monkeypatch.delenv("BACKEND_WORKERS", raising=False)
    expected = get_settings().backend_workers
    assert _resolve_backend_workers() == max(1, expected)


# ---------------------------------------------------------------------------
# M-19：profile base_url 空值回退 .env
# ---------------------------------------------------------------------------


def test_m19_profile_empty_base_url_falls_back_to_env(monkeypatch):
    """profile base_url 空 → 回退 settings.llm.base_url（.env），与 api_key 对称。"""
    from core.llm import provider_factory as pf

    # 模拟 .env 配了自定义 base_url
    monkeypatch.setattr(pf, "DASHSCOPE_BASE_URL", "https://env.example.com/v1")
    monkeypatch.setattr(pf, "DASHSCOPE_API_KEY", "sk-env")
    monkeypatch.setattr(pf, "LLM_BINDING", "openai")

    # profile base_url 空 → 应回退到 .env 端点
    captured: dict = {}

    def _fake_get_llm_client(*, binding, api_key, base_url, api_version, model=None, timeout=120):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return MagicMock()

    monkeypatch.setattr(pf, "get_llm_client", _fake_get_llm_client)
    pf.clear_llm_client_cache()

    pf.get_llm_client_for_profile({"binding": "openai", "api_key": "", "base_url": ""})
    assert captured["base_url"] == "https://env.example.com/v1"


# ---------------------------------------------------------------------------
# M-20：catalog 读-改-写原子（TOCTOU）
# ---------------------------------------------------------------------------


def test_m20_upsert_delete_set_active_round_trip(tmp_path, monkeypatch):
    """upsert / set_active / delete 端到端正确，且经 _atomic_update 写回。"""
    from core.llm import catalog as cat

    cat_path = tmp_path / "catalog.json"
    monkeypatch.setattr(cat, "CATALOG_PATH", str(cat_path))

    # upsert 两个 profile
    cat.upsert_profile("p1", {"name": "P1", "binding": "openai", "text_model": "gpt"})
    cat.upsert_profile("p2", {"name": "P2", "binding": "deepseek", "text_model": "ds"})
    data = cat.load_catalog()
    assert len(data["profiles"]) == 2

    # set_active
    assert cat.set_active("p2") is True
    assert cat.active_profile_id() == "p2"
    assert cat.set_active("nope") is False  # 不存在

    # delete
    assert cat.delete_profile("p1") is True
    assert cat.delete_profile("p1") is False  # 再删返回 False
    data = cat.load_catalog()
    assert len(data["profiles"]) == 1
    # 删 active 后自动切到剩余 profile
    assert data["active_profile"] == "p2"


def test_m20_save_catalog_atomic_replace(tmp_path, monkeypatch):
    """save_catalog 用临时文件 + os.replace：写入中途崩溃不留半截 JSON。"""
    from core.llm import catalog as cat

    cat_path = tmp_path / "catalog.json"
    monkeypatch.setattr(cat, "CATALOG_PATH", str(cat_path))

    cat.save_catalog({"active_profile": "default", "profiles": [{"id": "x"}]})
    # 文件是完整合法 JSON（os.replace 原子替换，不会被读到半截）
    with open(cat_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["profiles"][0]["id"] == "x"

