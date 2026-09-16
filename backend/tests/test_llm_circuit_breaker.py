"""CircuitBreaker 滑动窗口重写的行为测试（2026-09）。

覆盖四组语义（对标 litellm cooldown / resilience4j sliding window / pybreaker）：
1. classify_failure 错误分类：只有供应商侧故障（超时/连接/429/5xx）计入熔断；
   调用方错误（400/401/404）不误熔断健康供应商。
2. 滑动窗口失败率判定：看比率不看连续——抖动型故障（好一次坏一次）能触发；
   成功不清零窗口；样本不足（< minimum_calls）不判定。
3. per-attempt 记账：with_retry_and_circuit 每次 attempt 独立过熔断+记账，
   OPEN 时 fail-fast（func 不被调用）。
4. HALF_OPEN：探测名额/合闸成功数结构上无死锁；探测失败回 OPEN；
   名额耗尽拒绝；ignored 错误归还探测名额。

（旧 H-11 死锁测试与 M-14 worker 缩放测试已删——死锁组合结构性不可能、缩放机制移除。）
"""
from __future__ import annotations

import httpx
import openai
import pytest
from prometheus_client import REGISTRY

from core.llm.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    LLMRetryError,
    RetryConfig,
    classify_failure,
    with_retry_and_circuit,
)


# ---------------------------------------------------------------------------
# 构造工具
# ---------------------------------------------------------------------------

_REQ = httpx.Request("POST", "http://test.local/v1/chat/completions")


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQ)


def _status_exc(cls, status: int) -> openai.APIStatusError:
    return cls("test error", response=_resp(status), body=None)


def _fast_cb(
    name: str,
    *,
    window: int = 4,
    minimum_calls: int = 2,
    rate: float = 0.5,
    open_timeout: float = 0.0,
    probes: int = 3,
    success_needed: int = 2,
) -> CircuitBreaker:
    """小阈值快速触发的熔断器（open_timeout=0 → OPEN 后下一次调用即转 HALF_OPEN）。"""
    return CircuitBreaker(
        name,
        config=CircuitBreakerConfig(
            window_size=window,
            minimum_calls=minimum_calls,
            failure_rate_threshold=rate,
            open_timeout=open_timeout,
            half_open_probes=probes,
            half_open_success_needed=success_needed,
        ),
    )


async def _open_breaker(cb: CircuitBreaker, n: int = 2) -> None:
    """灌 n 次计入失败的调用把熔断器打 OPEN。"""
    for _ in range(n):
        await cb.before_call()
        await cb.record_failure(RuntimeError("503 Service Unavailable"))
    assert cb.state is CircuitState.OPEN


# ---------------------------------------------------------------------------
# 1. classify_failure 错误分类表
# ---------------------------------------------------------------------------


def test_classify_provider_side_failures_counted():
    """供应商侧故障全部计入：连接/超时/限流/5xx。"""
    assert classify_failure(openai.APIConnectionError(message="conn", request=_REQ))
    assert classify_failure(openai.APITimeoutError(request=_REQ))
    assert classify_failure(_status_exc(openai.RateLimitError, 429))
    assert classify_failure(_status_exc(openai.InternalServerError, 500))
    assert classify_failure(_status_exc(openai.InternalServerError, 503))
    # openai 树之外：httpx 传输层 + 内建超时（3.11+ asyncio.TimeoutError 即它）
    assert classify_failure(httpx.ConnectError("connection refused"))
    assert classify_failure(httpx.ConnectTimeout("timed out"))
    assert classify_failure(TimeoutError("async timeout"))


def test_classify_client_side_errors_ignored():
    """调用方错误不计入：400/401/404 换个请求照样成功，熔断只会误伤健康供应商。"""
    assert not classify_failure(_status_exc(openai.BadRequestError, 400))
    assert not classify_failure(_status_exc(openai.AuthenticationError, 401))
    assert not classify_failure(_status_exc(openai.NotFoundError, 404))
    # httpx.HTTPStatusError 按状态码同口径：400 不计 / 502 计
    assert not classify_failure(
        httpx.HTTPStatusError("bad request", request=_REQ, response=_resp(400))
    )
    assert classify_failure(
        httpx.HTTPStatusError("bad gateway", request=_REQ, response=_resp(502))
    )


def test_classify_unknown_exception_keyword_fallback():
    """未知异常兜底：文案命中可重试关键词 → 计入；其余不计（宁漏勿误）。"""
    assert classify_failure(RuntimeError("429 Too Many Requests"))
    assert classify_failure(RuntimeError("Connection reset by peer"))
    assert not classify_failure(RuntimeError("401 unauthorized"))
    assert not classify_failure(RuntimeError("invalid image format"))


def test_classify_408_counts_as_provider_side():
    """408 请求超时按供应商侧计入（litellm 同口径）。"""
    assert classify_failure(
        httpx.HTTPStatusError("timeout", request=_REQ, response=_resp(408))
    )


# ---------------------------------------------------------------------------
# 2. 滑动窗口判定语义
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alternating_failures_trip_open():
    """抖动型故障（好一次坏一次）也能触发——旧实现成功清零计数永不熔断。"""
    cb = _fast_cb("alt", window=6, minimum_calls=4)
    for i in range(3):
        await cb.before_call()
        await cb.record_success()
        await cb.before_call()
        await cb.record_failure(RuntimeError("503 Service Unavailable"))
    # 6 样本 3 失败 = 50% ≥ 阈值 → OPEN
    assert cb.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_success_does_not_clear_window():
    """成功不清零窗口：F,F,F,S,F 在 5 样本下失败率 80% → OPEN。"""
    cb = _fast_cb("no_clear", window=6, minimum_calls=5)
    for _ in range(3):
        await cb.before_call()
        await cb.record_failure(RuntimeError("503 Service Unavailable"))
    await cb.before_call()
    await cb.record_success()  # 若清零（旧语义），后续 1 次失败不足以触发
    assert cb.state is CircuitState.CLOSED  # 4 样本 < minimum_calls=5
    await cb.before_call()
    await cb.record_failure(RuntimeError("503 Service Unavailable"))
    assert cb.state is CircuitState.OPEN


@pytest.mark.asyncio
async def test_below_minimum_calls_never_trips():
    """样本不足不判定：min=10 时 9 连失败（100% 失败率）仍 CLOSED。"""
    cb = _fast_cb("min_calls", window=10, minimum_calls=10)
    for _ in range(9):
        await cb.before_call()
        await cb.record_failure(RuntimeError("503 Service Unavailable"))
    assert cb.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_healthy_traffic_stays_closed():
    """正常流量（偶发失败远低于阈值）不触发。"""
    cb = _fast_cb("healthy", window=10, minimum_calls=8)
    for i in range(20):
        await cb.before_call()
        if i % 10 == 0:
            await cb.record_failure(RuntimeError("503 Service Unavailable"))
        else:
            await cb.record_success()
    assert cb.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_client_error_does_not_count():
    """400 类调用方错误连打多次也熔不了闸（不进窗口）。"""
    cb = _fast_cb("client_err", window=4, minimum_calls=2)
    for _ in range(6):
        await cb.before_call()
        await cb.record_failure(RuntimeError("401 unauthorized"))
    assert cb.state is CircuitState.CLOSED
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# 3. with_retry_and_circuit：per-attempt 记账 + fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_attempt_counting_trips_fast():
    """每次 attempt 独立记账：一次调用 2 波 attempt 全失败即凑满 2 样本 → OPEN。

    旧实现整轮重试耗尽才算一次失败，同样场景只记 1 次（阈值 5 时要 5 整轮）。
    """
    cb = _fast_cb("per_attempt", window=4, minimum_calls=2, open_timeout=60.0)
    calls = 0

    async def _always_fail():
        nonlocal calls
        calls += 1
        raise RuntimeError("503 Service Unavailable")

    with pytest.raises(LLMRetryError):
        await with_retry_and_circuit(
            _always_fail,
            retry_config=RetryConfig(max_retries=1, base_delay=0.001, max_delay=0.001),
            circuit_breaker=cb,
        )
    assert calls == 2  # 首发 + 1 次重试
    assert cb.state is CircuitState.OPEN  # 2 次计入失败 ≥ minimum_calls=2


@pytest.mark.asyncio
async def test_open_breaker_fails_fast_without_calling():
    """OPEN 时毫秒级抛 CircuitOpenError，func 不被调用（不烧下游/不占超时）。"""
    cb = _fast_cb("fail_fast", open_timeout=60.0)
    await _open_breaker(cb)

    calls = 0

    async def _should_not_run():
        nonlocal calls
        calls += 1
        return "ok"

    with pytest.raises(CircuitOpenError):
        await with_retry_and_circuit(
            _should_not_run,
            retry_config=RetryConfig(max_retries=3, base_delay=0.001),
            circuit_breaker=cb,
        )
    assert calls == 0


@pytest.mark.asyncio
async def test_none_breaker_still_retries():
    """circuit_breaker=None（H-16 自配路径）：只重试不熔断，行为不变。"""
    calls = 0

    async def _flaky():
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    result = await with_retry_and_circuit(
        _flaky,
        retry_config=RetryConfig(max_retries=2, base_delay=0.001, max_delay=0.001),
        circuit_breaker=None,
    )
    assert result == "ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_non_retryable_error_raises_after_one_attempt():
    """不可重试错误（401）一波即弃，且不计入熔断窗口。"""
    cb = _fast_cb("non_retry", window=4, minimum_calls=2)
    calls = 0

    async def _always_401():
        nonlocal calls
        calls += 1
        raise RuntimeError("401 unauthorized")

    with pytest.raises(LLMRetryError):
        await with_retry_and_circuit(
            _always_401,
            retry_config=RetryConfig(max_retries=3, base_delay=0.001),
            circuit_breaker=cb,
        )
    assert calls == 1
    assert cb.state is CircuitState.CLOSED
    assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# 4. HALF_OPEN：探测名额 / 合闸 / 回 OPEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_half_open_two_successes_close():
    """needed=2：两次探测成功才合闸；名额释放语义保证第二次探测能放行（旧 H-11 场景）。"""
    cb = _fast_cb("ho_close", open_timeout=0.0)
    await _open_breaker(cb)

    await cb.before_call()  # OPEN → HALF_OPEN，占名额 1
    await cb.record_success()
    assert cb.state is CircuitState.HALF_OPEN  # 1 成功 < 2

    await cb.before_call()  # 名额已释放，可再探测
    await cb.record_success()
    assert cb.state is CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_probe_failure_reopens():
    """探测失败 → 立即回 OPEN；再等 open_timeout 后可重新探测。"""
    cb = _fast_cb("ho_reopen", open_timeout=0.0)
    await _open_breaker(cb)

    await cb.before_call()
    await cb.record_success()  # HALF_OPEN
    await cb.before_call()
    await cb.record_failure(RuntimeError("503 Service Unavailable"))  # 探测失败
    assert cb.state is CircuitState.OPEN

    await cb.before_call()  # open_timeout=0 → 再次放行探测
    await cb.record_success()
    assert cb.state is CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_probe_quota_exhaustion():
    """探测名额耗尽：第 probes+1 个并发调用被 CircuitOpenError 拒绝。"""
    cb = _fast_cb("ho_quota", probes=2, open_timeout=60.0)
    await _open_breaker(cb)

    cb._opened_at = 0.0  # 手动老化，让 before_call 转 HALF_OPEN
    await cb.before_call()  # 名额 1/2
    await cb.before_call()  # 名额 2/2
    with pytest.raises(CircuitOpenError):
        await cb.before_call()  # 超额 → 拒绝


@pytest.mark.asyncio
async def test_half_open_client_error_releases_quota():
    """探测遇到 ignored 错误（400 类）不回 OPEN，但要归还名额——一次 400 不能烧掉探测额度。"""
    cb = _fast_cb("ho_client_err", probes=2, open_timeout=0.0)
    await _open_breaker(cb)

    await cb.before_call()
    await cb.record_failure(RuntimeError("400 bad request"))
    assert cb.state is CircuitState.HALF_OPEN  # 未回 OPEN

    await cb.before_call()  # 名额已归还，可继续探测
    await cb.record_success()
    await cb.before_call()
    await cb.record_success()
    assert cb.state is CircuitState.CLOSED


def test_config_guard_raises_probes_to_success_needed():
    """success_needed > probes 时构造器抬升 probes——死锁组合结构性不可能。"""
    cb = CircuitBreaker(
        "guard",
        config=CircuitBreakerConfig(
            window_size=4, minimum_calls=2,
            half_open_probes=1, half_open_success_needed=3,
        ),
    )
    assert cb.config.half_open_probes >= cb.config.half_open_success_needed


@pytest.mark.asyncio
async def test_close_clears_window_for_fresh_start():
    """合闸后窗口清空：历史失败不拖累恢复后的判定（合闸即全新窗口）。"""
    cb = _fast_cb("fresh", open_timeout=0.0)
    await _open_breaker(cb)
    await cb.before_call()
    await cb.record_success()
    await cb.before_call()
    await cb.record_success()
    assert cb.state is CircuitState.CLOSED
    assert cb.failure_count == 0  # 窗口已清

    # 恢复后一次失败不会立刻再 OPEN（新窗口样本 < minimum_calls）
    await cb.before_call()
    await cb.record_failure(RuntimeError("503 Service Unavailable"))
    assert cb.state is CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 5. 注册表与指标
# ---------------------------------------------------------------------------


def test_registry_per_name_isolation():
    """按名注册互相隔离；同名返回同一实例（llm.py 与 loop.py 依赖此性质）。"""
    from core.llm.reliability import (
        _llm_circuit_breakers,
        get_llm_circuit_breaker,
    )

    a1 = get_llm_circuit_breaker("test_isolate_a")
    a2 = get_llm_circuit_breaker("test_isolate_a")
    b = get_llm_circuit_breaker("test_isolate_b")
    assert a1 is a2
    assert a1 is not b

    # 阈值吃 settings（单进程语义，无 worker 缩放）
    from settings.base import get_settings

    _llm = get_settings().llm
    assert a1.config.window_size == _llm.circuit_window_size
    assert a1.config.minimum_calls == _llm.circuit_minimum_calls
    assert a1.config.open_timeout == _llm.circuit_open_timeout

    # 清理：测试自造的注册名不泄漏进后续 /health/detailed 快照断言
    _llm_circuit_breakers.pop("test_isolate_a", None)
    _llm_circuit_breakers.pop("test_isolate_b", None)


@pytest.mark.asyncio
async def test_metrics_emitted_on_transitions():
    """状态转换打点：transitions counter 递增、state gauge 跟随。"""
    name = "test_metrics_cb"
    _t_open = REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "open"}
    )
    _t_half = REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "half_open"}
    )
    _t_closed = REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "closed"}
    )
    before = (_t_open or 0, _t_half or 0, _t_closed or 0)

    cb = _fast_cb(name, open_timeout=0.0)
    await _open_breaker(cb)
    assert REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "open"}
    ) == before[0] + 1
    # gauge：OPEN=2
    assert REGISTRY.get_sample_value(
        "ca_llm_circuit_state", {"breaker": name}
    ) == 2

    await cb.before_call()  # → HALF_OPEN
    await cb.record_success()
    await cb.before_call()
    await cb.record_success()  # → CLOSED
    assert REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "half_open"}
    ) == before[1] + 1
    assert REGISTRY.get_sample_value(
        "ca_llm_circuit_transitions_total", {"breaker": name, "to": "closed"}
    ) == before[2] + 1
    assert REGISTRY.get_sample_value(
        "ca_llm_circuit_state", {"breaker": name}
    ) == 0
