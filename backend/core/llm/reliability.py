"""
LLM 调用可靠性增强模块

提供以下功能：
1. 指数退避重试 - 处理临时性网络故障和 API 限流
2. 熔断器（滑动窗口失败率判定）- 防止持续故障时继续调用造成资源浪费
3. 错误分类 - 只有供应商侧故障（超时/连接错误/429/5xx）计入熔断

设计对标（2026-09 滑动窗口重写）：
- resilience4j CircuitBreaker：COUNT_BASED 滑动窗口 + minimumNumberOfCalls 最小样本量
  + HALF_OPEN 探测名额；OPEN→HALF_OPEN 由下一个请求被动触发（无后台定时器）。
- litellm router cooldown（router_utils/cooldown_handlers.py）：失败率判定 + 错误分类
  （429/408/5xx 记账、其余 4xx 是调用方错误不记账）+ 每次失败调用都参与判定
  （per-attempt，而非整轮重试耗尽才算一次失败）。
- pybreaker：HALF_OPEN 少量成功即合闸。

与旧实现（连续失败计数 + 成功清零 + //BACKEND_WORKERS 阈值缩放）的关键差异：
- 看比率不看连续：抖动型故障（好一次坏一次）也能触发；成功不清零窗口。
- 重试在熔断外层：每次 attempt 先过熔断检查、失败即记账（per-attempt），熔断打开时
  毫秒级拒绝（fail-fast），上层可立即切 fallback。旧实现整轮重试耗尽才算一次失败，
  阈值 5 实际要 ~60 次 HTTP 尝试 / 35s 才 OPEN，熔断的"快断"意义被重试稀释。
- 错误分类：400/401 等调用方错误不再误熔断健康供应商。
- 多 worker 各进程独立记账，阈值按单进程语义配置（旧 //workers 缩放补丁删除）。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TypeVar

import httpx
import openai

from core.observability.metrics import CIRCUIT_STATE, CIRCUIT_TRANSITIONS, LLM_RETRIES

logger = logging.getLogger(__name__)

T = TypeVar("T")

# LLM 服务不可用的统一用户文案：真值在 observability/errors.py 的错误信封注册表
#（llm_unavailable 码），此处 re-export 供 chat_complete 等直接使用，勿双写字面量。
from core.observability.errors import LLM_UNAVAILABLE_MESSAGE

LLM_SERVICE_UNAVAILABLE = LLM_UNAVAILABLE_MESSAGE

# 可重试错误关键词（_is_retryable 与 classify_failure 的未知异常兜底共用；
# 覆盖 openai/httpx 及自配 client 的常见错误文案）
_RETRYABLE_KEYWORDS = (
    'timeout', 'rate limit', 'too many requests',
    'connection', 'temporarily unavailable',
    'service unavailable', 'bad gateway',
    'gateway timeout', 'reset by peer',
)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态，允许请求通过
    OPEN = "open"          # 熔断状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，允许有限探测请求


# Prometheus 状态 gauge 的取值映射（0=closed 1=half_open 2=open）
_STATE_GAUGE_VALUE = {
    CircuitState.CLOSED: 0,
    CircuitState.HALF_OPEN: 1,
    CircuitState.OPEN: 2,
}


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 3           # 最大重试次数
    base_delay: float = 1.0         # 基础延迟（秒）
    max_delay: float = 30.0         # 最大延迟（秒）
    exponential_base: float = 2.0    # 指数退避基数

    # 可重试的错误类型（HTTP 状态码）
    retryable_status_codes: set[int] = field(default_factory=lambda: {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    })


@dataclass
class CircuitBreakerConfig:
    """熔断器配置（滑动窗口失败率判定，resilience4j 语义）"""
    window_size: int = 20               # 滑动窗口容量（按调用次数）
    minimum_calls: int = 8              # 判定前最小样本量（样本不足不判定，防低流量误跳）
    failure_rate_threshold: float = 0.5  # 窗口失败率 ≥ 此值 → OPEN
    open_timeout: float = 30.0           # OPEN 持续秒数（超时后被动转 HALF_OPEN）
    half_open_probes: int = 3            # 半开探测名额（并发上限）
    half_open_success_needed: int = 2    # 合闸（HALF_OPEN→CLOSED）所需成功次数


def status_of(exc: Exception) -> int | None:
    """提取异常的 HTTP 状态码（openai SDK / httpx / 任何带 status_code 或 response 属性的异常）。

    公开给跨模块消费（loop._is_context_length_error 沿因果链查 413 同用此提取），
    是全仓异常状态码提取的单一真相源。
    """
    status = (
        getattr(exc, 'status_code', None)
        or getattr(getattr(exc, 'response', None), 'status_code', None)
    )
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def classify_failure(exc: Exception) -> bool:
    """判定异常是否计入熔断失败（True=供应商侧故障；False=调用方错误/未知业务错）。

    对标 litellm ``_is_cooldown_required`` + resilience4j recordExceptions/ignoreExceptions：
    只有"换个请求也会失败"的供应商侧故障才该熔断；400/401/404 这类调用方错误熔断
    只会误伤健康供应商（我们的请求错了 ≠ 服务不可用）。
    """
    # 1. 异常类型（最准）：连接失败/超时/限流——openai SDK 树 + httpx 树 + 内建
    #    TimeoutError（3.11+ 起 asyncio.TimeoutError 即它）。APITimeoutError 是
    #    APIConnectionError 的子类，一并覆盖。
    if isinstance(exc, (openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
        return True
    # 2. 状态码：openai.APIStatusError / httpx.HTTPStatusError 及任何带状态码的异常。
    #    openai.InternalServerError(≥500) 在此覆盖；4xx 里 408/429 视为供应商侧。
    status = status_of(exc)
    if status is not None:
        return status >= 500 or status in (408, 429)
    # 3. 未知异常兜底：文案命中可重试关键词（timeout/connection/…）视为供应商侧；
    #    其余不计入——宁漏勿误（误熔断伤害健康供应商，漏记由后续真实错误类型补上）
    error_str = str(exc).lower()
    return any(kw in error_str for kw in _RETRYABLE_KEYWORDS)


def _is_retryable(exc: Exception, config: RetryConfig) -> bool:
    """判定异常是否值得重试（与是否计入熔断独立：重试管退避，熔断管跳闸）。"""
    error_str = str(exc).lower()
    if any(kw in error_str for kw in _RETRYABLE_KEYWORDS):
        return True
    status = status_of(exc)
    return status is not None and status in config.retryable_status_codes


class CircuitBreaker:
    """
    熔断器（滑动窗口失败率判定）

    工作原理：
    1. CLOSED：每次调用按 classify_failure 分类记账进滑动窗口（只记供应商侧故障）
    2. 窗口样本 ≥ minimum_calls 且失败率 ≥ failure_rate_threshold → OPEN
    3. OPEN：before_call 直接抛 CircuitOpenError（不发起真实调用，fail-fast）
    4. open_timeout 后由下一个请求被动转 HALF_OPEN（无后台定时器）
    5. HALF_OPEN 最多放行 half_open_probes 个探测，累计 half_open_success_needed
       次成功 → CLOSED；任一计入失败的探测 → 立即回 OPEN

    用法（配合 with_retry_and_circuit，per-attempt 记账）：
        await breaker.before_call()          # 不可放行抛 CircuitOpenError
        try:
            result = await call()
            await breaker.record_success()
        except Exception as e:
            await breaker.record_failure(e)
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        # 结构性无死锁保证：合闸所需成功数 ≤ 探测名额（旧实现两值失配曾致 HALF_OPEN
        # 永久卡死、需 H-11 运行时守卫；钳制后"名额耗尽仍凑不齐成功"的组合不可能出现）
        if self.config.half_open_success_needed > self.config.half_open_probes:
            logger.warning(
                "CircuitBreaker [%s]: half_open_probes(%d) < half_open_success_needed(%d)，"
                "抬升 probes 以结构性避免 HALF_OPEN 死锁",
                name, self.config.half_open_probes, self.config.half_open_success_needed,
            )
            self.config = replace(
                self.config, half_open_probes=self.config.half_open_success_needed
            )

        self.state = CircuitState.CLOSED
        self._window: deque[bool] = deque(maxlen=self.config.window_size)
        self._opened_at: float | None = None
        self._half_open_inflight = 0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()
        # 构造即打点：让 /metrics 从一开始就列出全部熔断器（缺省 closed=0）
        self._emit_state()

    async def before_call(self) -> None:
        """调用前熔断检查（持锁）。不可放行时抛 CircuitOpenError，不发起真实调用。

        OPEN→HALF_OPEN 被动转换：无后台定时器，open_timeout 由下一个到达的请求检查
        （resilience4j automaticTransitionFromOpenToHalfOpenEnabled=false 默认行为）。
        """
        async with self._lock:
            if self.state is CircuitState.OPEN:
                if time.time() - (self._opened_at or 0.0) >= self.config.open_timeout:
                    self._transition(CircuitState.HALF_OPEN)
                else:
                    raise CircuitOpenError(f"CircuitBreaker [{self.name}] is OPEN")
            if self.state is CircuitState.HALF_OPEN:
                if self._half_open_inflight >= self.config.half_open_probes:
                    raise CircuitOpenError(
                        f"CircuitBreaker [{self.name}] is HALF_OPEN, probe quota exhausted"
                    )
                self._half_open_inflight += 1

    async def record_success(self) -> None:
        """记录一次成功（持锁）。"""
        async with self._lock:
            if self.state is CircuitState.HALF_OPEN:
                self._half_open_inflight = max(0, self._half_open_inflight - 1)
                self._half_open_successes += 1
                # 跨周期迟到的旧探测成功可能计入新周期（探测响应慢于一次 OPEN→HALF_OPEN
                # 轮转）：最多让合闸略偏乐观，且合闸后窗口全新，供应商仍差会快速重开——
                # 可接受，不值得引入代际号复杂度。
                if self._half_open_successes >= self.config.half_open_success_needed:
                    self._transition(CircuitState.CLOSED)
            elif self.state is CircuitState.CLOSED:
                self._window.append(False)
            # OPEN 态的迟到成功不记：上一轮探测的回包，计入会让陈旧信号污染当前周期

    async def record_failure(self, exc: Exception) -> None:
        """记录一次失败（持锁）。调用方错误（classify_failure=False）完全不记。"""
        counted = classify_failure(exc)
        async with self._lock:
            if self.state is CircuitState.HALF_OPEN:
                if counted:
                    # 探测失败：立即回 OPEN（_transition 内重置半开计数）
                    self._transition(CircuitState.OPEN)
                else:
                    # 调用方错误不计，但探测名额要归还（否则一次 400 烧掉名额）
                    self._half_open_inflight = max(0, self._half_open_inflight - 1)
                return
            if not counted or self.state is CircuitState.OPEN:
                # 调用方错误不计（resilience4j ignoreExceptions 语义）；已 OPEN 无需再计
                return
            self._window.append(True)
            if (
                len(self._window) >= self.config.minimum_calls
                and sum(self._window) / len(self._window) >= self.config.failure_rate_threshold
            ):
                self._transition(CircuitState.OPEN)

    def _transition(self, new_state: CircuitState) -> None:
        """状态转换（须持 _lock 调用）：统一重置进入态计数 + 日志 + Prometheus 打点。"""
        old = self.state
        self.state = new_state
        if new_state is CircuitState.OPEN:
            self._opened_at = time.time()
        elif new_state is CircuitState.HALF_OPEN:
            # 上一轮残留的 in-flight 探测（含迟到回包）在此清算，不带入新周期
            self._half_open_inflight = 0
            self._half_open_successes = 0
        else:  # CLOSED：全新窗口，历史失败不拖累恢复后的判定
            # （半开计数不在进 CLOSED 时清：唯一入口是 HALF_OPEN 的 record_success，
            #  而下次进 HALF_OPEN 时 _transition 会先清零——在这清是死存储）
            self._window.clear()
        logger.info("CircuitBreaker [%s]: %s -> %s", self.name, old.value, new_state.value)
        CIRCUIT_TRANSITIONS.labels(breaker=self.name, to=new_state.value).inc()
        self._emit_state()

    def _emit_state(self) -> None:
        """同步 Prometheus 状态 gauge（每 worker 一条序列，liveall 聚合取 max）。"""
        CIRCUIT_STATE.labels(breaker=self.name).set(_STATE_GAUGE_VALUE[self.state])

    @property
    def failure_count(self) -> int:
        """当前窗口内计入的失败次数（测试/诊断只读视图）。"""
        return sum(self._window)

    def get_state(self) -> CircuitState:
        """获取当前状态"""
        return self.state

    def reset(self) -> None:
        """重置熔断器（运维一键恢复；多 worker 下仅重置当前进程，其余靠自动恢复兜底）。"""
        self.state = CircuitState.CLOSED
        self._window.clear()
        self._opened_at = None
        self._half_open_inflight = 0
        self._half_open_successes = 0
        self._emit_state()


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常"""
    pass


class LLMRetryError(Exception):
    """LLM 调用最终失败（所有重试都失败）"""
    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


async def with_retry_and_circuit(
    func,
    *args,
    retry_config: RetryConfig | None = None,
    circuit_breaker: CircuitBreaker | None = None,
    **kwargs,
) -> T:
    """
    带重试和熔断的 LLM 调用包装器（per-attempt 记账）

    原理（litellm 形态：熔断检查与记账下沉到重试循环的每次 attempt）：
    1. 每次 attempt 先 before_call：熔断 OPEN 时毫秒级抛 CircuitOpenError
       （fail-fast，不发起真实调用，上层可立即切 fallback/降级）
    2. 执行成功 → record_success；失败 → record_failure（按 classify_failure 分类，
       调用方错误不污染熔断计数）
    3. 可重试错误按指数退避重试；不可重试或耗尽次数抛 LLMRetryError

    参数：
        func: 要执行的异步函数
        *args: 位置参数
        retry_config: 重试配置
        circuit_breaker: 熔断器实例（None = 只重试不熔断，H-16 自配 client 路径）
        **kwargs: 关键字参数

    返回：
        函数执行结果
    """
    config = retry_config or RetryConfig()
    last_error: Exception | None = None

    for attempt in range(config.max_retries + 1):
        # per-attempt 熔断检查（语义与设计依据见模块 docstring「与旧实现的关键差异」）
        if circuit_breaker is not None:
            await circuit_breaker.before_call()

        try:
            result = await func(*args, **kwargs)
            if circuit_breaker is not None:
                await circuit_breaker.record_success()
            return result
        except Exception as e:
            last_error = e
            if circuit_breaker is not None:
                await circuit_breaker.record_failure(e)

            if not _is_retryable(e, config) or attempt >= config.max_retries:
                logger.error(
                    f"LLM call failed (non-retryable or max retries reached): {e}"
                )
                raise LLMRetryError(
                    f"LLM call failed after {attempt + 1} attempts: {e}",
                    last_error
                ) from last_error

            # 计算延迟：指数退避 + 随机抖动
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            # 添加 0-25% 的随机抖动，避免多请求同时重试
            delay *= (0.75 + random.random() * 0.5)

            LLM_RETRIES.inc()
            logger.warning(
                f"LLM call failed (attempt {attempt + 1}/{config.max_retries + 1}), "
                f"retrying in {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    # 不应该到达这里
    raise LLMRetryError(
        f"LLM call failed after {config.max_retries + 1} attempts",
        last_error
    )


# 全局熔断器实例（按注册名隔离：主链路按 binding（dashscope/deepseek/…）、辅助链路
# asr:{model} / vision:{model} / topic_build——一个供应商故障只熔它自己的闸）
_llm_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_llm_circuit_breaker(name: str = "default") -> CircuitBreaker:
    """获取指定名称的 LLM 熔断器（不存在则按 settings.llm 阈值注册）。

    阈值从 settings.llm 读取（circuit_window_size 等，单进程语义——多 worker 各自
    独立记账，不做 //BACKEND_WORKERS 缩放），settings 不可用时回退默认值。
    """
    if name not in _llm_circuit_breakers:
        # 从 settings 读取阈值；失败则回退硬编码默认值（避免循环导入 / 启动期 settings 缺失）
        try:
            from settings.base import get_settings

            _llm = get_settings().llm
            _cfg = CircuitBreakerConfig(
                window_size=_llm.circuit_window_size,
                minimum_calls=_llm.circuit_minimum_calls,
                failure_rate_threshold=_llm.circuit_failure_rate,
                open_timeout=_llm.circuit_open_timeout,
                half_open_probes=_llm.circuit_half_open_probes,
                half_open_success_needed=_llm.circuit_half_open_success,
            )
            del _llm
        except ImportError:  # pragma: no cover - 仅 settings 模块缺失兜底
            _cfg = CircuitBreakerConfig()

        _llm_circuit_breakers[name] = CircuitBreaker(name=name, config=_cfg)
    return _llm_circuit_breakers[name]


def all_llm_circuit_states() -> dict[str, str]:
    """返回所有已注册 LLM 熔断器的状态快照 {name: state.value}。

    注册名覆盖主链路 binding（dashscope/openai/...）与辅助链路（asr:*/vision:*/
    topic_build）；运维健康检查需一次性看到全部，而非只看主 binding。
    """
    return {name: cb.get_state().value for name, cb in _llm_circuit_breakers.items()}


def reset_all_llm_circuit_breakers() -> int:
    """重置所有已注册的 LLM 熔断器，返回重置数量（运维一键恢复）。

    多 worker 下仅重置当前进程的注册表；其余进程靠 open_timeout 后的 HALF_OPEN
    自动恢复兜底（30s 量级），人工 reset 只是应急加速。
    """
    for cb in _llm_circuit_breakers.values():
        cb.reset()
    return len(_llm_circuit_breakers)
