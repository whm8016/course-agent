"""
LLM 调用可靠性增强模块

提供以下功能：
1. 指数退避重试机制 - 处理临时性网络故障和 API 限流
2. 熔断器模式 - 防止持续故障时继续调用造成资源浪费
3. 限流器 - 控制并发请求数量

原理说明：
- 熔断器有三个状态：CLOSED（正常）、OPEN（熔断）、HALF_OPEN（探测）
- 当失败率超过阈值时，熔断器打开，后续请求直接拒绝
- 经过一定时间后，熔断器进入半开状态，允许一个请求探测是否恢复
- 指数退避：每次重试等待时间翻倍，避免对 API 造成压力
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"      # 正常状态，允许请求通过
    OPEN = "open"          # 熔断状态，拒绝请求
    HALF_OPEN = "half_open"  # 半开状态，允许一个探测请求


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
    """熔断器配置"""
    failure_threshold: int = 5       # 触发熔断的连续失败次数
    success_threshold: int = 2      # 关闭熔断所需的连续成功次数
    open_timeout: float = 30.0       # 熔断持续时间（秒）
    half_open_max_calls: int = 1     # 半开状态允许的探测请求数


class CircuitBreaker:
    """
    熔断器实现

    工作原理：
    1. CLOSED 状态：记录每次请求的成功/失败
    2. 连续失败达到阈值时，切换到 OPEN 状态
    3. OPEN 状态下，所有请求立即失败（不调用实际服务）
    4. 超过 open_timeout 后，切换到 HALF_OPEN
    5. HALF_OPEN 状态允许一个请求探测，如果成功则关闭熔断
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name

        # 多 worker 下调整阈值: 总阈值 / worker 数 (轻量方案)
        _workers = int(__import__("os").getenv("BACKEND_WORKERS", "4"))
        _default_config = CircuitBreakerConfig()
        if config is None:
            config = CircuitBreakerConfig(
                failure_threshold=max(2, _default_config.failure_threshold // _workers),
                success_threshold=_default_config.success_threshold,
                open_timeout=_default_config.open_timeout,
                half_open_max_calls=_default_config.half_open_max_calls,
            )
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """通过熔断器执行函数"""
        async with self._lock:
            # 检查是否应该从 OPEN 切换到 HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.config.open_timeout:
                    logger.info(f"CircuitBreaker [{self.name}]: OPEN -> HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0

            # OPEN 状态：直接拒绝
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(f"CircuitBreaker [{self.name}] is OPEN")

            # HALF_OPEN 状态：限制请求数
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"CircuitBreaker [{self.name}] is HALF_OPEN, max calls reached"
                    )
                self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self):
        """记录成功"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"CircuitBreaker [{self.name}]: HALF_OPEN -> CLOSED")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = 0

    async def _on_failure(self):
        """记录失败"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # 探测失败，重新打开熔断器
                logger.info(
                    f"CircuitBreaker [{self.name}]: HALF_OPEN -> OPEN "
                    f"(probe failed, failures={self.failure_count})"
                )
                self.state = CircuitState.OPEN
                self.success_count = 0
            elif self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"CircuitBreaker [{self.name}]: CLOSED -> OPEN "
                    f"(failures={self.failure_count})"
                )
                self.state = CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """获取当前状态"""
        return self.state

    def reset(self):
        """重置熔断器"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0


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
    带重试和熔断的 LLM 调用包装器

    原理：
    1. 首先检查熔断器状态
    2. 执行函数，捕获异常
    3. 如果是可重试的错误，使用指数退避策略重试
    4. 如果达到最大重试次数或不可重试的错误，抛出异常

    参数：
        func: 要执行的异步函数
        *args: 位置参数
        retry_config: 重试配置
        circuit_breaker: 熔断器实例
        **kwargs: 关键字参数

    返回：
        函数执行结果
    """
    config = retry_config or RetryConfig()

    async def _call():
        last_error: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except CircuitOpenError:
                # 熔断器打开，直接抛出
                raise
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # 判断是否可重试
                is_retryable = False

                # 检查错误消息中的关键词
                retryable_keywords = [
                    'timeout', 'rate limit', 'too many requests',
                    'connection', 'temporarily unavailable',
                    'service unavailable', 'bad gateway',
                    'gateway timeout', 'reset by peer'
                ]
                if any(kw in error_str for kw in retryable_keywords):
                    is_retryable = True

                # 检查 HTTP 429（限流）
                if '429' in str(e):
                    is_retryable = True

                if not is_retryable or attempt >= config.max_retries:
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
                import random
                delay *= (0.75 + random.random() * 0.5)

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

    if circuit_breaker:
        return await circuit_breaker.call(_call)
    else:
        return await _call()


# 全局熔断器实例
_llm_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_llm_circuit_breaker(name: str = "default") -> CircuitBreaker:
    """获取指定名称的 LLM 熔断器

    阈值从 settings 读取（llm_circuit_*），settings 不可用时回退默认值。
    传给 CircuitBreaker 的 failure_threshold 仍会经其 __init__ 内的
    BACKEND_WORKERS 除法调整，以适配多 worker 场景。
    """
    if name not in _llm_circuit_breakers:
        # 从 settings 读取阈值；失败则回退硬编码默认值（避免循环导入 / 启动期 settings 缺失）
        try:
            from settings.base import get_settings

            _s = get_settings()
            _fail = _s.llm_circuit_failure_threshold
            _succ = _s.llm_circuit_success_threshold
            _open = _s.llm_circuit_open_timeout
        except Exception:  # pragma: no cover
            _fail, _succ, _open = 5, 2, 30.0

        _llm_circuit_breakers[name] = CircuitBreaker(
            name=f"llm_{name}",
            config=CircuitBreakerConfig(
                failure_threshold=_fail,
                success_threshold=_succ,
                open_timeout=_open,
            )
        )
    return _llm_circuit_breakers[name]
