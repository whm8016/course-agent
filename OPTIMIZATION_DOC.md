# 高校智能体助手 - 优化改进文档

> 本文档详细说明本次中高优先级优化的内容、原理和代码解释

## 目录

1. [前端测试配置修复](#1-前端测试配置修复)
2. [Nginx 生产配置优化](#2-nginx-生产配置优化)
3. [LLM 可靠性增强](#3-llm-可靠性增强)
4. [RAG 查询缓存](#4-rag-查询缓存)
5. [增强的健康检查](#5-增强的健康检查)

---

## 1. 前端测试配置修复

### 问题描述

前端 Vitest 测试失败，错误信息：
```
Cannot read properties of undefined (reading 'clear')
Cannot read properties of undefined (reading 'setItem')
```

**根本原因**：Vitest 运行在 jsdom 环境中，默认不提供 `localStorage` 和 `sessionStorage` API。

### 解决方案

**文件**: `frontend/src/__tests__/setup.ts`

```typescript
// 核心代码
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key]
    }),
    clear: vi.fn(() => {
      store = {}
    }),
    get length() {
      return Object.keys(store).length
    },
    key: vi.fn((i: number) => Object.keys(store)[i] ?? null),
  }
})()

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
})
```

### 原理说明

1. **Mock 模式**：使用 IIFE（立即执行函数表达式）创建闭包存储
2. **Vitest spy**：使用 `vi.fn()` 创建可追踪的函数，用于测试断言
3. **globalThis**：在全局对象上定义属性，确保 TypeScript 类型兼容

### 测试结果

```
✓ src/__tests__/auth.test.ts (14 tests)
✓ src/__tests__/api.test.ts (9 tests)
✓ src/__tests__/LoginPage.test.tsx (8 tests)

Test Files  3 passed (3)
Tests       31 passed (31)
```

---

## 2. Nginx 生产配置优化

### 问题描述

原有 nginx.conf 配置过于简单，缺少：
- Gzip 压缩
- 安全响应头
- 静态资源缓存
- 监控端点保护

### 解决方案

**文件**: `frontend/nginx.conf`

#### 2.1 Gzip 压缩

```nginx
gzip on;
gzip_vary on;
gzip_min_length 1024;
gzip_proxied any;
gzip_comp_level 6;
gzip_types
    text/plain text/css text/javascript
    application/javascript application/json
    application/xml application/xml+rss
    font/woff font/woff2 font/ttf;
```

**原理**：
- `gzip_comp_level 6`：压缩级别 1-9，6 是性能和压缩比的平衡点
- `gzip_min_length 1024`：小于 1KB 的内容压缩收益不大
- `gzip_types`：只压缩文本类资源，图片等已压缩格式不需要

#### 2.2 安全响应头

```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
```

**原理**：
- `X-Frame-Options`：防止点击劫持攻击
- `X-Content-Type-Options`：防止 MIME 类型嗅探
- `X-XSS-Protection`：XSS 过滤器（现代浏览器已内置）
- `Referrer-Policy`：控制 Referer 头泄露

#### 2.3 静态资源缓存

```nginx
location ~* \.(js|css|png|jpg|ico|svg|woff|woff2)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

**原理**：
- `expires 1y`：缓存 1 年（长期缓存策略）
- `immutable`：告诉浏览器资源永不变，避免重新验证
- 使用内容哈希命名的文件才适合这种策略

#### 2.4 SPA 路由支持

```nginx
location / {
    try_files $uri $uri/ /index.html;
}
```

**原理**：将所有 404 请求重定向到 index.html，让 React Router 处理路由

#### 2.5 监控端点保护

```nginx
location /metrics {
    allow 127.0.0.1;
    allow 10.0.0.0/8;
    allow 172.16.0.0/12;
    allow 192.168.0.0/16;
    deny all;
}
```

**原理**：只允许内网访问 Prometheus 指标端点

---

## 3. LLM 可靠性增强

### 问题描述

LLM API 调用面临以下风险：
- 网络瞬时故障
- API 限流（429 错误）
- 服务端临时不可用（502/503）
- 持续故障时的资源浪费

### 解决方案

**新增文件**: `backend/core/llm/reliability.py`

#### 3.1 指数退避重试机制

```python
@dataclass
class RetryConfig:
    max_retries: int = 3           # 最大重试次数
    base_delay: float = 1.0         # 基础延迟（秒）
    max_delay: float = 30.0         # 最大延迟（秒）
    exponential_base: float = 2.0    # 指数退避基数
```

**重试延迟计算**：
```
delay = min(base_delay * (exponential_base ** attempt), max_delay)
delay *= (0.75 + random() * 0.5)  # 添加随机抖动
```

**示例**（base_delay=1, max_delay=30）：
- 第 1 次重试：1-1.5 秒
- 第 2 次重试：2-3 秒
- 第 3 次重试：4-6 秒

**原理**：
- 指数退避避免对故障服务造成更大压力
- 随机抖动防止"惊群效应"（多客户端同时重试）

#### 3.2 熔断器模式

```python
class CircuitBreaker:
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.state = CircuitState.CLOSED
        self.failure_count = 0

    async def call(self, func, *args, **kwargs):
        # 检查熔断器状态
        if self.state == CircuitState.OPEN:
            raise CircuitOpenError("Circuit breaker is OPEN")
        # 执行请求
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception:
            await self._on_failure()
            raise
```

**熔断器状态机**：

```
                    ┌─────────────────┐
                    │     CLOSED       │ ← 正常状态
                    │  (允许请求通过)   │
                    └────────┬─────────┘
                             │ 连续失败 >= 5
                             ▼
                    ┌─────────────────┐
                    │      OPEN       │ ← 熔断状态
                    │ (拒绝所有请求)   │
                    └────────┬─────────┘
                             │ 30秒后
                             ▼
                    ┌─────────────────┐
                    │   HALF_OPEN     │ ← 探测状态
                    │ (允许1个探测)   │
                    └────────┬─────────┘
                             │ 探测成功 >= 2
                             ▼
                    ┌─────────────────┐
                    │     CLOSED      │
                    └─────────────────┘
```

**原理**：
- 快速失败：熔断打开后立即返回错误，不等待超时
- 防止雪崩：避免持续向故障服务发请求
- 自动恢复：半开状态探测服务是否恢复

#### 3.3 集成到 LLM 模块

```python
# backend/core/llm/llm.py

async def _make_chat_completion(model, messages, stream, ...):
    async def _call():
        return await client.chat.completions.create(...)

    return await with_retry_and_circuit(
        _call,
        retry_config=_retry_config,
        circuit_breaker=_llm_circuit_breaker,
    )
```

#### 3.4 错误分类

```python
# 可重试的错误
is_retryable = False
retryable_keywords = [
    'timeout', 'rate limit', 'too many requests',
    'connection', 'temporarily unavailable',
]
if any(kw in error_str for kw in retryable_keywords):
    is_retryable = True
```

---

## 4. RAG 查询缓存

### 问题描述

- 相同查询重复调用 RAG，浪费 API 调用
- 知识库不经常变化，缓存收益高
- 无缓存统计，无法监控命中率

### 解决方案

**新增文件**: `backend/core/rag/cache.py`

#### 4.1 缓存键设计

```python
def _compute_query_hash(course_id: str, query: str, top_k: int) -> str:
    """使用 SHA-256 生成缓存键"""
    raw = json.dumps({
        "course_id": course_id,
        "query": query,
        "top_k": top_k,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
```

**原理**：
- SHA-256 生成固定长度哈希
- `sort_keys=True` 保证相同内容不同顺序产生相同哈希
- 截断到 32 字符足够唯一且键长度可控

#### 4.2 缓存读写

```python
async def get(self, course_id: str, query: str, top_k: int) -> list[dict] | None:
    """获取缓存，如命中返回结果，否则返回 None"""
    if not self._redis:
        return None  # Redis 不可用时禁用缓存

    key = f"rag_cache:{course_id}:{query_hash}"
    cached = await self._redis.get(key)
    if cached:
        _cache_stats["hits"] += 1
        return json.loads(cached)
    else:
        _cache_stats["misses"] += 1
        return None
```

#### 4.3 缓存模式

```python
async def get_or_set(
    self,
    course_id: str,
    query: str,
    top_k: int,
    fetch_func,  # 异步获取函数
) -> list[dict]:
    """
    缓存友好的 fetch 模式

    原理：
    1. 先查缓存
    2. 命中则直接返回
    3. 未命中则调用 fetch_func
    4. 存入缓存后返回
    """
    cached = await self.get(course_id, query, top_k)
    if cached is not None:
        return cached

    results = await fetch_func()
    await self.set(course_id, query, top_k, results)
    return results
```

#### 4.4 TTL 和失效

```python
# TTL 设置
await self._redis.setex(key, ttl_seconds, serialized)

# 知识库更新时失效缓存
async def invalidate(self, course_id: str | None = None) -> int:
    """失效缓存，支持按课程失效"""
    pattern = f"rag_cache:{course_id}:*" if course_id else "rag_cache:*"
    # 使用 SCAN 避免 KEYS 阻塞
    while cursor != 0:
        cursor, keys = await redis.scan(cursor=cursor, match=pattern)
        await redis.delete(*keys)
```

#### 4.5 缓存统计

```python
@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total
```

---

## 5. 增强的健康检查

### 问题描述

原有健康检查只检查 DB、Redis、LLM 状态，缺少：
- 熔断器状态
- 缓存状态
- 运维操作接口

### 解决方案

#### 5.1 详细健康检查

```python
@app.get("/api/health/detailed")
async def health_detailed():
    """详细健康检查 - 包含熔断器和缓存状态"""
    details = {
        "llm_circuit_breaker": get_llm_circuit_state(),
        "rag_cache": get_rag_cache_stats(),
    }
    return JSONResponse(content={
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
        "details": details,
    })
```

#### 5.2 熔断器重置

```python
@app.post("/api/admin/circuit-breaker/reset")
async def reset_circuit_breaker():
    """手动重置 LLM 熔断器"""
    reset_llm_circuit_breaker()
    return {"state": get_llm_circuit_state()}
```

**使用场景**：当 LLM 服务长时间不可用后恢复，运维人员可手动重置熔断器

---

## 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        客户端请求                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Nginx 反向代理                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Gzip压缩 │  │ 安全头   │  │ 静态缓存 │  │ API 代理     │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI 应用层                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    LLM 可靠性层                            │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │   熔断器    │───▶│ 指数退避    │───▶│   LLM API   │  │  │
│  │  │ CircuitBreaker│  │ Retry      │    │ (DashScope) │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    RAG 缓存层                             │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │  │   查询缓存   │───▶│ 知识检索    │───▶│  LightRAG   │  │  │
│  │  │ (Redis)     │    │ Chroma/FS   │    │             │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MAX_CONCURRENT_LLM` | LLM 并发限制 | 25 |
| `RAG_CACHE_TTL` | 缓存过期时间(秒) | 3600 |
| `LLM_RETRY_MAX` | 最大重试次数 | 3 |
| `LLM_CIRCUIT_THRESHOLD` | 熔断失败阈值 | 5 |

### 监控指标

- `GET /api/health` - 基础健康检查
- `GET /api/health/detailed` - 详细健康检查（含缓存和熔断器状态）
- `GET /metrics` - Prometheus 指标
- `POST /api/admin/circuit-breaker/reset` - 重置熔断器

---

## 性能提升预期

| 优化项 | 预期效果 |
|--------|----------|
| Gzip 压缩 | 传输体积减少 60-80% |
| 静态资源缓存 | 首屏加载时间减少 50% |
| RAG 缓存 | 命中时响应时间减少 90% |
| 指数退避重试 | API 临时故障恢复时间减少 |
| 熔断器 | 防止级联故障，服务可用性提升 |

---

## 后续建议

1. **添加真实监控**：接入 Prometheus + Grafana
2. **日志聚合**：接入 ELK 或 Loki
3. **告警规则**：基于熔断器状态和缓存命中率设置告警
4. **性能测试**：使用 wrk/k6 进行压力测试验证
