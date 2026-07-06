"""HTTP 请求观测中间件。

职责：
1. 读取或生成 X-Trace-Id，写入 request_id contexvar，让该请求所有 log 携带它。
2. 请求完成后输出 http.request stage log（方法、路径、状态码、耗时）。

WebSocket 连接不经过 BaseHTTPMiddleware，需在 WS handler 里手动 bind_context。
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .context import bind_context
from .flow import log_flow


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-trace-id") or uuid.uuid4().hex[:12]
        bind_context(request_id=request_id)
        t0 = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
        finally:
            elapsed = int((time.perf_counter() - t0) * 1000)
            path = request.url.path
            # 跳过健康检查和 metrics，减少日志噪音
            if path not in ("/api/health", "/metrics"):
                # call_next 抛异常时 response 仍为 None：记 status=0，
                # 原异常由 finally 自动向上抛出（无需 except:raise，也避免 UnboundLocalError）
                status = getattr(response, "status_code", 0) if response is not None else 0
                log_flow(
                    "http.request",
                    elapsed_ms=elapsed,
                    method=request.method,
                    path=path,
                    status=status,
                )
        return response
