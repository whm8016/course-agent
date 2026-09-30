"""HTTP 层错误编号回传规格（可观测性重整 Step 7）。

- middleware 回写 X-Request-Id 响应头（外部 X-Trace-Id 透传回显）
- 500 兜底 handler：detail 字符串带编号（前端 readErrorMessage 兼容）+ code/ref 字段 + 头
"""
from __future__ import annotations

from starlette.requests import Request

import pytest


@pytest.mark.asyncio
async def test_response_carries_request_id_header(client):
    """任意响应带 X-Request-Id；外部传入的 X-Trace-Id 原样回显（跨服务串联）。"""
    r = await client.get("/api/health", headers={"X-Trace-Id": "tracetest1234"})
    assert r.status_code in (200, 503)  # 健康探测结果与 header 断言无关
    assert r.headers.get("X-Request-Id") == "tracetest1234"

    r2 = await client.get("/api/health")  # 未传 → 自动生成 12 位 hex
    assert r2.headers.get("X-Request-Id")


@pytest.mark.asyncio
async def test_unhandled_500_carries_ref(client, monkeypatch):
    """未处理异常 → 500 响应：detail 带「编号 xxx」+ code=internal + ref 字段 + 头。

    detail 必须保持字符串（前端 readErrorMessage 只读 data.detail/message）。
    """
    from core.observability.context import bind_context
    from main import unhandled_exception_handler

    bind_context(request_id="req_abc123456789")
    scope = {
        "type": "http", "http_version": "1.1", "method": "GET",
        "path": "/api/boom", "headers": [], "query_string": b"",
    }
    resp = await unhandled_exception_handler(Request(scope), RuntimeError("internal secret"))
    import json

    body = json.loads(resp.body)
    assert resp.status_code == 500
    assert "编号 req_abc1" in body["detail"]
    assert "internal secret" not in body["detail"]
    assert body["code"] == "internal"
    assert body["ref"] == "req_abc1"
    assert resp.headers.get("X-Request-Id") == "req_abc123456789"
