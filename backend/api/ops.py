"""运维端点（自 main.py 归位）：健康检查 + 熔断器重置。

非业务路由，与 main 的装配职责分离；路径不变（/api/health、/api/health/detailed、
/api/admin/circuit-breaker/reset），instrumentation 的 excluded_handlers 引用不受影响。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.auth import get_current_admin
from core.llm.llm import (
    get_all_llm_circuit_states,
    reset_all_llm_circuit_breakers,
)
from settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])

REDIS_URL = get_settings().db.redis_url.get_secret_value()

@router.get("/health")
async def health():
    checks: dict[str, str] = {}

    # DB check
    try:
        from core.db.database import engine
        from sqlalchemy import text as sa_text
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Redis check
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # LLM check：仅验证 API key 已配置；不做实际调用（避免消耗 token / 拖慢探针）
    try:
        from settings import get_settings
        DASHSCOPE_API_KEY = get_settings().llm.api_key.get_secret_value()
        if DASHSCOPE_API_KEY:
            checks["llm"] = "ok (api_key configured)"
        else:
            checks["llm"] = "error: DASHSCOPE_API_KEY not set"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    all_ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )


@router.get("/health/detailed")
async def health_detailed():
    """
    详细健康检查（包含熔断器和缓存状态）

    用于运维监控和调试
    """
    checks: dict[str, str] = {}
    details: dict = {}

    # DB check
    try:
        from core.db.database import engine
        from sqlalchemy import text as sa_text
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Redis check
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(REDIS_URL, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    # LLM check
    try:
        from settings import get_settings
        DASHSCOPE_API_KEY = get_settings().llm.api_key.get_secret_value()
        if DASHSCOPE_API_KEY:
            checks["llm"] = "ok (api_key configured)"
            # 熔断器按 binding 拆分：展示全部供应商状态；旧单数 key 读主 binding 状态
            #（主流量已不再注册 "default"，读它会永远 closed 造成假绿）
            _cb_states = get_all_llm_circuit_states()
            details["llm_circuit_breakers"] = _cb_states
            details["llm_circuit_breaker"] = _cb_states.get(
                get_settings().llm.binding, "closed"
            )
        else:
            checks["llm"] = "error: DASHSCOPE_API_KEY not set"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    all_ok = all(v.startswith("ok") for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
            "details": details,
        },
    )


@router.post("/admin/circuit-breaker/reset")
async def reset_circuit_breaker(_: dict = Depends(get_current_admin)):
    """
    重置 LLM 熔断器（运维操作）

    当熔断器长时间处于 OPEN 状态时，可以手动重置
    """
    n = reset_all_llm_circuit_breakers()
    return {
        "message": (
            f"所有 LLM 熔断器已重置（共 {n} 个；多 worker 下仅当前进程，"
            f"其余进程靠 HALF_OPEN 自动恢复兜底）"
        ),
        "states": get_all_llm_circuit_states(),
    }
