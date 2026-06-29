"""core/orchestrator.py 路由逻辑单元测试。

覆盖三个场景：
  1. 已知 mode → 路由到正确 capability，收到其发出的事件
  2. 未知 mode → 返回 type=error 事件，包含「未知能力」说明
  3. capability 抛异常 → orchestrator 捕获并发 error 事件，不向上抛
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core.capability_protocol import BaseCapability, CapabilityManifest
from core.context import UnifiedContext
from core.orchestrator import CourseOrchestrator
from core.registry import CapabilityRegistry
from core.stream_bus import StreamBus


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _make_ctx(mode: str = "chat") -> UnifiedContext:
    return UnifiedContext(
        course_id="c1",
        user_id="u1",
        user_message="test question",
        mode=mode,
    )


async def _collect(gen) -> list[dict]:
    """把 async generator 里的 StreamEvent 统一转为 plain dict。"""
    events = []
    async for event in gen:
        events.append(event.to_dict())
    return events


class _SimpleCapability(BaseCapability):
    """发出一个 done 事件后结束，记录被调用的 mode。"""

    manifest = CapabilityManifest(name="chat", description="test cap", stages=[])
    called_modes: list[str] = []

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        self.called_modes.append(context.mode or "")
        await stream.emit({"type": "done", "metadata": {}})


class _CrashCapability(BaseCapability):
    """run() 直接抛 RuntimeError，测试 orchestrator 的异常处理。"""

    manifest = CapabilityManifest(name="chat", description="crash cap", stages=[])

    async def run(self, context: UnifiedContext, stream: StreamBus) -> None:
        raise RuntimeError("capability crash")


def _make_registry(*capabilities: BaseCapability) -> CapabilityRegistry:
    reg = CapabilityRegistry()
    for cap in capabilities:
        reg.register(cap)
    return reg


# ---------------------------------------------------------------------------
# 场景 1：已知 mode → 路由到正确 capability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_known_mode_routes_to_capability():
    """context.mode='chat' 时，orchestrator 路由到 ChatCapability 并收到其 done 事件。"""
    cap = _SimpleCapability()
    cap.called_modes.clear()

    with patch("core.orchestrator.get_capability_registry", return_value=_make_registry(cap)):
        orch = CourseOrchestrator()
        events = await _collect(orch.handle(_make_ctx("chat")))

    assert cap.called_modes == ["chat"]
    assert any(e["type"] == "done" for e in events)


# ---------------------------------------------------------------------------
# 场景 2：未知 mode → error 事件
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_mode_emits_error():
    """未注册的 mode 应返回 type=error 事件，包含「未知能力」说明。"""
    empty_reg = _make_registry()  # 空注册表

    with patch("core.orchestrator.get_capability_registry", return_value=empty_reg):
        orch = CourseOrchestrator()
        events = await _collect(orch.handle(_make_ctx("nonexistent")))

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert "未知能力" in events[0].get("message", "")


# ---------------------------------------------------------------------------
# 场景 3：capability 抛异常 → error 事件，不崩溃
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capability_exception_emits_error():
    """Capability.run() 抛异常时，orchestrator 捕获并发 error 事件，不向上抛。"""
    cap = _CrashCapability()

    with patch("core.orchestrator.get_capability_registry", return_value=_make_registry(cap)):
        orch = CourseOrchestrator()
        events = await _collect(orch.handle(_make_ctx("chat")))

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) >= 1
    assert any("capability crash" in e.get("message", "") for e in error_events)
