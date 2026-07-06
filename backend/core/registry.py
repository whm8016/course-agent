"""
Capability Registry
===================

将能力名称映射到 BaseCapability 实例。

【架构角色】
- CourseOrchestrator 在启动时注册所有内置能力。
- 运行时按 context.mode 查找对应能力实例。

使用示例：

    registry = CapabilityRegistry()
    registry.register(ChatCapability())
    registry.register(DeepSolveCapability())

    cap = registry.get("deep_solve")   # -> DeepSolveCapability 实例
    cap = registry.get("unknown")      # -> None
"""
from __future__ import annotations

import logging

from core.capability_protocol import BaseCapability

logger = logging.getLogger(__name__)


class CapabilityRegistry:
    """简单 dict 注册表，将能力名称映射到实例。"""

    def __init__(self) -> None:
        self._capabilities: dict[str, BaseCapability] = {}

    def register(self, capability: BaseCapability) -> None:
        """注册一个能力实例，允许覆盖同名能力。"""
        self._capabilities[capability.name] = capability
        logger.debug("CapabilityRegistry: registered '%s'", capability.name)

    def get(self, name: str) -> BaseCapability | None:
        """按名称查找能力，未找到返回 None。"""
        return self._capabilities.get(name)

    def list_capabilities(self) -> list[str]:
        return list(self._capabilities.keys())

    def get_manifests(self) -> list[dict]:
        return [
            {
                "name": cap.manifest.name,
                "description": cap.manifest.description,
                "stages": cap.manifest.stages,
            }
            for cap in self._capabilities.values()
        ]


# ---- 全局单例 ----

_registry: CapabilityRegistry | None = None


def get_capability_registry() -> CapabilityRegistry:
    """返回已初始化的全局注册表（懒加载）。"""
    global _registry
    if _registry is None:
        _registry = _build_default_registry()
    return _registry


def _build_default_registry() -> CapabilityRegistry:
    """注册所有内置能力并返回注册表。"""
    from core.capabilities.chat import ChatCapability
    from core.capabilities.deep_solve import DeepSolveCapability
    from core.capabilities.deep_research import DeepResearchCapability
    from core.capabilities.quiz import QuizCapability

    reg = CapabilityRegistry()
    reg.register(ChatCapability())
    reg.register(DeepSolveCapability())
    reg.register(DeepResearchCapability())
    reg.register(QuizCapability())
    return reg
