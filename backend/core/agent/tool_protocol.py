from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    content: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    # 若非 None，tool_dispatch 会设置 DispatchOutcome.pause=True，loop 挂起等待用户回复
    pause_for_user: dict[str, Any] | None = None

    def __str__(self) -> str:
        return self.content
