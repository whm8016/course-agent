"""ManagerAgent - Queue management Agent (faithful port from DeepTutor)."""

from __future__ import annotations

import asyncio
from typing import Any

from ..data_structures import DynamicTopicQueue, TopicBlock


class ManagerAgent:
    """Queue management Agent"""

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
    ):
        self.queue: DynamicTopicQueue | None = None
        self.primary_topic: str | None = None
        self._lock = asyncio.Lock()

    def set_queue(self, queue: DynamicTopicQueue) -> None:
        self.queue = queue

    def set_primary_topic(self, topic: str | None) -> None:
        self.primary_topic = (topic or "").strip() or None

    def get_next_task(self) -> TopicBlock | None:
        if not self.queue:
            return None
        block = self.queue.get_pending_block()
        if block:
            self.queue.mark_researching(block.block_id)
            print(f"\nManagerAgent: Assigned task {block.block_id} ({block.sub_topic})")
        return block

    def complete_task(self, block_id: str) -> bool:
        if not self.queue:
            return False
        success = self.queue.mark_completed(block_id)
        if success:
            print(f"ManagerAgent: Task {block_id} completed")
        return success

    def fail_task(self, block_id: str, reason: str = "") -> bool:
        if not self.queue:
            return False
        success = self.queue.mark_failed(block_id)
        if success:
            print(f"ManagerAgent: Task {block_id} failed" + (f" — {reason}" if reason else ""))
        return success

    def add_new_topic(self, sub_topic: str, overview: str) -> TopicBlock | None:
        if not self.queue:
            raise RuntimeError("Queue not initialized")
        normalized = (sub_topic or "").strip()
        if not normalized:
            raise ValueError("New topic title cannot be empty")
        if self.queue.has_topic(normalized):
            print(f"ManagerAgent: Topic '{normalized}' already exists, skipping")
            return None
        block = self.queue.add_block(normalized, overview)
        print(f"ManagerAgent: Added new topic {block.block_id} '{sub_topic}'")
        return block

    def is_research_complete(self) -> bool:
        return bool(self.queue) and self.queue.is_all_completed()

    def get_queue_status(self) -> dict[str, Any]:
        if not self.queue:
            return {}
        stats = self.queue.get_statistics()
        print(
            f"\nQueue Status: total={stats['total_blocks']} pending={stats['pending']} "
            f"researching={stats['researching']} completed={stats['completed']} failed={stats['failed']}"
        )
        return stats

    # ------------------------------------------------------------------
    # Async thread-safe wrappers
    # ------------------------------------------------------------------

    async def get_next_task_async(self) -> TopicBlock | None:
        async with self._lock:
            return self.get_next_task()

    async def complete_task_async(self, block_id: str) -> bool:
        async with self._lock:
            return self.complete_task(block_id)

    async def fail_task_async(self, block_id: str, reason: str = "") -> bool:
        async with self._lock:
            return self.fail_task(block_id, reason)

    async def add_new_topic_async(self, sub_topic: str, overview: str) -> TopicBlock | None:
        async with self._lock:
            return self.add_new_topic(sub_topic, overview)

    async def process(self, *args: Any, **kwargs: Any) -> None:
        """Manager is called through task management methods; no standalone process needed."""


__all__ = ["ManagerAgent"]
