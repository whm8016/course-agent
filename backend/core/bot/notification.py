"""Notification service — push messages to students via social platforms."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.bot.bus.events import OutboundMessage
from core.db.database import Enrollment, UserSocialBinding

logger = logging.getLogger(__name__)


class NotificationService:
    """Push messages to students through their bound social platform channels."""

    def __init__(self, bot_manager: Any):
        self._bot_manager = bot_manager

    async def push_to_student(
        self,
        db: AsyncSession,
        user_id: str,
        content: str,
        *,
        platform: str | None = None,
    ) -> list[str]:
        """Push a message to a specific student.

        Returns list of platforms the message was sent to.
        """
        query = select(UserSocialBinding).where(UserSocialBinding.user_id == user_id)
        if platform:
            query = query.where(UserSocialBinding.platform == platform)
        result = await db.execute(query)
        bindings = result.scalars().all()

        sent_to: list[str] = []
        for binding in bindings:
            success = await self._send_via_channel(
                binding.platform, binding.chat_id or binding.platform_user_id, content
            )
            if success:
                sent_to.append(binding.platform)

        return sent_to

    async def broadcast(
        self,
        db: AsyncSession,
        course_id: str,
        content: str,
    ) -> int:
        """Broadcast a message to all students enrolled in a course.

        Returns count of students notified.
        """
        # Find all students enrolled in this course
        enrollment_result = await db.execute(
            select(Enrollment.student_id).where(Enrollment.course_id == course_id)
        )
        student_ids = [row[0] for row in enrollment_result.all()]

        if not student_ids:
            return 0

        # Find bindings for all enrolled students
        bindings_result = await db.execute(
            select(UserSocialBinding).where(UserSocialBinding.user_id.in_(student_ids))
        )
        bindings = bindings_result.scalars().all()

        notified = 0
        for binding in bindings:
            success = await self._send_via_channel(
                binding.platform, binding.chat_id or binding.platform_user_id, content
            )
            if success:
                notified += 1

        return notified

    async def _send_via_channel(self, platform: str, chat_id: str, content: str) -> bool:
        """Send a message through a specific channel."""
        # Find a running bot that has this channel enabled
        for bot_info in self._bot_manager.list_bots():
            if not bot_info.get("running"):
                continue
            if platform in bot_info.get("channels", []):
                instance = self._bot_manager.get_bot(bot_info["bot_id"])
                if instance and instance.channel_manager:
                    channel = instance.channel_manager.get_channel(platform)
                    if channel:
                        try:
                            msg = OutboundMessage(
                                channel=platform, chat_id=chat_id, content=content
                            )
                            await channel.send(msg)
                            return True
                        except Exception:
                            logger.exception(
                                "Failed to send notification via %s to %s", platform, chat_id
                            )
        return False
