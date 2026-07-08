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
        prefer_owner: str = "",
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
                binding.platform,
                binding.chat_id or binding.platform_user_id,
                content,
                prefer_owner=prefer_owner,
            )
            if success:
                sent_to.append(binding.platform)

        return sent_to

    async def broadcast(
        self,
        db: AsyncSession,
        course_id: str,
        content: str,
        *,
        prefer_owner: str = "",
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
                binding.platform,
                binding.chat_id or binding.platform_user_id,
                content,
                prefer_owner=prefer_owner,
            )
            if success:
                notified += 1

        return notified

    async def _send_via_channel(
        self, platform: str, chat_id: str, content: str, *, prefer_owner: str = ""
    ) -> bool:
        """Send a message through a running bot that has this channel enabled.

        M-41：优先用 ``prefer_owner``（发起通知的教师/管理员）自己的 bot 作为发送载体，
        避免误用其他用户的 bot（cross-owner：用别人的 bot 发给某学生，上下文/日志归属
        错乱）。若该 owner 无可用 bot，退回任意启用了目标 platform 的运行实例（保证
        广播可用性——教师广播场景下「发出去」优先于「严格归属」）。
        """
        instances = self._bot_manager.all_running_instances()

        def _pick():
            # 1. 优先：prefer_owner 自己的、启用了目标 channel 的 bot
            if prefer_owner:
                for inst in instances:
                    if inst.owner_id == prefer_owner and inst.channel_manager:
                        ch = inst.channel_manager.get_channel(platform)
                        if ch:
                            return inst
            # 2. 退回：任意启用了目标 channel 的运行实例（保持广播可用）
            for inst in instances:
                if inst.channel_manager:
                    ch = inst.channel_manager.get_channel(platform)
                    if ch:
                        return inst
            return None

        instance = _pick()
        if instance is None:
            return False
        channel = instance.channel_manager.get_channel(platform)
        try:
            msg = OutboundMessage(channel=platform, chat_id=chat_id, content=content)
            await channel.send(msg)
            return True
        except Exception:
            logger.exception(
                "Failed to send notification via %s to %s", platform, chat_id
            )
            return False
