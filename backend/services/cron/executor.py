"""Cron job 执行器。精确对齐 DeepTutor services/cron/executor.py。

partner job：通过 bot 的 AgentLoop.process_direct() 执行，结果经
MessageBus → _outbound_router → ChannelManager 自动推向 QQ/飞书。
"""

from __future__ import annotations

import logging

from services.cron.service import CronJob

logger = logging.getLogger(__name__)


def _reminder_prompt(job: CronJob) -> str:
    return (
        "定时提醒时间已到，请现在用简洁自然的语言直接告知用户以下提醒内容，"
        "不要提及调度器状态或任务 ID。\n\n"
        f"提醒内容：{job.message}"
    )


async def execute_job(job: CronJob) -> tuple[str, str | None]:
    """返回 (status, error)，status 为 ok / error / skipped。"""
    if job.owner.kind == "partner":
        return await _execute_partner_job(job)
    return "skipped", f"unsupported owner kind: {job.owner.kind!r}"


async def _execute_partner_job(job: CronJob) -> tuple[str, str | None]:
    """通过 bot AgentLoop 执行提醒，结果推向对应 IM channel。"""
    from core.bot.bus.events import OutboundMessage
    from core.bot.manager import get_bot_manager

    partner_id = job.owner.partner_id
    # partner_id 格式 "<owner>:<bot_id>"（legacy 无冒号 → owner 为空）
    if ":" in partner_id:
        owner_id, bot_id = partner_id.split(":", 1)
    else:
        owner_id, bot_id = "", partner_id
    manager = get_bot_manager()
    instance = manager.get_bot(owner_id, bot_id)
    if not instance or not instance.running:
        # 学生 bot 按需启动：cron 到点才拉起
        try:
            instance = await manager.start_bot(owner_id, bot_id)
        except Exception as exc:
            logger.warning("Cron job %s: bot %s lazy start failed: %s", job.id, partner_id, exc)
            return "skipped", f"bot '{partner_id}' 启动失败"

    channel_name = (job.owner.channel or "web").strip()
    chat_id = (job.owner.chat_id or "").strip()

    try:
        result = await instance.agent_loop.process_direct(
            _reminder_prompt(job),
            session_key=job.owner.session_key or None,
            channel=channel_name,
            chat_id=chat_id or "cron",
            user_id=job.owner.user_id,
        )
    except Exception as exc:
        logger.exception("Partner cron job %s 执行失败", job.id)
        return "error", f"{type(exc).__name__}: {exc}"

    text = (result or "").strip()
    if not text:
        return "error", "agent 未产生回复"

    # process_direct 只返回文本，不会走 MessageBus；按渠道触达
    if channel_name != "web" and chat_id:
        # IM 渠道（QQ/飞书）：显式 send 实时推送
        if not instance.channel_manager:
            return "error", "bot 未配置 IM channel"
        channel = instance.channel_manager.get_channel(channel_name)
        if not channel:
            return "error", f"channel '{channel_name}' 未启用"
        try:
            await channel.send(
                OutboundMessage(channel=channel_name, chat_id=chat_id, content=text)
            )
            logger.info(
                "Cron job %s: reminder sent via %s to %s",
                job.id, channel_name, chat_id,
            )
        except Exception as exc:
            logger.exception("Cron job %s: IM send failed", job.id)
            return "error", f"IM 发送失败: {exc}"
    elif channel_name == "web" and job.owner.user_id:
        # web 渠道：落库 BotNotification，前端轮询拉取（补齐 web 触达最后一公里）
        try:
            from core.db.database import AsyncSessionLocal, BotNotification
            async with AsyncSessionLocal() as db:
                db.add(BotNotification(
                    user_id=job.owner.user_id,
                    bot_id=bot_id,
                    content=text,
                ))
                await db.commit()
            logger.info("Cron job %s: web notification stored for user %s", job.id, job.owner.user_id)
        except Exception as exc:
            logger.exception("Cron job %s: web notification store failed", job.id)
            return "error", f"web 通知落库失败: {exc}"

    return "ok", None


__all__ = ["execute_job"]
