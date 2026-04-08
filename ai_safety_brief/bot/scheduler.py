"""Scheduled delivery loop."""

from __future__ import annotations

import logging
from datetime import timedelta

from ai_safety_brief.bot.ui import build_alert_keyboard, build_digest_keyboard
from telegram.ext import Application, ContextTypes

from ai_safety_brief.bot.handlers import get_runtime
from ai_safety_brief.utils.time import compute_next_run, utc_now

logger = logging.getLogger(__name__)


def schedule_jobs(application: Application) -> None:
    runtime = get_runtime(application)
    application.job_queue.run_repeating(
        scheduled_delivery_check,
        interval=runtime.settings.scheduler_poll_seconds,
        first=5,
        name="scheduled_delivery_check",
    )
    application.job_queue.run_repeating(
        scheduled_alert_check,
        interval=300,
        first=30,
        name="scheduled_alert_check",
    )


async def scheduled_delivery_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    runtime = get_runtime(context.application)
    now = utc_now()
    chats = await runtime.db.list_due_chats(now)
    for chat in chats:
        lock = runtime.chat_lock(chat.chat_id)
        if lock.locked():
            continue
        async with lock:
            try:
                result = await runtime.pipeline.generate_digest(chat, triggered_by="schedule")
                for index, message in enumerate(result.messages):
                    reply_markup = build_digest_keyboard() if index == len(result.messages) - 1 else None
                    await context.bot.send_message(chat_id=chat.chat_id, text=message, reply_markup=reply_markup)
                chat.last_digest_at = result.generated_at
                chat.next_run_at = compute_next_run(chat, now_utc=result.generated_at)
                await runtime.db.update_chat(
                    chat.chat_id,
                    last_digest_at=result.generated_at.isoformat(),
                    next_run_at=chat.next_run_at.isoformat(),
                )
            except Exception as exc:
                logger.exception("Scheduled digest failed for chat %s", chat.chat_id)
                retry_at = now + timedelta(minutes=15)
                await runtime.db.save_digest_run(
                    chat.chat_id,
                    [],
                    [f"Scheduled digest failed: {exc}"],
                    triggered_by="schedule",
                    status="failed",
                    error=str(exc),
                )
                await runtime.db.update_chat(chat.chat_id, next_run_at=retry_at.isoformat())


async def scheduled_alert_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    runtime = get_runtime(context.application)
    chats = await runtime.db.list_alert_enabled_chats()
    for chat in chats:
        lock = runtime.chat_lock(chat.chat_id)
        if lock.locked():
            continue
        async with lock:
            try:
                result = await runtime.pipeline.generate_alert(chat)
                if not result:
                    continue
                await context.bot.send_message(
                    chat_id=chat.chat_id,
                    text=result.message,
                    reply_markup=build_alert_keyboard(result.item),
                )
                await runtime.db.mark_alerted(chat.chat_id, result.item.id, result.generated_at)
            except Exception:
                logger.exception("scheduled alert failed for chat %s", chat.chat_id)
