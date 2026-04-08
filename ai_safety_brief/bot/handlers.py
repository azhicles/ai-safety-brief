"""Telegram command handlers."""

from __future__ import annotations

import logging
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import Application, ContextTypes

from ai_safety_brief.bot.runtime import Runtime
from ai_safety_brief.models import ChatSettings
from ai_safety_brief.utils.time import compute_next_run, format_schedule, parse_weekdays

logger = logging.getLogger(__name__)


def get_runtime(application: Application) -> Runtime:
    return application.bot_data["runtime"]


async def require_registered(update: Update, context: ContextTypes.DEFAULT_TYPE) -> ChatSettings | None:
    chat = update.effective_chat
    if not chat or not update.effective_message:
        return None
    runtime = get_runtime(context.application)
    stored = await runtime.db.get_chat(chat.id)
    if not stored:
        await update.effective_message.reply_text("Use /start first to register this chat.")
        return None
    return stored


async def can_manage_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    if not chat or not user:
        return False
    if chat.type == "private":
        return True
    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in {"creator", "administrator"}


def chat_defaults(runtime: Runtime) -> dict[str, object]:
    disabled = [
        source.key
        for source in runtime.collector.list_sources()
        if not source.enabled_by_default
    ]
    base = ChatSettings(
        chat_id=0,
        timezone=runtime.settings.default_timezone,
        top_k=runtime.settings.default_top_k,
        schedule_type="daily",
        schedule_value="",
        send_hour=runtime.settings.default_send_hour,
        send_minute=runtime.settings.default_send_minute,
        disabled_sources=disabled,
        repeat_window_days=runtime.settings.default_repeat_window_days,
    )
    next_run = compute_next_run(base)
    return {
        "timezone": base.timezone,
        "top_k": base.top_k,
        "schedule_type": base.schedule_type,
        "schedule_value": base.schedule_value,
        "send_hour": base.send_hour,
        "send_minute": base.send_minute,
        "repeat_window_days": base.repeat_window_days,
        "next_run_at": next_run.isoformat(),
    }


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return
    runtime = get_runtime(context.application)
    chat = await runtime.db.upsert_chat(
        update.effective_chat.id,
        update.effective_chat.type,
        update.effective_chat.title or getattr(update.effective_chat, "full_name", "") or "",
        defaults=chat_defaults(runtime),
    )
    await update.effective_message.reply_text(
        (
            "AI Safety Brief is ready.\n\n"
            "Commands:\n"
            "/brief - generate an immediate digest\n"
            "/status - current schedule and settings\n"
            "/sources - list source toggles\n"
            "/settings - view or change k, cadence, timezone, and sources\n"
            "/pause and /resume - stop or restart scheduled sends\n"
            "/history - recent digest runs\n\n"
            f"Current schedule: {format_schedule(chat)} ({chat.timezone})"
        )
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        (
            "AI Safety Brief commands:\n"
            "/start\n"
            "/brief\n"
            "/status\n"
            "/history\n"
            "/sources\n"
            "/settings\n"
            "/settings k 5\n"
            "/settings timezone Asia/Singapore\n"
            "/settings cadence daily 19:00\n"
            "/settings cadence hourly 6\n"
            "/settings cadence weekly Mon,Wed,Fri 18:30\n"
            "/settings source enable x_openai\n"
            "/settings source disable lesswrong\n"
            "/pause\n"
            "/resume"
        )
    )


async def brief_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    lock = runtime.chat_lock(chat.chat_id)
    if lock.locked():
        await update.effective_message.reply_text("A digest run is already in progress for this chat.")
        return
    await update.effective_message.reply_text("Building the latest AI safety brief...")
    async with lock:
        result = await runtime.pipeline.generate_digest(chat, triggered_by="manual")
    for message in result.messages:
        await update.effective_message.reply_text(message)


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    enabled_count = len(runtime.collector.enabled_sources(chat))
    next_run = (
        chat.next_run_at.astimezone(ZoneInfo(chat.timezone)).strftime("%Y-%m-%d %H:%M")
        if chat.next_run_at
        else "Paused"
    )
    await update.effective_message.reply_text(
        (
            f"Status: {'Active' if chat.is_active else 'Paused'}\n"
            f"Top k: {chat.top_k}\n"
            f"Timezone: {chat.timezone}\n"
            f"Cadence: {format_schedule(chat)}\n"
            f"Enabled sources: {enabled_count}\n"
            f"Repeat window: {chat.repeat_window_days} day(s)\n"
            f"Next scheduled run: {next_run}"
        )
    )


async def pause_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("Only chat admins can pause scheduled digests in groups.")
        return
    runtime = get_runtime(context.application)
    await runtime.db.update_chat(chat.chat_id, is_active=0, next_run_at=None)
    await update.effective_message.reply_text("Scheduled AI Safety Brief deliveries are paused.")


async def resume_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("Only chat admins can resume scheduled digests in groups.")
        return
    runtime = get_runtime(context.application)
    chat.is_active = True
    next_run = compute_next_run(chat)
    await runtime.db.update_chat(
        chat.chat_id,
        is_active=1,
        next_run_at=next_run.isoformat(),
    )
    local_next = next_run.astimezone(ZoneInfo(chat.timezone)).strftime("%Y-%m-%d %H:%M")
    await update.effective_message.reply_text(f"Scheduled deliveries resumed. Next run: {local_next}.")


async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    rows = await runtime.db.list_recent_runs(chat.chat_id)
    if not rows:
        await update.effective_message.reply_text("No digest history yet.")
        return
    lines = ["Recent runs:"]
    for row in rows:
        lines.append(
            f"- {row['created_at']}: {row['status']} ({row['item_count']} item(s), via {row['triggered_by']})"
        )
    await update.effective_message.reply_text("\n".join(lines))


async def sources_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    disabled = set(chat.disabled_sources)
    lines = ["Configured sources:"]
    for source in runtime.collector.list_sources():
        status = "enabled" if source.key not in disabled else "disabled"
        lines.append(f"- {source.key}: {source.name} ({status})")
    await update.effective_message.reply_text("\n".join(lines))


async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    if not context.args:
        await status_handler(update, context)
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("Only chat admins can change persistent settings in groups.")
        return

    subcommand = context.args[0].lower()
    try:
        if subcommand == "k" and len(context.args) >= 2:
            top_k = int(context.args[1])
            if not 1 <= top_k <= 10:
                raise ValueError("k must be between 1 and 10.")
            chat.top_k = top_k
            await runtime.db.update_chat(chat.chat_id, top_k=top_k)
            await update.effective_message.reply_text(f"Top k updated to {top_k}.")
            return

        if subcommand == "timezone" and len(context.args) >= 2:
            ZoneInfo(context.args[1])
            chat.timezone = context.args[1]
            chat.next_run_at = compute_next_run(chat)
            await runtime.db.update_chat(
                chat.chat_id,
                timezone=chat.timezone,
                next_run_at=chat.next_run_at.isoformat(),
            )
            await update.effective_message.reply_text(f"Timezone updated to {chat.timezone}.")
            return

        if subcommand == "cadence" and len(context.args) >= 3:
            cadence_type = context.args[1].lower()
            if cadence_type == "hourly":
                interval = int(context.args[2])
                if not 1 <= interval <= 24:
                    raise ValueError("Hourly cadence must be between 1 and 24 hours.")
                chat.schedule_type = "hourly"
                chat.schedule_value = str(interval)
                if len(context.args) >= 4:
                    chat.send_minute = parse_time_component(context.args[3], minute_only=True)
            elif cadence_type == "daily":
                hour, minute = parse_hhmm(context.args[2])
                chat.schedule_type = "daily"
                chat.schedule_value = ""
                chat.send_hour = hour
                chat.send_minute = minute
            elif cadence_type == "weekly" and len(context.args) >= 4:
                parse_weekdays(context.args[2])
                hour, minute = parse_hhmm(context.args[3])
                chat.schedule_type = "weekly"
                chat.schedule_value = context.args[2]
                chat.send_hour = hour
                chat.send_minute = minute
            else:
                raise ValueError("Unsupported cadence syntax.")

            chat.next_run_at = compute_next_run(chat)
            await runtime.db.update_chat(
                chat.chat_id,
                schedule_type=chat.schedule_type,
                schedule_value=chat.schedule_value,
                send_hour=chat.send_hour,
                send_minute=chat.send_minute,
                next_run_at=chat.next_run_at.isoformat(),
            )
            await update.effective_message.reply_text(f"Cadence updated: {format_schedule(chat)}.")
            return

        if subcommand == "source" and len(context.args) >= 3:
            action = context.args[1].lower()
            source_key = context.args[2]
            all_keys = {source.key for source in runtime.collector.list_sources()}
            if source_key not in all_keys:
                raise ValueError(f"Unknown source key: {source_key}")
            disabled = set(chat.disabled_sources)
            if action == "enable":
                disabled.discard(source_key)
            elif action == "disable":
                disabled.add(source_key)
            else:
                raise ValueError("Use /settings source enable <key> or disable <key>.")
            chat.disabled_sources = sorted(disabled)
            import json

            await runtime.db.update_chat(chat.chat_id, disabled_sources=json.dumps(chat.disabled_sources))
            await update.effective_message.reply_text(
                f"Source {source_key} is now {'enabled' if source_key not in disabled else 'disabled'}."
            )
            return

        if subcommand == "repeat-window" and len(context.args) >= 2:
            days = int(context.args[1])
            if days < 0 or days > 60:
                raise ValueError("Repeat window must be between 0 and 60 days.")
            chat.repeat_window_days = days
            await runtime.db.update_chat(chat.chat_id, repeat_window_days=days)
            await update.effective_message.reply_text(f"Repeat window updated to {days} day(s).")
            return

        raise ValueError("Unknown settings command.")
    except Exception as exc:
        await update.effective_message.reply_text(
            (
                f"{exc}\n\n"
                "Examples:\n"
                "/settings k 5\n"
                "/settings timezone Asia/Singapore\n"
                "/settings cadence daily 19:00\n"
                "/settings cadence hourly 6\n"
                "/settings cadence weekly Mon,Wed,Fri 18:30\n"
                "/settings source enable x_openai\n"
                "/settings source disable lesswrong\n"
                "/settings repeat-window 7"
            )
        )


def parse_hhmm(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError("Time must use HH:MM format.")
    hour_str, minute_str = value.split(":", 1)
    hour = int(hour_str)
    minute = int(minute_str)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Time must be a valid 24-hour clock value.")
    return hour, minute


def parse_time_component(value: str, minute_only: bool = False) -> int:
    parsed = int(value)
    if minute_only and not 0 <= parsed <= 59:
        raise ValueError("Minute value must be between 0 and 59.")
    return parsed
