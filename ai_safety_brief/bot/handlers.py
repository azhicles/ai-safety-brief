"""Telegram command and callback handlers."""

from __future__ import annotations

import json
import logging
from zoneinfo import ZoneInfo

from telegram import CallbackQuery, Update
from telegram.ext import Application, ContextTypes

from ai_safety_brief.bot.runtime import Runtime
from ai_safety_brief.bot.ui import (
    build_alert_keyboard,
    build_alerts_keyboard,
    build_digest_keyboard,
    build_mix_keyboard,
    build_quiet_hours_keyboard,
    build_settings_keyboard,
    build_settings_summary,
    build_sources_keyboard,
    build_topics_keyboard,
)
from ai_safety_brief.models import ChatSettings, StoredItem
from ai_safety_brief.personalization import (
    coerce_alert_mode,
    coerce_content_mix,
    format_quiet_hours,
    parse_quiet_hours,
    parse_topics_csv,
    topic_label,
)
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
        await update.effective_message.reply_text("use /start first to register this chat.")
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
        focus_topics=[],
        content_mix="balanced",
        alert_mode="off",
        quiet_hours_start=None,
        quiet_hours_end=None,
    )
    next_run = compute_next_run(base)
    return {
        "timezone": base.timezone,
        "top_k": base.top_k,
        "schedule_type": base.schedule_type,
        "schedule_value": base.schedule_value,
        "send_hour": base.send_hour,
        "send_minute": base.send_minute,
        "disabled_sources": json.dumps(base.disabled_sources),
        "focus_topics": json.dumps(base.focus_topics),
        "content_mix": base.content_mix,
        "alert_mode": base.alert_mode,
        "quiet_hours_start": base.quiet_hours_start,
        "quiet_hours_end": base.quiet_hours_end,
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
            "ai safety brief is ready.\n\n"
            "commands:\n"
            "/brief - generate an immediate digest\n"
            "/more - show more results below the cutoff\n"
            "/why 2 - explain why an item was picked\n"
            "/status - current schedule and settings\n"
            "/settings - open the control panel\n"
            "/pause and /resume - stop or restart scheduled sends\n"
            "/history - recent digest runs\n\n"
            f"current schedule: {format_schedule(chat)} ({chat.timezone})"
        )
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    await update.effective_message.reply_text(
        (
            "ai safety brief commands:\n"
            "/start\n"
            "/brief\n"
            "/more\n"
            "/why 2\n"
            "/status\n"
            "/history\n"
            "/sources\n"
            "/settings\n"
            "/settings k 5\n"
            "/settings topics alignment,evals,security\n"
            "/settings mix news-heavy\n"
            "/settings alerts moderate\n"
            "/settings quiet-hours 23:00-07:00\n"
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
        await update.effective_message.reply_text("a digest run is already in progress for this chat.")
        return
    await update.effective_message.reply_text("building the latest ai safety brief...")
    async with lock:
        result = await runtime.pipeline.generate_digest(chat, triggered_by="manual")
    await _reply_messages(update.effective_message.reply_text, result.messages, build_digest_keyboard())


async def more_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    lock = runtime.chat_lock(chat.chat_id)
    if lock.locked():
        await update.effective_message.reply_text("a digest run is already in progress for this chat.")
        return
    async with lock:
        result = await runtime.pipeline.generate_more(chat)
    await _reply_messages(update.effective_message.reply_text, result.messages, build_digest_keyboard())


async def why_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    if not context.args:
        await update.effective_message.reply_text("use /why <rank>, for example /why 2.")
        return
    try:
        rank = int(context.args[0])
    except ValueError:
        await update.effective_message.reply_text("use /why <rank>, for example /why 2.")
        return
    runtime = get_runtime(context.application)
    item = await runtime.db.latest_run_item_by_rank(chat.chat_id, rank)
    if not item:
        await update.effective_message.reply_text("i could not find that rank in the latest digest.")
        return
    await update.effective_message.reply_text(runtime.pipeline.explain_item(item, chat))


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
        else "paused"
    )
    topics = ", ".join(topic_label(topic) for topic in chat.focus_topics) if chat.focus_topics else "all topics"
    await update.effective_message.reply_text(
        (
            f"status: {'active' if chat.is_active else 'paused'}\n"
            f"top k: {chat.top_k}\n"
            f"timezone: {chat.timezone}\n"
            f"cadence: {format_schedule(chat)}\n"
            f"focus topics: {topics}\n"
            f"content mix: {chat.content_mix}\n"
            f"alerts: {chat.alert_mode}\n"
            f"quiet hours: {format_quiet_hours(chat)}\n"
            f"enabled sources: {enabled_count}\n"
            f"repeat window: {chat.repeat_window_days} day(s)\n"
            f"next scheduled run: {next_run}"
        ),
        reply_markup=build_settings_keyboard(),
    )


async def pause_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("only chat admins can pause scheduled digests in groups.")
        return
    runtime = get_runtime(context.application)
    await runtime.db.update_chat(chat.chat_id, is_active=0, next_run_at=None)
    await update.effective_message.reply_text("scheduled ai safety brief deliveries are paused.")


async def resume_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("only chat admins can resume scheduled digests in groups.")
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
    await update.effective_message.reply_text(f"scheduled deliveries resumed. next run: {local_next}.")


async def history_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    rows = await runtime.db.list_recent_runs(chat.chat_id)
    if not rows:
        await update.effective_message.reply_text("no digest history yet.")
        return
    lines = ["recent runs:"]
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
    lines = ["configured sources:"]
    for source in runtime.collector.list_sources():
        status = "enabled" if source.key not in disabled else "disabled"
        lines.append(f"- {source.key}: {source.name} ({status})")
    await update.effective_message.reply_text(
        "\n".join(lines),
        reply_markup=build_sources_keyboard(runtime.collector.list_sources(), chat),
    )


async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    chat = await require_registered(update, context)
    if not chat:
        return
    runtime = get_runtime(context.application)
    if not context.args:
        await update.effective_message.reply_text(
            build_settings_summary(chat, len(runtime.collector.enabled_sources(chat))),
            reply_markup=build_settings_keyboard(),
        )
        return
    if not await can_manage_chat(update, context):
        await update.effective_message.reply_text("only chat admins can change persistent settings in groups.")
        return

    subcommand = context.args[0].lower()
    try:
        if subcommand == "k" and len(context.args) >= 2:
            top_k = int(context.args[1])
            if not 1 <= top_k <= 15:
                raise ValueError("k must be between 1 and 15.")
            chat.top_k = top_k
            await runtime.db.update_chat(chat.chat_id, top_k=top_k)
            await update.effective_message.reply_text(f"top k updated to {top_k}.")
            return

        if subcommand == "topics" and len(context.args) >= 2:
            chat.focus_topics = parse_topics_csv(context.args[1])
            await runtime.db.update_chat(chat.chat_id, focus_topics=json.dumps(chat.focus_topics))
            await update.effective_message.reply_text(
                f"focus topics updated to {', '.join(chat.focus_topics) or 'all topics'}."
            )
            return

        if subcommand == "mix" and len(context.args) >= 2:
            chat.content_mix = coerce_content_mix(context.args[1])
            await runtime.db.update_chat(chat.chat_id, content_mix=chat.content_mix)
            await update.effective_message.reply_text(f"content mix updated to {chat.content_mix}.")
            return

        if subcommand == "alerts" and len(context.args) >= 2:
            chat.alert_mode = coerce_alert_mode(context.args[1])
            await runtime.db.update_chat(chat.chat_id, alert_mode=chat.alert_mode)
            await update.effective_message.reply_text(f"alerts updated to {chat.alert_mode}.")
            return

        if subcommand == "quiet-hours" and len(context.args) >= 2:
            start, end = parse_quiet_hours(context.args[1])
            chat.quiet_hours_start = start
            chat.quiet_hours_end = end
            await runtime.db.update_chat(
                chat.chat_id,
                quiet_hours_start=start,
                quiet_hours_end=end,
            )
            await update.effective_message.reply_text(f"quiet hours updated to {format_quiet_hours(chat)}.")
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
            await update.effective_message.reply_text(f"timezone updated to {chat.timezone}.")
            return

        if subcommand == "cadence" and len(context.args) >= 3:
            cadence_type = context.args[1].lower()
            if cadence_type == "hourly":
                interval = int(context.args[2])
                if not 1 <= interval <= 24:
                    raise ValueError("hourly cadence must be between 1 and 24 hours.")
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
                raise ValueError("unsupported cadence syntax.")

            chat.next_run_at = compute_next_run(chat)
            await runtime.db.update_chat(
                chat.chat_id,
                schedule_type=chat.schedule_type,
                schedule_value=chat.schedule_value,
                send_hour=chat.send_hour,
                send_minute=chat.send_minute,
                next_run_at=chat.next_run_at.isoformat(),
            )
            await update.effective_message.reply_text(f"cadence updated: {format_schedule(chat)}.")
            return

        if subcommand == "source" and len(context.args) >= 3:
            action = context.args[1].lower()
            source_key = context.args[2]
            all_keys = {source.key for source in runtime.collector.list_sources()}
            if source_key not in all_keys:
                raise ValueError(f"unknown source key: {source_key}")
            disabled = set(chat.disabled_sources)
            if action == "enable":
                disabled.discard(source_key)
            elif action == "disable":
                disabled.add(source_key)
            else:
                raise ValueError("use /settings source enable <key> or disable <key>.")
            chat.disabled_sources = sorted(disabled)
            await runtime.db.update_chat(chat.chat_id, disabled_sources=json.dumps(chat.disabled_sources))
            await update.effective_message.reply_text(
                f"source {source_key} is now {'enabled' if source_key not in disabled else 'disabled'}."
            )
            return

        if subcommand == "repeat-window" and len(context.args) >= 2:
            days = int(context.args[1])
            if days < 0 or days > 60:
                raise ValueError("repeat window must be between 0 and 60 days.")
            chat.repeat_window_days = days
            await runtime.db.update_chat(chat.chat_id, repeat_window_days=days)
            await update.effective_message.reply_text(f"repeat window updated to {days} day(s).")
            return

        raise ValueError("unknown settings command.")
    except Exception as exc:
        await update.effective_message.reply_text(
            (
                f"{exc}\n\n"
                "examples:\n"
                "/settings k 5\n"
                "/settings topics alignment,evals,security\n"
                "/settings mix news-heavy\n"
                "/settings alerts moderate\n"
                "/settings quiet-hours 23:00-07:00\n"
                "/settings timezone Asia/Singapore\n"
                "/settings cadence daily 19:00\n"
                "/settings source enable x_openai\n"
                "/settings repeat-window 7"
            )
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.message or not update.effective_chat:
        return
    runtime = get_runtime(context.application)
    chat = await runtime.db.get_chat(update.effective_chat.id)
    if not chat:
        await query.answer("use /start first.", show_alert=True)
        return

    data = query.data or ""
    if _callback_requires_manage(data) and not await can_manage_chat(update, context):
        await query.answer("only chat admins can change persistent settings in groups.", show_alert=True)
        return

    if data == "settings:panel":
        await query.answer()
        await query.message.reply_text(
            build_settings_summary(chat, len(runtime.collector.enabled_sources(chat))),
            reply_markup=build_settings_keyboard(),
        )
        return

    if data == "settings:topics":
        await query.answer()
        await query.edit_message_text("choose focus topics:", reply_markup=build_topics_keyboard(chat))
        return

    if data.startswith("settings:topic_toggle:"):
        topic = data.split(":", 2)[2]
        if topic in chat.focus_topics:
            chat.focus_topics = [existing for existing in chat.focus_topics if existing != topic]
        else:
            chat.focus_topics = sorted(set(chat.focus_topics + [topic]))
        await runtime.db.update_chat(chat.chat_id, focus_topics=json.dumps(chat.focus_topics))
        await query.answer("focus topics updated.")
        await query.edit_message_text("choose focus topics:", reply_markup=build_topics_keyboard(chat))
        return

    if data == "settings:mix":
        await query.answer()
        await query.edit_message_text("choose a content mix:", reply_markup=build_mix_keyboard(chat))
        return

    if data.startswith("settings:mix_set:"):
        chat.content_mix = coerce_content_mix(data.split(":", 2)[2])
        await runtime.db.update_chat(chat.chat_id, content_mix=chat.content_mix)
        await query.answer("content mix updated.")
        await query.edit_message_text("choose a content mix:", reply_markup=build_mix_keyboard(chat))
        return

    if data == "settings:alerts":
        await query.answer()
        await query.edit_message_text("choose an alert mode:", reply_markup=build_alerts_keyboard(chat))
        return

    if data.startswith("settings:alerts_set:"):
        chat.alert_mode = coerce_alert_mode(data.split(":", 2)[2])
        await runtime.db.update_chat(chat.chat_id, alert_mode=chat.alert_mode)
        await query.answer("alerts updated.")
        await query.edit_message_text("choose an alert mode:", reply_markup=build_alerts_keyboard(chat))
        return

    if data == "settings:quiet_hours":
        await query.answer()
        await query.edit_message_text(
            "choose quiet hours, or use /settings quiet-hours HH:MM-HH:MM for a custom window:",
            reply_markup=build_quiet_hours_keyboard(chat),
        )
        return

    if data.startswith("settings:quiet_set:"):
        raw = data.split(":", 2)[2]
        start, end = parse_quiet_hours(raw)
        chat.quiet_hours_start = start
        chat.quiet_hours_end = end
        await runtime.db.update_chat(
            chat.chat_id,
            quiet_hours_start=start,
            quiet_hours_end=end,
        )
        await query.answer("quiet hours updated.")
        await query.edit_message_text(
            "choose quiet hours, or use /settings quiet-hours HH:MM-HH:MM for a custom window:",
            reply_markup=build_quiet_hours_keyboard(chat),
        )
        return

    if data.startswith("settings:sources:"):
        page = int(data.rsplit(":", 1)[1])
        await query.answer()
        await query.edit_message_text(
            "toggle sources on or off:",
            reply_markup=build_sources_keyboard(runtime.collector.list_sources(), chat, page=page),
        )
        return

    if data.startswith("settings:source_toggle:"):
        _, _, source_key, page_text = data.split(":")
        disabled = set(chat.disabled_sources)
        if source_key in disabled:
            disabled.remove(source_key)
        else:
            disabled.add(source_key)
        chat.disabled_sources = sorted(disabled)
        await runtime.db.update_chat(chat.chat_id, disabled_sources=json.dumps(chat.disabled_sources))
        await query.answer("source settings updated.")
        await query.edit_message_text(
            "toggle sources on or off:",
            reply_markup=build_sources_keyboard(
                runtime.collector.list_sources(),
                chat,
                page=int(page_text),
            ),
        )
        return

    if data == "digest:more":
        await query.answer()
        result = await runtime.pipeline.generate_more(chat)
        await _reply_messages(query.message.reply_text, result.messages, build_digest_keyboard())
        return

    if data == "digest:why":
        await query.answer()
        entries = await runtime.db.latest_run_entries(chat.chat_id)
        await query.message.reply_text(runtime.pipeline.explain_digest_selection(chat, entries))
        return

    if data.startswith("item:why:"):
        await query.answer()
        item = await _resolve_item_from_callback(runtime, chat.chat_id, data)
        if not item:
            await query.message.reply_text("i could not find the item behind that button.")
            return
        await query.message.reply_text(runtime.pipeline.explain_item(item, chat))
        return

    if data.startswith("item:more_like_this:"):
        await query.answer()
        item = await _resolve_item_from_callback(runtime, chat.chat_id, data)
        if not item:
            await query.message.reply_text("i could not find the item behind that button.")
            return
        message = await _apply_item_feedback(runtime, chat, item, positive=True)
        await query.message.reply_text(message)
        return

    if data.startswith("item:less_like_this:"):
        await query.answer()
        item = await _resolve_item_from_callback(runtime, chat.chat_id, data)
        if not item:
            await query.message.reply_text("i could not find the item behind that button.")
            return
        message = await _apply_item_feedback(runtime, chat, item, positive=False)
        await query.message.reply_text(message)
        return

    await query.answer()


async def _apply_item_feedback(runtime: Runtime, chat: ChatSettings, item: StoredItem, positive: bool) -> str:
    strongest_topic = item.metadata.get("strongest_topic") or ""
    disabled = set(chat.disabled_sources)
    if positive:
        if not strongest_topic:
            return "i could not infer a strong topic from that item."
        chat.focus_topics = sorted(set(chat.focus_topics + [strongest_topic]))
        await runtime.db.update_chat(chat.chat_id, focus_topics=json.dumps(chat.focus_topics))
        return f"got it. i’ll lean a bit more toward {topic_label(strongest_topic)}."

    if strongest_topic and strongest_topic in chat.focus_topics:
        chat.focus_topics = [topic for topic in chat.focus_topics if topic != strongest_topic]
        await runtime.db.update_chat(chat.chat_id, focus_topics=json.dumps(chat.focus_topics))
        return f"got it. i’ll ease off {topic_label(strongest_topic)} a little."

    if item.source_key not in disabled:
        disabled.add(item.source_key)
        chat.disabled_sources = sorted(disabled)
        await runtime.db.update_chat(chat.chat_id, disabled_sources=json.dumps(chat.disabled_sources))
        return f"got it. i’ve muted {item.source_name} for this chat."
    return "that source was already muted for this chat."


async def _resolve_item_from_callback(runtime: Runtime, chat_id: int, data: str) -> StoredItem | None:
    parts = data.split(":")
    if len(parts) >= 4 and parts[3].isdigit():
        item = await runtime.db.get_item(int(parts[3]))
        if item:
            return item
    if len(parts) >= 3 and parts[2].isdigit():
        return await runtime.db.latest_run_item_by_rank(chat_id, int(parts[2]))
    return None


def _callback_requires_manage(data: str) -> bool:
    return data.startswith("settings:topic_toggle:") or data.startswith("settings:mix_set:") or data.startswith(
        "settings:alerts_set:"
    ) or data.startswith("settings:quiet_set:") or data.startswith("settings:source_toggle:") or data.startswith(
        "item:more_like_this:"
    ) or data.startswith("item:less_like_this:")


async def _reply_messages(reply_fn, messages: list[str], last_markup) -> None:
    for index, message in enumerate(messages):
        reply_markup = last_markup if index == len(messages) - 1 else None
        await reply_fn(message, reply_markup=reply_markup)


def parse_hhmm(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError("time must use HH:MM format.")
    hour_str, minute_str = value.split(":", 1)
    hour = int(hour_str)
    minute = int(minute_str)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("time must be a valid 24-hour clock value.")
    return hour, minute


def parse_time_component(value: str, minute_only: bool = False) -> int:
    parsed = int(value)
    if minute_only and not 0 <= parsed <= 59:
        raise ValueError("minute value must be between 0 and 59.")
    return parsed
