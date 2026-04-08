"""Telegram UI helpers."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from ai_safety_brief.models import ChatSettings, SourceDefinition, StoredItem
from ai_safety_brief.personalization import (
    ALERT_MODES,
    CONTENT_MIXES,
    FOCUS_TOPICS,
    QUIET_HOUR_PRESETS,
    format_quiet_hours,
    topic_label,
)


def build_settings_summary(chat: ChatSettings, enabled_count: int) -> str:
    topics = ", ".join(topic_label(topic) for topic in chat.focus_topics) if chat.focus_topics else "all topics"
    return (
        "current settings:\n"
        f"- status: {'active' if chat.is_active else 'paused'}\n"
        f"- top k: {chat.top_k}\n"
        f"- timezone: {chat.timezone}\n"
        f"- content mix: {chat.content_mix}\n"
        f"- focus topics: {topics}\n"
        f"- alerts: {chat.alert_mode}\n"
        f"- quiet hours: {format_quiet_hours(chat)}\n"
        f"- enabled sources: {enabled_count}\n"
        f"- repeat window: {chat.repeat_window_days} day(s)"
    )


def build_settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("topics", callback_data="settings:topics"),
                InlineKeyboardButton("content mix", callback_data="settings:mix"),
            ],
            [
                InlineKeyboardButton("alerts", callback_data="settings:alerts"),
                InlineKeyboardButton("quiet hours", callback_data="settings:quiet_hours"),
            ],
            [
                InlineKeyboardButton("sources", callback_data="settings:sources:0"),
                InlineKeyboardButton("why these picks", callback_data="digest:why"),
            ],
        ]
    )


def build_digest_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("more results", callback_data="digest:more"),
                InlineKeyboardButton("why these picks", callback_data="digest:why"),
            ],
            [InlineKeyboardButton("tune brief", callback_data="settings:panel")],
        ]
    )


def build_alert_keyboard(item: StoredItem) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("more like this", callback_data=f"item:more_like_this:1:{item.id}"),
                InlineKeyboardButton("less like this", callback_data=f"item:less_like_this:1:{item.id}"),
            ],
            [
                InlineKeyboardButton("why picked", callback_data=f"item:why:1:{item.id}"),
                InlineKeyboardButton("tune alerts", callback_data="settings:alerts"),
            ],
        ]
    )


def build_topics_keyboard(chat: ChatSettings) -> InlineKeyboardMarkup:
    rows = []
    for index in range(0, len(FOCUS_TOPICS), 2):
        row = []
        for topic in FOCUS_TOPICS[index : index + 2]:
            selected = "✓ " if topic in chat.focus_topics else ""
            row.append(
                InlineKeyboardButton(
                    f"{selected}{topic_label(topic)}",
                    callback_data=f"settings:topic_toggle:{topic}",
                )
            )
        rows.append(row)
    rows.append([InlineKeyboardButton("back", callback_data="settings:panel")])
    return InlineKeyboardMarkup(rows)


def build_mix_keyboard(chat: ChatSettings) -> InlineKeyboardMarkup:
    rows = []
    for mix in CONTENT_MIXES:
        label = f"✓ {mix}" if mix == chat.content_mix else mix
        rows.append([InlineKeyboardButton(label, callback_data=f"settings:mix_set:{mix}")])
    rows.append([InlineKeyboardButton("back", callback_data="settings:panel")])
    return InlineKeyboardMarkup(rows)


def build_alerts_keyboard(chat: ChatSettings) -> InlineKeyboardMarkup:
    rows = []
    for mode in ALERT_MODES:
        label = f"✓ {mode}" if mode == chat.alert_mode else mode
        rows.append([InlineKeyboardButton(label, callback_data=f"settings:alerts_set:{mode}")])
    rows.append([InlineKeyboardButton("back", callback_data="settings:panel")])
    return InlineKeyboardMarkup(rows)


def build_quiet_hours_keyboard(chat: ChatSettings) -> InlineKeyboardMarkup:
    current = format_quiet_hours(chat)
    rows = []
    for preset in QUIET_HOUR_PRESETS:
        label = f"✓ {preset}" if preset == current else preset
        rows.append([InlineKeyboardButton(label, callback_data=f"settings:quiet_set:{preset}")])
    rows.append([InlineKeyboardButton("back", callback_data="settings:panel")])
    return InlineKeyboardMarkup(rows)


def build_sources_keyboard(
    sources: list[SourceDefinition],
    chat: ChatSettings,
    page: int = 0,
    page_size: int = 8,
) -> InlineKeyboardMarkup:
    start = page * page_size
    slice_sources = sources[start : start + page_size]
    rows = []
    disabled = set(chat.disabled_sources)
    for source in slice_sources:
        enabled = source.key not in disabled
        label = f"{'✓' if enabled else '✕'} {source.name}"
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=f"settings:source_toggle:{source.key}:{page}",
                )
            ]
        )
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("prev", callback_data=f"settings:sources:{page - 1}"))
    if start + page_size < len(sources):
        nav.append(InlineKeyboardButton("next", callback_data=f"settings:sources:{page + 1}"))
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("back", callback_data="settings:panel")])
    return InlineKeyboardMarkup(rows)
