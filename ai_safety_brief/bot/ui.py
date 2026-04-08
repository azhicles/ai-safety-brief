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


def _scope_callback(callback_data: str, target_chat_id: int | None = None) -> str:
    if target_chat_id is None:
        return callback_data
    return f"chat:{target_chat_id}:{callback_data}"


def chat_label(chat: ChatSettings) -> str:
    title = chat.chat_title.strip()
    if title:
        return title
    if chat.chat_type == "channel":
        return f"channel {chat.chat_id}"
    return str(chat.chat_id)


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


def build_settings_keyboard(target_chat_id: int | None = None) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("topics", callback_data=_scope_callback("settings:topics", target_chat_id)),
                InlineKeyboardButton("content mix", callback_data=_scope_callback("settings:mix", target_chat_id)),
            ],
            [
                InlineKeyboardButton("alerts", callback_data=_scope_callback("settings:alerts", target_chat_id)),
                InlineKeyboardButton(
                    "quiet hours",
                    callback_data=_scope_callback("settings:quiet_hours", target_chat_id),
                ),
            ],
            [
                InlineKeyboardButton("sources", callback_data=_scope_callback("settings:sources:0", target_chat_id)),
                InlineKeyboardButton("why these picks", callback_data=_scope_callback("digest:why", target_chat_id)),
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


def build_topics_keyboard(chat: ChatSettings, target_chat_id: int | None = None) -> InlineKeyboardMarkup:
    rows = []
    for index in range(0, len(FOCUS_TOPICS), 2):
        row = []
        for topic in FOCUS_TOPICS[index : index + 2]:
            selected = "✓ " if topic in chat.focus_topics else ""
            row.append(
                InlineKeyboardButton(
                    f"{selected}{topic_label(topic)}",
                    callback_data=_scope_callback(f"settings:topic_toggle:{topic}", target_chat_id),
                )
            )
        rows.append(row)
    rows.append([InlineKeyboardButton("back", callback_data=_scope_callback("settings:panel", target_chat_id))])
    return InlineKeyboardMarkup(rows)


def build_mix_keyboard(chat: ChatSettings, target_chat_id: int | None = None) -> InlineKeyboardMarkup:
    rows = []
    for mix in CONTENT_MIXES:
        label = f"✓ {mix}" if mix == chat.content_mix else mix
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=_scope_callback(f"settings:mix_set:{mix}", target_chat_id),
                )
            ]
        )
    rows.append([InlineKeyboardButton("back", callback_data=_scope_callback("settings:panel", target_chat_id))])
    return InlineKeyboardMarkup(rows)


def build_alerts_keyboard(chat: ChatSettings, target_chat_id: int | None = None) -> InlineKeyboardMarkup:
    rows = []
    for mode in ALERT_MODES:
        label = f"✓ {mode}" if mode == chat.alert_mode else mode
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=_scope_callback(f"settings:alerts_set:{mode}", target_chat_id),
                )
            ]
        )
    rows.append([InlineKeyboardButton("back", callback_data=_scope_callback("settings:panel", target_chat_id))])
    return InlineKeyboardMarkup(rows)


def build_quiet_hours_keyboard(chat: ChatSettings, target_chat_id: int | None = None) -> InlineKeyboardMarkup:
    current = format_quiet_hours(chat)
    rows = []
    for preset in QUIET_HOUR_PRESETS:
        label = f"✓ {preset}" if preset == current else preset
        rows.append(
            [
                InlineKeyboardButton(
                    label,
                    callback_data=_scope_callback(f"settings:quiet_set:{preset}", target_chat_id),
                )
            ]
        )
    rows.append([InlineKeyboardButton("back", callback_data=_scope_callback("settings:panel", target_chat_id))])
    return InlineKeyboardMarkup(rows)


def build_sources_keyboard(
    sources: list[SourceDefinition],
    chat: ChatSettings,
    page: int = 0,
    page_size: int = 8,
    target_chat_id: int | None = None,
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
                    callback_data=_scope_callback(f"settings:source_toggle:{source.key}:{page}", target_chat_id),
                )
            ]
        )
    nav = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                "prev",
                callback_data=_scope_callback(f"settings:sources:{page - 1}", target_chat_id),
            )
        )
    if start + page_size < len(sources):
        nav.append(
            InlineKeyboardButton(
                "next",
                callback_data=_scope_callback(f"settings:sources:{page + 1}", target_chat_id),
            )
        )
    if nav:
        rows.append(nav)
    rows.append([InlineKeyboardButton("back", callback_data=_scope_callback("settings:panel", target_chat_id))])
    return InlineKeyboardMarkup(rows)


def build_channel_picker_keyboard(
    chats: list[ChatSettings],
    page: int = 0,
    page_size: int = 8,
) -> InlineKeyboardMarkup:
    start = page * page_size
    slice_chats = chats[start : start + page_size]
    rows = [
        [InlineKeyboardButton(chat_label(chat), callback_data=f"channel:open:{chat.chat_id}")]
        for chat in slice_chats
    ]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("prev", callback_data=f"channel:list:{page - 1}"))
    if start + page_size < len(chats):
        nav.append(InlineKeyboardButton("next", callback_data=f"channel:list:{page + 1}"))
    if nav:
        rows.append(nav)
    return InlineKeyboardMarkup(rows)
