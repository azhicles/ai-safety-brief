from ai_safety_brief.bot.ui import (
    build_alert_keyboard,
    build_channel_picker_keyboard,
    build_digest_keyboard,
    build_settings_keyboard,
    build_sources_keyboard,
)
from ai_safety_brief.models import ChatSettings, SourceDefinition, StoredItem


def test_settings_keyboard_exposes_control_panel_actions():
    keyboard = build_settings_keyboard().inline_keyboard
    callback_data = {button.callback_data for row in keyboard for button in row}
    assert "settings:topics" in callback_data
    assert "settings:mix" in callback_data
    assert "settings:alerts" in callback_data
    assert "settings:quiet_hours" in callback_data
    assert "settings:sources:0" in callback_data


def test_settings_keyboard_can_be_scoped_to_a_remote_channel():
    keyboard = build_settings_keyboard(target_chat_id=-100123).inline_keyboard
    callback_data = {button.callback_data for row in keyboard for button in row}
    assert "chat:-100123:settings:topics" in callback_data
    assert "chat:-100123:settings:mix" in callback_data
    assert "chat:-100123:settings:sources:0" in callback_data


def test_digest_and_alert_keyboards_expose_expected_actions():
    digest_callbacks = {
        button.callback_data
        for row in build_digest_keyboard().inline_keyboard
        for button in row
    }
    assert digest_callbacks == {"digest:more", "digest:why", "settings:panel"}

    item = StoredItem(
        id=42,
        source_key="anthropic_news",
        source_name="Anthropic News",
        title="Project Glasswing",
        canonical_url="https://example.org/project-glasswing",
        content_type="news",
    )
    alert_callbacks = {
        button.callback_data
        for row in build_alert_keyboard(item).inline_keyboard
        for button in row
    }
    assert f"item:more_like_this:1:{item.id}" in alert_callbacks
    assert f"item:less_like_this:1:{item.id}" in alert_callbacks
    assert f"item:why:1:{item.id}" in alert_callbacks


def test_sources_keyboard_toggles_sources_and_pages():
    sources = [
        SourceDefinition(key=f"source_{index}", name=f"Source {index}", mode="rss", url=f"https://example.org/{index}")
        for index in range(9)
    ]
    keyboard = build_sources_keyboard(sources, ChatSettings(chat_id=1), page=0).inline_keyboard
    callback_data = [button.callback_data for row in keyboard for button in row]
    assert "settings:source_toggle:source_0:0" in callback_data
    assert "settings:sources:1" in callback_data


def test_channel_picker_keyboard_exposes_channel_targets():
    chats = [
        ChatSettings(chat_id=-1001, chat_type="channel", chat_title="alpha"),
        ChatSettings(chat_id=-1002, chat_type="channel", chat_title="beta"),
    ]
    keyboard = build_channel_picker_keyboard(chats).inline_keyboard
    callback_data = [button.callback_data for row in keyboard for button in row]
    assert "channel:open:-1001" in callback_data
    assert "channel:open:-1002" in callback_data
