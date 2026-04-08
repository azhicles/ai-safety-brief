from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_safety_brief.bot.handlers import _extract_target_chat_id, my_chat_member_handler
from ai_safety_brief.bot.runtime import Runtime
from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database


class DummyCollector:
    def list_sources(self):
        return []

    def enabled_sources(self, chat):
        return []


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="",
        groq_api_key="",
        groq_model="llama-3.3-70b-versatile",
        data_dir=tmp_path,
        db_path=tmp_path / "bot.db",
        default_timezone="Asia/Singapore",
        default_top_k=5,
        default_send_hour=19,
        default_send_minute=0,
        default_repeat_window_days=7,
        lookback_hours=72,
        scheduler_poll_seconds=60,
        item_fetch_timeout_seconds=20,
        x_rss_base_url="",
        x_accounts=(),
    )


def test_extract_target_chat_id_supports_scoped_callbacks():
    assert _extract_target_chat_id("settings:topics", 123) == (123, "settings:topics")
    assert _extract_target_chat_id("chat:-100987:settings:mix", 123) == (-100987, "settings:mix")


@pytest.mark.asyncio
async def test_my_chat_member_handler_registers_and_pauses_channels(tmp_path: Path):
    settings = make_settings(tmp_path)
    db = Database(settings.db_path)
    await db.init()
    runtime = Runtime(settings=settings, db=db, collector=DummyCollector(), pipeline=SimpleNamespace())
    context = SimpleNamespace(application=SimpleNamespace(bot_data={"runtime": runtime}))

    added_update = SimpleNamespace(
        my_chat_member=SimpleNamespace(
            chat=SimpleNamespace(id=-100123, type="channel", title="AI Safety Brief HQ"),
            new_chat_member=SimpleNamespace(status="administrator"),
        )
    )
    await my_chat_member_handler(added_update, context)

    chat = await db.get_chat(-100123)
    assert chat is not None
    assert chat.chat_type == "channel"
    assert chat.chat_title == "AI Safety Brief HQ"
    assert chat.is_active is True
    assert chat.next_run_at is not None

    removed_update = SimpleNamespace(
        my_chat_member=SimpleNamespace(
            chat=SimpleNamespace(id=-100123, type="channel", title="AI Safety Brief HQ"),
            new_chat_member=SimpleNamespace(status="left"),
        )
    )
    await my_chat_member_handler(removed_update, context)

    removed = await db.get_chat(-100123)
    assert removed is not None
    assert removed.is_active is False
    assert removed.next_run_at is None
