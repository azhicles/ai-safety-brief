import sqlite3

import pytest

from ai_safety_brief.db import Database


@pytest.mark.asyncio
async def test_database_init_adds_new_chat_columns_and_alert_table(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE chats (
            chat_id INTEGER PRIMARY KEY,
            chat_type TEXT NOT NULL DEFAULT 'private',
            chat_title TEXT NOT NULL DEFAULT '',
            is_active INTEGER NOT NULL DEFAULT 1,
            timezone TEXT NOT NULL DEFAULT 'Asia/Singapore',
            top_k INTEGER NOT NULL DEFAULT 5,
            schedule_type TEXT NOT NULL DEFAULT 'daily',
            schedule_value TEXT NOT NULL DEFAULT '',
            send_hour INTEGER NOT NULL DEFAULT 19,
            send_minute INTEGER NOT NULL DEFAULT 0,
            disabled_sources TEXT NOT NULL DEFAULT '[]',
            repeat_window_days INTEGER NOT NULL DEFAULT 7,
            next_run_at TEXT,
            last_digest_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    await db.init()
    chat = await db.upsert_chat(
        1,
        "private",
        "tester",
        defaults={
            "timezone": "Asia/Singapore",
            "top_k": 5,
            "schedule_type": "daily",
            "schedule_value": "",
            "send_hour": 19,
            "send_minute": 0,
            "disabled_sources": "[]",
            "focus_topics": "[]",
            "content_mix": "balanced",
            "alert_mode": "off",
            "quiet_hours_start": None,
            "quiet_hours_end": None,
            "repeat_window_days": 7,
            "next_run_at": None,
        },
    )
    assert chat.focus_topics == []
    assert chat.content_mix == "balanced"
    assert chat.alert_mode == "off"

    conn = sqlite3.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"focus_topics", "content_mix", "alert_mode", "quiet_hours_start", "quiet_hours_end"} <= columns
    assert "chat_alert_items" in tables
