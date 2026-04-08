"""SQLite repository layer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from ai_safety_brief.models import CandidateItem, ChatSettings, DigestEntry, StoredItem


def _to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def connect(self) -> aiosqlite.Connection:
        return aiosqlite.connect(self.path)

    async def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
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

                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    canonical_url TEXT NOT NULL UNIQUE,
                    author TEXT NOT NULL DEFAULT '',
                    published_at TEXT,
                    content_type TEXT NOT NULL DEFAULT 'news',
                    excerpt TEXT NOT NULL DEFAULT '',
                    raw_text TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    why_it_matters TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    discovered_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_seen_at TEXT
                );

                CREATE TABLE IF NOT EXISTS digest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    triggered_by TEXT NOT NULL DEFAULT 'schedule',
                    status TEXT NOT NULL DEFAULT 'success',
                    error TEXT NOT NULL DEFAULT '',
                    item_count INTEGER NOT NULL DEFAULT 0,
                    message_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id)
                );

                CREATE TABLE IF NOT EXISTS digest_run_items (
                    digest_run_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    section TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (digest_run_id, item_id),
                    FOREIGN KEY (digest_run_id) REFERENCES digest_runs(id),
                    FOREIGN KEY (item_id) REFERENCES items(id)
                );

                CREATE TABLE IF NOT EXISTS chat_seen_items (
                    chat_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    seen_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (chat_id, item_id),
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id),
                    FOREIGN KEY (item_id) REFERENCES items(id)
                );

                CREATE TABLE IF NOT EXISTS chat_alert_items (
                    chat_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    alerted_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (chat_id, item_id),
                    FOREIGN KEY (chat_id) REFERENCES chats(chat_id),
                    FOREIGN KEY (item_id) REFERENCES items(id)
                );
                """
            )
            await self._ensure_chat_column(db, "focus_topics", "TEXT NOT NULL DEFAULT '[]'")
            await self._ensure_chat_column(db, "content_mix", "TEXT NOT NULL DEFAULT 'balanced'")
            await self._ensure_chat_column(db, "alert_mode", "TEXT NOT NULL DEFAULT 'off'")
            await self._ensure_chat_column(db, "quiet_hours_start", "TEXT DEFAULT NULL")
            await self._ensure_chat_column(db, "quiet_hours_end", "TEXT DEFAULT NULL")
            await db.commit()

    async def upsert_chat(
        self,
        chat_id: int,
        chat_type: str,
        chat_title: str,
        defaults: dict[str, Any],
    ) -> ChatSettings:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO chats (
                    chat_id, chat_type, chat_title, timezone, top_k, schedule_type,
                    schedule_value, send_hour, send_minute, disabled_sources, focus_topics,
                    content_mix, alert_mode, quiet_hours_start, quiet_hours_end,
                    repeat_window_days, next_run_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    chat_type=excluded.chat_type,
                    chat_title=excluded.chat_title,
                    updated_at=datetime('now')
                """,
                (
                    chat_id,
                    chat_type,
                    chat_title or "",
                    defaults["timezone"],
                    defaults["top_k"],
                    defaults["schedule_type"],
                    defaults["schedule_value"],
                    defaults["send_hour"],
                    defaults["send_minute"],
                    defaults["disabled_sources"],
                    defaults["focus_topics"],
                    defaults["content_mix"],
                    defaults["alert_mode"],
                    defaults["quiet_hours_start"],
                    defaults["quiet_hours_end"],
                    defaults["repeat_window_days"],
                    defaults["next_run_at"],
                ),
            )
            await db.commit()
        chat = await self.get_chat(chat_id)
        assert chat is not None
        return chat

    async def get_chat(self, chat_id: int) -> ChatSettings | None:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            row = await self._fetchone(db, "SELECT * FROM chats WHERE chat_id = ?", (chat_id,))
        if not row:
            return None
        return self._row_to_chat(row)

    async def update_chat(self, chat_id: int, **fields: Any) -> None:
        if not fields:
            return
        updates = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [chat_id]
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                f"UPDATE chats SET {updates}, updated_at = datetime('now') WHERE chat_id = ?",
                values,
            )
            await db.commit()

    async def list_due_chats(self, now_utc: datetime) -> list[ChatSettings]:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT * FROM chats
                WHERE is_active = 1
                  AND next_run_at IS NOT NULL
                  AND next_run_at <= ?
                ORDER BY next_run_at ASC
                """,
                (_to_iso(now_utc),),
            )
        return [self._row_to_chat(row) for row in rows]

    async def list_alert_enabled_chats(self) -> list[ChatSettings]:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT * FROM chats
                WHERE is_active = 1
                  AND alert_mode != 'off'
                ORDER BY chat_id ASC
                """,
                (),
            )
        return [self._row_to_chat(row) for row in rows]

    async def list_recent_runs(self, chat_id: int, limit: int = 5) -> list[aiosqlite.Row]:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            return await self._fetchall(
                db,
                """
                SELECT * FROM digest_runs
                WHERE chat_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )

    async def save_item(self, candidate: CandidateItem) -> StoredItem:
        metadata_json = json.dumps(candidate.metadata, sort_keys=True)
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO items (
                    source_key, source_name, title, canonical_url, author, published_at,
                    content_type, excerpt, raw_text, summary, why_it_matters, metadata, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_url) DO UPDATE SET
                    source_key = excluded.source_key,
                    source_name = excluded.source_name,
                    title = excluded.title,
                    author = excluded.author,
                    published_at = COALESCE(excluded.published_at, items.published_at),
                    content_type = excluded.content_type,
                    excerpt = CASE WHEN excluded.excerpt != '' THEN excluded.excerpt ELSE items.excerpt END,
                    raw_text = CASE WHEN excluded.raw_text != '' THEN excluded.raw_text ELSE items.raw_text END,
                    summary = CASE WHEN excluded.summary != '' THEN excluded.summary ELSE items.summary END,
                    why_it_matters = CASE WHEN excluded.why_it_matters != '' THEN excluded.why_it_matters ELSE items.why_it_matters END,
                    metadata = excluded.metadata
                """,
                (
                    candidate.source_key,
                    candidate.source_name,
                    candidate.title,
                    candidate.canonical_url,
                    candidate.author,
                    _to_iso(candidate.published_at),
                    candidate.content_type,
                    candidate.excerpt,
                    candidate.raw_text,
                    candidate.summary,
                    candidate.why_it_matters,
                    metadata_json,
                    _to_iso(candidate.published_at),
                ),
            )
            await db.commit()
            row = await self._fetchone(
                db,
                "SELECT * FROM items WHERE canonical_url = ?",
                (candidate.canonical_url,),
            )
        assert row is not None
        return self._row_to_item(row)

    async def get_seen_item_ids(self, chat_id: int, within_days: int) -> set[int]:
        since = utc_midnight_cutoff(within_days)
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT item_id FROM chat_seen_items
                WHERE chat_id = ? AND seen_at >= ?
                """,
                (chat_id, _to_iso(since)),
            )
        return {int(row["item_id"]) for row in rows}

    async def mark_seen(self, chat_id: int, item_ids: list[int], seen_at: datetime) -> None:
        if not item_ids:
            return
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            for item_id in item_ids:
                await db.execute(
                    """
                    INSERT INTO chat_seen_items (chat_id, item_id, seen_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chat_id, item_id) DO UPDATE SET seen_at = excluded.seen_at
                    """,
                    (chat_id, item_id, _to_iso(seen_at)),
                )
            await db.commit()

    async def get_alerted_item_ids(self, chat_id: int) -> set[int]:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT item_id FROM chat_alert_items
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
        return {int(row["item_id"]) for row in rows}

    async def mark_alerted(self, chat_id: int, item_id: int, alerted_at: datetime) -> None:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO chat_alert_items (chat_id, item_id, alerted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(chat_id, item_id) DO UPDATE SET alerted_at = excluded.alerted_at
                """,
                (chat_id, item_id, _to_iso(alerted_at)),
            )
            await db.commit()

    async def count_alerts_since(self, chat_id: int, since: datetime) -> int:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            row = await self._fetchone(
                db,
                """
                SELECT COUNT(*) AS count
                FROM chat_alert_items
                WHERE chat_id = ? AND alerted_at >= ?
                """,
                (chat_id, _to_iso(since)),
            )
        return int(row["count"]) if row else 0

    async def latest_alert_at(self, chat_id: int) -> datetime | None:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            row = await self._fetchone(
                db,
                """
                SELECT MAX(alerted_at) AS alerted_at
                FROM chat_alert_items
                WHERE chat_id = ?
                """,
                (chat_id,),
            )
        return _from_iso(row["alerted_at"]) if row and row["alerted_at"] else None

    async def save_digest_run(
        self,
        chat_id: int,
        entries: list[DigestEntry],
        messages: list[str],
        triggered_by: str,
        status: str = "success",
        error: str = "",
    ) -> int:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                INSERT INTO digest_runs (chat_id, triggered_by, status, error, item_count, message_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    triggered_by,
                    status,
                    error,
                    len(entries),
                    "\n\n---\n\n".join(messages),
                    _to_iso(datetime.now(timezone.utc)),
                ),
            )
            run_id = cursor.lastrowid
            for entry in entries:
                await db.execute(
                    """
                    INSERT INTO digest_run_items (digest_run_id, item_id, rank, score, section)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, entry.item.id, entry.rank, entry.score, entry.section),
                )
            await db.commit()
        return int(run_id)

    async def list_sources_state(self, chat_id: int) -> list[str]:
        chat = await self.get_chat(chat_id)
        return chat.disabled_sources if chat else []

    async def get_item(self, item_id: int) -> StoredItem | None:
        return await self._get_item_by_id(item_id)

    async def latest_run_row(self, chat_id: int) -> aiosqlite.Row | None:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            return await self._fetchone(
                db,
                """
                SELECT * FROM digest_runs
                WHERE chat_id = ? AND status = 'success' AND item_count > 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (chat_id,),
            )

    async def latest_run_item_ids(self, chat_id: int) -> list[int]:
        row = await self.latest_run_row(chat_id)
        if not row:
            return []
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT item_id FROM digest_run_items
                WHERE digest_run_id = ?
                ORDER BY rank ASC
                """,
                (int(row["id"]),),
            )
        return [int(entry["item_id"]) for entry in rows]

    async def latest_run_item_by_rank(self, chat_id: int, rank: int) -> StoredItem | None:
        row = await self.latest_run_row(chat_id)
        if not row:
            return None
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            item_row = await self._fetchone(
                db,
                """
                SELECT items.*
                FROM digest_run_items
                JOIN items ON items.id = digest_run_items.item_id
                WHERE digest_run_items.digest_run_id = ? AND digest_run_items.rank = ?
                """,
                (int(row["id"]), rank),
            )
        return self._row_to_item(item_row) if item_row else None

    async def latest_run_entries(self, chat_id: int) -> list[DigestEntry]:
        row = await self.latest_run_row(chat_id)
        if not row:
            return []
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            rows = await self._fetchall(
                db,
                """
                SELECT digest_run_items.rank, digest_run_items.score, digest_run_items.section, items.*
                FROM digest_run_items
                JOIN items ON items.id = digest_run_items.item_id
                WHERE digest_run_items.digest_run_id = ?
                ORDER BY digest_run_items.rank ASC
                """,
                (int(row["id"]),),
            )
        return [
            DigestEntry(
                item=self._row_to_item(entry),
                rank=int(entry["rank"]),
                section=entry["section"],
                score=float(entry["score"]),
            )
            for entry in rows
        ]

    async def _get_item_by_id(self, item_id: int) -> StoredItem | None:
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            row = await self._fetchone(db, "SELECT * FROM items WHERE id = ?", (item_id,))
        return self._row_to_item(row) if row else None

    async def _fetchone(
        self,
        db: aiosqlite.Connection,
        query: str,
        params: tuple[Any, ...],
    ) -> aiosqlite.Row | None:
        cursor = await db.execute(query, params)
        return await cursor.fetchone()

    async def _fetchall(
        self,
        db: aiosqlite.Connection,
        query: str,
        params: tuple[Any, ...],
    ) -> list[aiosqlite.Row]:
        cursor = await db.execute(query, params)
        return await cursor.fetchall()

    def _row_to_chat(self, row: aiosqlite.Row) -> ChatSettings:
        return ChatSettings(
            chat_id=int(row["chat_id"]),
            chat_type=row["chat_type"],
            chat_title=row["chat_title"],
            is_active=bool(row["is_active"]),
            timezone=row["timezone"],
            top_k=int(row["top_k"]),
            schedule_type=row["schedule_type"],
            schedule_value=row["schedule_value"],
            send_hour=int(row["send_hour"]),
            send_minute=int(row["send_minute"]),
            disabled_sources=json.loads(row["disabled_sources"]),
            focus_topics=json.loads(row["focus_topics"] or "[]"),
            content_mix=row["content_mix"],
            alert_mode=row["alert_mode"],
            quiet_hours_start=row["quiet_hours_start"],
            quiet_hours_end=row["quiet_hours_end"],
            repeat_window_days=int(row["repeat_window_days"]),
            next_run_at=_from_iso(row["next_run_at"]),
            last_digest_at=_from_iso(row["last_digest_at"]),
            created_at=_from_iso(row["created_at"]),
        )

    def _row_to_item(self, row: aiosqlite.Row) -> StoredItem:
        return StoredItem(
            id=int(row["id"]),
            source_key=row["source_key"],
            source_name=row["source_name"],
            title=row["title"],
            canonical_url=row["canonical_url"],
            author=row["author"],
            published_at=_from_iso(row["published_at"]),
            content_type=row["content_type"],
            excerpt=row["excerpt"],
            raw_text=row["raw_text"],
            summary=row["summary"],
            why_it_matters=row["why_it_matters"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def _ensure_chat_column(self, db: aiosqlite.Connection, name: str, ddl: str) -> None:
        rows = await self._fetchall(db, "PRAGMA table_info(chats)", ())
        if any(row["name"] == name for row in rows):
            return
        await db.execute(f"ALTER TABLE chats ADD COLUMN {name} {ddl}")


def utc_midnight_cutoff(within_days: int) -> datetime:
    now = datetime.now(timezone.utc)
    return now - timedelta(days=max(0, within_days))
