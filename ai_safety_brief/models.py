"""Shared dataclasses used across the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

ScheduleType = Literal["daily", "hourly", "weekly"]
ContentType = Literal["news", "paper", "opinion"]
SourceMode = Literal["rss", "listing", "arxiv", "x_rss"]
ContentMix = Literal["balanced", "news-heavy", "papers-heavy", "policy-heavy"]
AlertMode = Literal["off", "strict", "moderate", "broad"]


@dataclass
class ChatSettings:
    chat_id: int
    chat_type: str = "private"
    chat_title: str = ""
    is_active: bool = True
    timezone: str = "Asia/Singapore"
    top_k: int = 5
    schedule_type: ScheduleType = "daily"
    schedule_value: str = ""
    send_hour: int = 19
    send_minute: int = 0
    disabled_sources: list[str] = field(default_factory=list)
    focus_topics: list[str] = field(default_factory=list)
    content_mix: ContentMix = "balanced"
    alert_mode: AlertMode = "off"
    quiet_hours_start: str | None = None
    quiet_hours_end: str | None = None
    repeat_window_days: int = 7
    next_run_at: datetime | None = None
    last_digest_at: datetime | None = None
    created_at: datetime | None = None


@dataclass
class SourceDefinition:
    key: str
    name: str
    mode: SourceMode
    url: str
    listing_url: str | None = None
    authority_weight: float = 1.0
    content_type: ContentType = "news"
    enabled_by_default: bool = True
    recency_hours: int = 96
    query: str = ""


@dataclass
class CandidateItem:
    source_key: str
    source_name: str
    title: str
    canonical_url: str
    content_type: ContentType
    published_at: datetime | None = None
    author: str = ""
    excerpt: str = ""
    raw_text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    summary: str = ""
    why_it_matters: str = ""


@dataclass
class StoredItem:
    id: int
    source_key: str
    source_name: str
    title: str
    canonical_url: str
    content_type: ContentType
    published_at: datetime | None = None
    author: str = ""
    excerpt: str = ""
    raw_text: str = ""
    summary: str = ""
    why_it_matters: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class DigestEntry:
    item: StoredItem
    rank: int
    section: str
    score: float


@dataclass
class DigestResult:
    entries: list[DigestEntry]
    messages: list[str]
    generated_at: datetime


@dataclass
class AlertResult:
    item: StoredItem
    message: str
    generated_at: datetime
