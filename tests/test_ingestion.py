from datetime import datetime, timezone
from pathlib import Path

import feedparser

from ai_safety_brief.config import Settings
from ai_safety_brief.models import SourceDefinition
from ai_safety_brief.services.ingestion import SourceCollector


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


def test_parse_listing_extracts_articles(tmp_path: Path):
    collector = SourceCollector(make_settings(tmp_path))
    html = (Path(__file__).parent / "fixtures" / "sample_listing.html").read_text()
    source = SourceDefinition(
        key="metr_blog",
        name="METR",
        mode="listing",
        url="https://metr.org/blog",
        listing_url="https://metr.org/blog",
        content_type="news",
    )
    items = collector._parse_listing(source, html, datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc))
    assert len(items) == 2
    assert items[0].canonical_url == "https://metr.org/blog/evaluating-frontier-models"


def test_feed_entries_to_candidates_filters_recent(tmp_path: Path):
    collector = SourceCollector(make_settings(tmp_path))
    xml = (Path(__file__).parent / "fixtures" / "sample_feed.xml").read_text()
    parsed = feedparser.parse(xml)
    source = SourceDefinition(
        key="alignment_forum",
        name="Alignment Forum",
        mode="rss",
        url="https://alignmentforum.org/feed.xml",
        content_type="opinion",
    )
    items = collector._feed_entries_to_candidates(
        parsed.entries,
        source,
        datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc),
    )
    assert len(items) == 2
    assert "interpretability" in items[0].title.lower()

