from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database
from ai_safety_brief.models import CandidateItem, ChatSettings, SourceDefinition
from ai_safety_brief.services.digest import DigestPipeline
from ai_safety_brief.services.llm_refiner import GroqRefiner
from ai_safety_brief.services.summarizer import Summarizer


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        telegram_bot_token="",
        groq_api_key="",
        groq_model="llama-3.3-70b-versatile",
        data_dir=tmp_path,
        db_path=tmp_path / "bot.db",
        default_timezone="Asia/Singapore",
        default_top_k=2,
        default_send_hour=19,
        default_send_minute=0,
        default_repeat_window_days=7,
        lookback_hours=72,
        scheduler_poll_seconds=60,
        item_fetch_timeout_seconds=20,
        x_rss_base_url="",
        x_accounts=(),
    )


class FakeCollector:
    def __init__(self) -> None:
        self._sources = [
            SourceDefinition(
                key="anthropic_news",
                name="Anthropic News",
                mode="listing",
                url="https://www.anthropic.com/news",
                authority_weight=1.35,
                content_type="news",
            ),
            SourceDefinition(
                key="metr_blog",
                name="METR",
                mode="listing",
                url="https://metr.org/blog",
                authority_weight=1.3,
                content_type="news",
            ),
            SourceDefinition(
                key="arxiv_alignment",
                name="arXiv Alignment",
                mode="arxiv",
                url="https://export.arxiv.org/api/query",
                authority_weight=1.3,
                content_type="paper",
            ),
        ]

    def list_sources(self):
        return list(self._sources)

    def enabled_sources(self, chat: ChatSettings):
        return list(self._sources)

    async def collect_recent(self, chat: ChatSettings):
        return [
            CandidateItem(
                source_key="anthropic_news",
                source_name="Anthropic News",
                title="Project Glasswing",
                canonical_url="https://www.anthropic.com/news/project-glasswing",
                content_type="news",
                excerpt="A new initiative to secure the world's most critical software.",
                raw_text="Anthropic announced a new initiative focused on securing critical software and improving frontier AI safeguards.",
                published_at=datetime(2026, 4, 8, 11, 0, tzinfo=timezone.utc),
            ),
            CandidateItem(
                source_key="metr_blog",
                source_name="METR",
                title="Oversight benchmark for autonomous coding agents",
                canonical_url="https://metr.org/post/oversight",
                content_type="news",
                excerpt="A benchmark for sabotage detection and scalable oversight.",
                raw_text="A benchmark for sabotage detection and scalable oversight in autonomous coding agents.",
                published_at=datetime(2026, 4, 8, 10, 0, tzinfo=timezone.utc),
            ),
            CandidateItem(
                source_key="arxiv_alignment",
                source_name="arXiv Alignment",
                title="Interpretability methods for deceptive alignment detection",
                canonical_url="https://arxiv.org/abs/2604.12345",
                content_type="paper",
                excerpt="The paper studies interpretability and deception signals.",
                raw_text="The paper studies interpretability and deception signals in frontier language models.",
                published_at=datetime(2026, 4, 8, 6, 0, tzinfo=timezone.utc),
            ),
        ]

    async def enrich_candidates(self, candidates):
        return candidates


@pytest.mark.asyncio
async def test_pipeline_generates_digest_and_respects_seen_window(tmp_path: Path):
    settings = make_settings(tmp_path)
    db = Database(settings.db_path)
    await db.init()
    collector = FakeCollector()
    pipeline = DigestPipeline(settings, db, collector, Summarizer(), GroqRefiner("", settings.groq_model))
    chat = ChatSettings(chat_id=123, timezone="UTC", top_k=2)

    first = await pipeline.generate_digest(chat, triggered_by="manual")
    assert len(first.entries) == 2
    assert "AI Safety Brief" in first.messages[0]
    assert "1. Project Glasswing 🚨" in first.messages[0]
    assert "\nsummary: " not in first.messages[0]
    assert "\nwhy it matters: " not in first.messages[0]
    assert "\nfrom: " not in first.messages[0]
    assert "Anthropic News | https://www.anthropic.com/news/project-glasswing" in first.messages[0]
    assert "\n\nAnthropic News | https://www.anthropic.com/news/project-glasswing" in first.messages[0]
    assert "📰 News" not in first.messages[0]
    assert "\n   " not in first.messages[0]

    second = await pipeline.generate_digest(chat, triggered_by="manual")
    assert len(second.entries) == 1
    assert second.entries[0].item.content_type == "paper"

    third = await pipeline.generate_digest(chat, triggered_by="manual")
    assert third.entries == []
