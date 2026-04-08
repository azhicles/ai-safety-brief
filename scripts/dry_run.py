"""Local dry run using in-memory-style fixture data."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database
from ai_safety_brief.models import CandidateItem, ChatSettings, SourceDefinition
from ai_safety_brief.services.digest import DigestPipeline
from ai_safety_brief.services.llm_refiner import GroqRefiner
from ai_safety_brief.services.summarizer import Summarizer


class DryRunCollector:
    def __init__(self) -> None:
        self._sources = [
            SourceDefinition(
                key="metr_blog",
                name="METR",
                mode="listing",
                url="https://metr.org/blog",
                authority_weight=1.3,
                content_type="news",
            ),
            SourceDefinition(
                key="govai_blog",
                name="GovAI",
                mode="listing",
                url="https://www.governance.ai/blog",
                authority_weight=1.2,
                content_type="opinion",
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
                source_key="metr_blog",
                source_name="METR",
                title="Evaluating sabotage risk in autonomous coding agents",
                canonical_url="https://metr.org/blog/sabotage-risk",
                content_type="news",
                excerpt="A new benchmark measures sabotage risk, oversight failures, and deployment readiness.",
                raw_text="A new benchmark measures sabotage risk, oversight failures, and deployment readiness for autonomous coding agents.",
                published_at=datetime(2026, 4, 8, 8, 0, tzinfo=timezone.utc),
            ),
            CandidateItem(
                source_key="govai_blog",
                source_name="GovAI",
                title="A governance roadmap for frontier deployment thresholds",
                canonical_url="https://www.governance.ai/blog/deployment-thresholds",
                content_type="opinion",
                excerpt="The roadmap lays out deployment gates, external audits, and reporting norms.",
                raw_text="The roadmap lays out deployment gates, external audits, and reporting norms for frontier AI systems.",
                published_at=datetime(2026, 4, 8, 5, 0, tzinfo=timezone.utc),
            ),
            CandidateItem(
                source_key="arxiv_alignment",
                source_name="arXiv Alignment",
                title="Interpretability methods for deceptive alignment detection",
                canonical_url="https://arxiv.org/abs/2604.12345",
                content_type="paper",
                excerpt="The paper studies interpretability and deception signals.",
                raw_text="The paper studies interpretability and deception signals in frontier language models.",
                published_at=datetime(2026, 4, 8, 2, 0, tzinfo=timezone.utc),
            ),
        ]

    async def enrich_candidates(self, candidates):
        return candidates


async def main() -> None:
    tmp_dir = Path("data")
    db_path = tmp_dir / "dry_run.db"
    if db_path.exists():
        db_path.unlink()
    settings = Settings(
        telegram_bot_token="",
        groq_api_key="",
        groq_model="llama-3.3-70b-versatile",
        data_dir=tmp_dir,
        db_path=db_path,
        default_timezone="Asia/Singapore",
        default_top_k=3,
        default_send_hour=19,
        default_send_minute=0,
        default_repeat_window_days=7,
        lookback_hours=72,
        scheduler_poll_seconds=60,
        item_fetch_timeout_seconds=20,
        x_rss_base_url="",
        x_accounts=(),
    )
    db = Database(settings.db_path)
    await db.init()
    pipeline = DigestPipeline(settings, db, DryRunCollector(), Summarizer(), GroqRefiner("", settings.groq_model))
    chat = ChatSettings(chat_id=1, timezone="UTC", top_k=3)
    result = await pipeline.generate_digest(chat, triggered_by="manual")
    print("\n\n".join(result.messages))


if __name__ == "__main__":
    asyncio.run(main())
