"""Digest generation pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database
from ai_safety_brief.models import CandidateItem, ChatSettings, DigestEntry, DigestResult, StoredItem
from ai_safety_brief.services.ingestion import SourceCollector
from ai_safety_brief.services.llm_refiner import GroqRefiner
from ai_safety_brief.services.ranking import dedupe_candidates, is_relevant, score_candidate
from ai_safety_brief.services.summarizer import Summarizer
from ai_safety_brief.utils.text import shorten, split_for_telegram

logger = logging.getLogger(__name__)

SECTION_LABELS = {
    "news": "News",
    "paper": "Papers",
    "opinion": "Opinion",
}


class DigestPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        collector: SourceCollector,
        summarizer: Summarizer,
        refiner: GroqRefiner,
    ) -> None:
        self.settings = settings
        self.db = db
        self.collector = collector
        self.summarizer = summarizer
        self.refiner = refiner

    async def generate_digest(self, chat: ChatSettings, triggered_by: str) -> DigestResult:
        now = datetime.now(timezone.utc)
        sources = {source.key: source for source in self.collector.enabled_sources(chat)}
        candidates = await self.collector.collect_recent(chat)
        relevant = [item for item in candidates if is_relevant(item, sources[item.source_key])]
        relevant = dedupe_candidates(relevant)

        for candidate in relevant:
            candidate.score = score_candidate(candidate, sources[candidate.source_key], now)

        relevant.sort(key=lambda item: item.score, reverse=True)
        to_enrich = relevant[: max(chat.top_k * 4, 12)]
        await self.collector.enrich_candidates(to_enrich)
        for candidate in to_enrich:
            candidate.score = score_candidate(candidate, sources[candidate.source_key], now)

        merged = dedupe_candidates(to_enrich + relevant[len(to_enrich) :])
        merged.sort(key=lambda item: item.score, reverse=True)

        seen_ids = await self.db.get_seen_item_ids(chat.chat_id, chat.repeat_window_days)
        selected_candidates: list[CandidateItem] = []
        selected_entries: list[StoredItem] = []
        for candidate in merged:
            if len(selected_entries) >= chat.top_k:
                break
            candidate.summary, candidate.why_it_matters = self.summarizer.summarize(candidate)
            stored = await self.db.save_item(candidate)
            if stored.id in seen_ids:
                continue
            selected_candidates.append(candidate)
            selected_entries.append(stored)

        if not selected_entries:
            message = (
                "AI Safety Brief checked the configured sources but did not find enough fresh, high-confidence items "
                "for this run."
            )
            await self.db.save_digest_run(
                chat.chat_id,
                [],
                [message],
                triggered_by=triggered_by,
                status="empty",
            )
            return DigestResult(entries=[], messages=[message], generated_at=now)

        if self.refiner.enabled:
            await self.refiner.maybe_refine(selected_candidates)
            selected_entries = [await self.db.save_item(candidate) for candidate in selected_candidates]

        entries = self._to_digest_entries(selected_entries, selected_candidates)
        messages = self._format_digest_messages(chat, entries, now)
        await self.db.save_digest_run(chat.chat_id, entries, messages, triggered_by=triggered_by)
        await self.db.mark_seen(chat.chat_id, [entry.item.id for entry in entries], now)
        return DigestResult(entries=entries, messages=messages, generated_at=now)

    def _to_digest_entries(
        self,
        stored_items: list[StoredItem],
        candidates: list[CandidateItem],
    ) -> list[DigestEntry]:
        by_url = {item.canonical_url: item for item in stored_items}
        entries: list[DigestEntry] = []
        for rank, candidate in enumerate(candidates, start=1):
            stored = by_url.get(candidate.canonical_url)
            if not stored:
                continue
            entries.append(
                DigestEntry(
                    item=stored,
                    rank=rank,
                    section=SECTION_LABELS[stored.content_type],
                    score=candidate.score,
                )
            )
        return entries

    def _format_digest_messages(
        self,
        chat: ChatSettings,
        entries: list[DigestEntry],
        now: datetime,
    ) -> list[str]:
        lines = [
            "AI Safety Brief",
            now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "",
        ]

        grouped: dict[str, list[DigestEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.section].append(entry)

        for section in ("News", "Papers", "Opinion"):
            section_entries = grouped.get(section)
            if not section_entries:
                continue
            lines.append(section)
            for entry in section_entries:
                lines.append(f"- {entry.item.title}")
                lines.append(f"   {shorten(entry.item.summary, 280)}")
                lines.append(f"   Why it matters: {shorten(entry.item.why_it_matters, 180)}")
                lines.append(f"   Source: {entry.item.source_name} | {entry.item.canonical_url}")
                lines.append("")

        footer = ["Commands: /brief for an immediate digest, /settings to tune k, cadence, timezone, or sources."]
        text = "\n".join(lines + footer).strip()
        return split_for_telegram(text, self.settings.digest_message_limit)
