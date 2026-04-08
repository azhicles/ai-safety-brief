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
from ai_safety_brief.services.ranking import (
    adjusted_selection_score,
    dedupe_candidates,
    is_relevant,
    score_candidate,
)
from ai_safety_brief.services.summarizer import Summarizer
from ai_safety_brief.utils.text import shorten, split_for_telegram

logger = logging.getLogger(__name__)

SECTION_LABELS = {
    "news": "📰 News",
    "paper": "📄 Papers",
    "opinion": "💭 Opinion",
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
        selected_entries: list[StoredItem] = []
        selected_candidates = self._select_diverse_candidates(merged, sources, chat.top_k)
        for candidate in selected_candidates:
            candidate.summary, candidate.why_it_matters = self.summarizer.summarize(candidate)
            stored = await self.db.save_item(candidate)
            if stored.id in seen_ids:
                continue
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
            "🧠 AI Safety Brief",
            "a quick lap around the most relevant ai safety stories.",
            f"top {len(entries)} picks | {now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        grouped: dict[str, list[DigestEntry]] = defaultdict(list)
        for entry in entries:
            grouped[entry.section].append(entry)

        counter = 1
        for section in ("📰 News", "📄 Papers", "💭 Opinion"):
            section_entries = grouped.get(section)
            if not section_entries:
                continue
            lines.append(section)
            for entry in section_entries:
                headline_emoji = self._entry_emoji(entry)
                lines.append(f"{counter}. {headline_emoji} {entry.item.title}")
                lines.append(f"summary: {shorten(entry.item.summary, 180)}")
                lines.append(f"why it matters: {shorten(entry.item.why_it_matters, 120)}")
                lines.append(f"from: {entry.item.source_name}")
                lines.append(entry.item.canonical_url)
                lines.append("")
                counter += 1

        footer = ["use /brief for a refresh or /settings to tune k, cadence, timezone, and sources."]
        text = "\n".join(lines + footer).strip()
        return split_for_telegram(text, self.settings.digest_message_limit)

    def _select_diverse_candidates(
        self,
        candidates: list[CandidateItem],
        sources: dict[str, object],
        top_k: int,
    ) -> list[CandidateItem]:
        remaining = list(candidates)
        selected: list[CandidateItem] = []

        while remaining and len(selected) < top_k:
            remaining_by_type = defaultdict(int)
            for candidate in remaining:
                remaining_by_type[candidate.content_type] += 1

            best = max(
                remaining,
                key=lambda candidate: adjusted_selection_score(
                    candidate,
                    sources[candidate.source_key],
                    selected,
                    remaining_by_type,
                    top_k,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def _entry_emoji(self, entry: DigestEntry) -> str:
        title = entry.item.title.lower()
        if any(term in title for term in ("announce", "launch", "introduc", "initiative", "project")):
            return "🚨"
        if entry.item.content_type == "news":
            return "📰"
        if entry.item.content_type == "paper":
            return "📄"
        return "💭"
