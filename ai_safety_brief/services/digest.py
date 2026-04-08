"""Digest generation pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database
from ai_safety_brief.models import AlertResult, CandidateItem, ChatSettings, DigestEntry, DigestResult, StoredItem
from ai_safety_brief.personalization import within_quiet_hours
from ai_safety_brief.services.ingestion import SourceCollector
from ai_safety_brief.services.llm_refiner import GroqRefiner
from ai_safety_brief.services.ranking import (
    adjusted_selection_score,
    build_item_explanation,
    dedupe_candidates,
    is_relevant,
    passes_alert_threshold,
    score_candidate,
)
from ai_safety_brief.services.summarizer import Summarizer
from ai_safety_brief.utils.text import normalize_whitespace, shorten, split_for_telegram

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
        seen_ids = await self.db.get_seen_item_ids(chat.chat_id, chat.repeat_window_days)
        entries = await self._prepare_entries(
            chat,
            limit=chat.top_k,
            seen_ids=seen_ids,
            exclude_item_ids=set(),
            now=now,
        )
        if not entries:
            message = (
                "AI Safety Brief checked the configured sources but did not find enough fresh, relevant items "
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

        messages = self._format_digest_messages(chat, entries, now)
        await self.db.save_digest_run(chat.chat_id, entries, messages, triggered_by=triggered_by)
        await self.db.mark_seen(chat.chat_id, [entry.item.id for entry in entries], now)
        return DigestResult(entries=entries, messages=messages, generated_at=now)

    async def generate_more(self, chat: ChatSettings) -> DigestResult:
        now = datetime.now(timezone.utc)
        seen_ids = await self.db.get_seen_item_ids(chat.chat_id, chat.repeat_window_days)
        latest_item_ids = set(await self.db.latest_run_item_ids(chat.chat_id))
        entries = await self._prepare_entries(
            chat,
            limit=max(1, min(chat.top_k, 5)),
            seen_ids=seen_ids,
            exclude_item_ids=latest_item_ids,
            now=now,
        )
        if not entries:
            message = "nothing especially fresh sits just below the current shortlist right now."
            return DigestResult(entries=[], messages=[message], generated_at=now)
        messages = self._format_digest_messages(
            chat,
            entries,
            now,
            intro="more picks just below the main cutoff.",
        )
        return DigestResult(entries=entries, messages=messages, generated_at=now)

    async def generate_alert(self, chat: ChatSettings) -> AlertResult | None:
        now = datetime.now(timezone.utc)
        if chat.alert_mode == "off" or within_quiet_hours(chat, now):
            return None
        if await self.db.count_alerts_since(chat.chat_id, now - timedelta(hours=24)) >= 3:
            return None
        latest_alert = await self.db.latest_alert_at(chat.chat_id)
        if latest_alert and (now - latest_alert).total_seconds() < 1800:
            return None

        sources, ranked = await self._collect_ranked_candidates(chat, now)
        alerted_ids = await self.db.get_alerted_item_ids(chat.chat_id)
        selection_pool = self._select_diverse_candidates(
            ranked,
            sources,
            selection_limit=min(len(ranked), max(chat.top_k * 4, 12)),
            target_top_k=max(chat.top_k, 3),
            chat=chat,
        )
        for candidate in selection_pool:
            if not passes_alert_threshold(candidate, sources[candidate.source_key], chat, now):
                continue
            candidate.summary, candidate.why_it_matters = self.summarizer.summarize(candidate)
            stored = await self.db.save_item(candidate)
            if stored.id in alerted_ids:
                continue
            if self.refiner.enabled:
                await self.refiner.maybe_refine([candidate])
                stored = await self.db.save_item(candidate)
            message = self._format_alert_message(stored, candidate, now)
            return AlertResult(item=stored, message=message, generated_at=now)
        return None

    async def _collect_ranked_candidates(
        self,
        chat: ChatSettings,
        now: datetime,
    ) -> tuple[dict[str, object], list[CandidateItem]]:
        sources = {source.key: source for source in self.collector.enabled_sources(chat)}
        candidates = await self.collector.collect_recent(chat)
        relevant = [item for item in candidates if is_relevant(item, sources[item.source_key])]
        relevant = dedupe_candidates(relevant)

        for candidate in relevant:
            candidate.score = score_candidate(candidate, sources[candidate.source_key], now, chat)

        relevant.sort(key=lambda item: item.score, reverse=True)
        to_enrich = relevant[: max(chat.top_k * 4, 12)]
        await self.collector.enrich_candidates(to_enrich)
        for candidate in to_enrich:
            candidate.score = score_candidate(candidate, sources[candidate.source_key], now, chat)

        merged = dedupe_candidates(to_enrich + relevant[len(to_enrich) :])
        merged.sort(key=lambda item: item.score, reverse=True)
        return sources, merged

    async def _prepare_entries(
        self,
        chat: ChatSettings,
        limit: int,
        seen_ids: set[int],
        exclude_item_ids: set[int],
        now: datetime,
    ) -> list[DigestEntry]:
        sources, merged = await self._collect_ranked_candidates(chat, now)
        selected_entries: list[StoredItem] = []
        selected_candidates: list[CandidateItem] = []
        selection_pool = self._select_diverse_candidates(
            merged,
            sources,
            selection_limit=min(len(merged), max(limit * 6, 18)),
            target_top_k=limit,
            chat=chat,
        )
        for candidate in selection_pool:
            candidate.summary, candidate.why_it_matters = self.summarizer.summarize(candidate)
            stored = await self.db.save_item(candidate)
            if stored.id in seen_ids or stored.id in exclude_item_ids:
                continue
            selected_entries.append(stored)
            selected_candidates.append(candidate)
            if len(selected_entries) >= limit:
                break

        if not selected_entries:
            return []

        if self.refiner.enabled:
            await self.refiner.maybe_refine(selected_candidates)
            selected_entries = [await self.db.save_item(candidate) for candidate in selected_candidates]

        return self._to_digest_entries(selected_entries, selected_candidates)

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
        intro: str | None = None,
    ) -> list[str]:
        lines = [
            "🧠 AI Safety Brief",
            intro or "a quick lap around the most relevant ai safety stories.",
            f"top {len(entries)} picks | {now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        for entry in sorted(entries, key=lambda digest_entry: digest_entry.rank):
            headline_emoji = self._entry_emoji(entry)
            lines.append(f"{entry.rank}. {entry.item.title} {headline_emoji}")
            lines.append("")
            paragraph = shorten(
                f"{normalize_whitespace(entry.item.summary)} {normalize_whitespace(entry.item.why_it_matters)}",
                170,
            )
            lines.append(paragraph)
            lines.append("")
            lines.append(f"{entry.item.source_name} | {entry.item.canonical_url}")
            lines.append("")

        footer = ["use /brief for a refresh or /settings to tune k, cadence, timezone, and sources."]
        text = "\n".join(lines + footer).strip()
        return split_for_telegram(text, self.settings.digest_message_limit)

    def _format_alert_message(
        self,
        stored: StoredItem,
        candidate: CandidateItem,
        now: datetime,
    ) -> str:
        lines = [
            "🚨 AI Safety Alert",
            "",
            f"{stored.title} {self._entry_emoji(DigestEntry(item=stored, rank=1, section='', score=candidate.score))}",
            "",
            shorten(
                f"{normalize_whitespace(stored.summary)} {normalize_whitespace(stored.why_it_matters)}",
                170,
            ),
            "",
            f"{stored.source_name} | {stored.canonical_url}",
            "",
            f"score snapshot: {candidate.metadata.get('final_score', round(candidate.score, 2))} | {now.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ]
        return "\n".join(lines).strip()

    def _select_diverse_candidates(
        self,
        candidates: list[CandidateItem],
        sources: dict[str, object],
        selection_limit: int,
        target_top_k: int,
        chat: ChatSettings | None = None,
    ) -> list[CandidateItem]:
        remaining = list(candidates)
        selected: list[CandidateItem] = []

        while remaining and len(selected) < selection_limit:
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
                    target_top_k,
                    chat,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return selected

    def explain_item(self, item: StoredItem, chat: ChatSettings) -> str:
        return build_item_explanation(item, chat)

    def explain_digest_selection(self, chat: ChatSettings, entries: list[DigestEntry]) -> str:
        topics = ", ".join(chat.focus_topics) if chat.focus_topics else "all topics"
        top_reasons = []
        for entry in entries[:3]:
            reasons = entry.item.metadata.get("score_reasons", [])
            if reasons:
                top_reasons.append(f"{entry.rank}. {entry.item.title}: {', '.join(reasons[:2])}")
        body = "\n".join(top_reasons) if top_reasons else "the current shortlist leans on source authority, recency, and ai-safety relevance."
        return (
            "why these picks:\n"
            f"- focus topics: {topics}\n"
            f"- content mix: {chat.content_mix}\n"
            f"- alerts mode: {chat.alert_mode}\n"
            f"{body}"
        )

    def _entry_emoji(self, entry: DigestEntry) -> str:
        title = entry.item.title.lower()
        if any(term in title for term in ("announce", "launch", "introduc", "initiative", "project")):
            return "🚨"
        if entry.item.content_type == "news":
            return "📰"
        if entry.item.content_type == "paper":
            return "📄"
        return "💭"
