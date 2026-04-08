"""Source collection and content extraction."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.parse import urlencode, urljoin, urlsplit

import aiohttp
import feedparser
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ai_safety_brief.config import Settings
from ai_safety_brief.models import CandidateItem, ChatSettings, SourceDefinition
from ai_safety_brief.sources import DEFAULT_SOURCES, build_x_sources
from ai_safety_brief.utils.text import normalize_url, normalize_whitespace, shorten
from ai_safety_brief.utils.time import utc_now

logger = logging.getLogger(__name__)


class SourceCollector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sources = list(DEFAULT_SOURCES) + build_x_sources(
            settings.x_rss_base_url, settings.x_accounts
        )

    def list_sources(self) -> list[SourceDefinition]:
        return list(self.sources)

    def enabled_sources(self, chat: ChatSettings) -> list[SourceDefinition]:
        disabled = set(chat.disabled_sources)
        return [source for source in self.sources if source.key not in disabled]

    async def collect_recent(self, chat: ChatSettings) -> list[CandidateItem]:
        now = utc_now()
        sources = self.enabled_sources(chat)
        timeout = aiohttp.ClientTimeout(total=self.settings.item_fetch_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._default_headers()) as session:
            tasks = [self._collect_source(session, source, now) for source in sources]
            groups = await asyncio.gather(*tasks, return_exceptions=True)

            candidates: list[CandidateItem] = []
            for source, group in zip(sources, groups):
                if isinstance(group, Exception):
                    logger.warning("Source %s failed: %s", source.key, group)
                    continue
                candidates.extend(group)
            return candidates

    async def enrich_candidates(self, candidates: list[CandidateItem]) -> list[CandidateItem]:
        timeout = aiohttp.ClientTimeout(total=self.settings.item_fetch_timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._default_headers()) as session:
            tasks = [self._enrich_candidate(session, candidate) for candidate in candidates]
            await asyncio.gather(*tasks, return_exceptions=True)
        return candidates

    async def _collect_source(
        self,
        session: aiohttp.ClientSession,
        source: SourceDefinition,
        now: datetime,
    ) -> list[CandidateItem]:
        if source.mode in {"rss", "x_rss"}:
            return await self._collect_feed(session, source, now)
        if source.mode == "listing":
            return await self._collect_listing(session, source, now)
        if source.mode == "arxiv":
            return await self._collect_arxiv(session, source, now)
        return []

    async def _collect_feed(
        self,
        session: aiohttp.ClientSession,
        source: SourceDefinition,
        now: datetime,
    ) -> list[CandidateItem]:
        url = source.url
        if source.mode == "x_rss":
            url = source.url.rstrip("/") + "/rss"
        text = await self._fetch_text(session, url)
        if not text:
            return []
        parsed = feedparser.parse(text)
        return self._feed_entries_to_candidates(parsed.entries, source, now)

    async def _collect_listing(
        self,
        session: aiohttp.ClientSession,
        source: SourceDefinition,
        now: datetime,
    ) -> list[CandidateItem]:
        html = await self._fetch_text(session, source.listing_url or source.url)
        if not html:
            return []
        return self._parse_listing(source, html, now)

    async def _collect_arxiv(
        self,
        session: aiohttp.ClientSession,
        source: SourceDefinition,
        now: datetime,
    ) -> list[CandidateItem]:
        params = {
            "search_query": source.query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": "12",
        }
        text = await self._fetch_text(session, f"{source.url}?{urlencode(params)}")
        if not text:
            return []
        parsed = feedparser.parse(text)
        items: list[CandidateItem] = []
        for entry in parsed.entries:
            published = self._parse_feed_date(entry)
            if published and published < now - timedelta(hours=source.recency_hours):
                continue
            link = entry.get("link") or ""
            items.append(
                CandidateItem(
                    source_key=source.key,
                    source_name=source.name,
                    title=normalize_whitespace(entry.get("title", "")),
                    canonical_url=normalize_url(link),
                    content_type=source.content_type,
                    published_at=published,
                    author=", ".join(author.get("name", "") for author in entry.get("authors", [])),
                    excerpt=normalize_whitespace(unescape(entry.get("summary", ""))),
                    raw_text=normalize_whitespace(unescape(entry.get("summary", ""))),
                    metadata={"query": source.query},
                )
            )
        return items

    async def _enrich_candidate(
        self,
        session: aiohttp.ClientSession,
        candidate: CandidateItem,
    ) -> None:
        if candidate.raw_text and len(candidate.raw_text) >= 400:
            return
        html = await self._fetch_text(session, candidate.canonical_url)
        if not html:
            return
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            favor_recall=True,
        )
        if extracted:
            candidate.raw_text = shorten(normalize_whitespace(extracted), self.settings.item_preview_chars)

    async def _fetch_text(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            async with session.get(url, allow_redirects=True) as response:
                if response.status >= 400:
                    logger.warning("Request failed for %s with status %s", url, response.status)
                    return ""
                return await response.text()
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return ""

    def _feed_entries_to_candidates(
        self,
        entries: list[dict],
        source: SourceDefinition,
        now: datetime,
    ) -> list[CandidateItem]:
        items: list[CandidateItem] = []
        for entry in entries[:12]:
            published = self._parse_feed_date(entry)
            if published and published < now - timedelta(hours=source.recency_hours):
                continue
            link = normalize_url(entry.get("link", ""))
            if not link:
                continue
            items.append(
                CandidateItem(
                    source_key=source.key,
                    source_name=source.name,
                    title=normalize_whitespace(unescape(entry.get("title", ""))),
                    canonical_url=link,
                    content_type=source.content_type,
                    published_at=published,
                    author=self._feed_author(entry),
                    excerpt=normalize_whitespace(unescape(entry.get("summary", ""))),
                    metadata={"source_url": source.url},
                )
            )
        return items

    def _parse_listing(
        self,
        source: SourceDefinition,
        html: str,
        now: datetime,
    ) -> list[CandidateItem]:
        soup = BeautifulSoup(html, "html.parser")
        articles = soup.find_all("article")
        candidates: list[CandidateItem] = []
        seen_links: set[str] = set()

        def add_candidate(anchor, container) -> None:
            href = anchor.get("href")
            title = normalize_whitespace(anchor.get_text(" ", strip=True))
            if not href or not title:
                return
            url = normalize_url(urljoin(source.url, href))
            if url == normalize_url(source.url) or url in seen_links:
                return
            if urlsplit(url).netloc and urlsplit(url).netloc not in urlsplit(source.url).netloc:
                return
            excerpt = ""
            if container is not None:
                para = container.find("p")
                if para:
                    excerpt = normalize_whitespace(para.get_text(" ", strip=True))
            published = None
            if container is not None:
                time_tag = container.find("time")
                if time_tag:
                    published = self._parse_any_date(
                        time_tag.get("datetime") or time_tag.get_text(" ", strip=True)
                    )
            if published and published < now - timedelta(hours=source.recency_hours):
                return
            seen_links.add(url)
            candidates.append(
                CandidateItem(
                    source_key=source.key,
                    source_name=source.name,
                    title=title,
                    canonical_url=url,
                    content_type=source.content_type,
                    published_at=published,
                    excerpt=excerpt,
                    metadata={"listing_url": source.listing_url or source.url},
                )
            )

        for article in articles[:16]:
            anchor = article.find("a", href=True)
            if anchor:
                add_candidate(anchor, article)

        if not candidates:
            for heading_name in ("h1", "h2", "h3", "h4"):
                for heading in soup.find_all(heading_name):
                    anchor = heading.find("a", href=True)
                    if anchor:
                        add_candidate(anchor, heading.parent)
                if len(candidates) >= 12:
                    break

        return candidates[:12]

    def _feed_author(self, entry: dict) -> str:
        if "author" in entry:
            return normalize_whitespace(entry["author"])
        authors = entry.get("authors", [])
        if authors:
            return ", ".join(author.get("name", "") for author in authors)
        return ""

    def _parse_feed_date(self, entry: dict) -> datetime | None:
        if entry.get("published_parsed"):
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        for key in ("published", "updated"):
            if entry.get(key):
                return self._parse_any_date(entry.get(key))
        return None

    def _parse_any_date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = parsedate_to_datetime(value)
        except Exception:
            try:
                parsed = date_parser.parse(value)
            except Exception:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "AI-Safety-Brief/1.0 (+https://github.com/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

