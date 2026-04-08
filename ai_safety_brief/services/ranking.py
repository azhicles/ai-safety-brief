"""Ranking, relevance, and dedupe helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from rapidfuzz import fuzz

from ai_safety_brief.models import CandidateItem, SourceDefinition
from ai_safety_brief.sources import AI_SAFETY_KEYWORDS
from ai_safety_brief.utils.text import normalize_url, normalize_whitespace, words

TRUSTED_ALWAYS_RELEVANT = {
    "alignment_forum",
    "metr_blog",
    "redwood_blog",
    "apollo_blog",
    "cais_blog",
    "govai_blog",
    "law_ai_blog",
    "ai_safety_newsletter",
    "uk_aisi",
    "arxiv_alignment",
    "arxiv_evals",
}


def score_candidate(candidate: CandidateItem, source: SourceDefinition, now: datetime) -> float:
    title_text = normalize_whitespace(candidate.title.lower())
    body_text = normalize_whitespace(
        f"{candidate.excerpt} {candidate.raw_text[:1200]}".lower()
    )

    keyword_hits = sum(1 for keyword in AI_SAFETY_KEYWORDS if keyword in body_text)
    title_hits = sum(1 for keyword in AI_SAFETY_KEYWORDS if keyword in title_text)
    title_words = Counter(words(title_text))
    body_words = Counter(words(body_text))
    overlap = sum(min(title_words[token], body_words[token]) for token in title_words)

    score = source.authority_weight * 10.0
    score += title_hits * 4.0
    score += keyword_hits * 1.1
    score += overlap * 0.25

    if candidate.content_type == "paper":
        score += 1.5
    elif candidate.content_type == "opinion":
        score += 0.5

    if candidate.published_at is not None:
        age_hours = max(
            0.0,
            (now - candidate.published_at.astimezone(timezone.utc)).total_seconds() / 3600,
        )
        score += max(0.0, 8.0 - age_hours / 12.0)

    if source.key in TRUSTED_ALWAYS_RELEVANT:
        score += 2.0

    return score


def is_relevant(candidate: CandidateItem, source: SourceDefinition) -> bool:
    text = f"{candidate.title} {candidate.excerpt} {candidate.raw_text[:600]}".lower()
    if source.key in TRUSTED_ALWAYS_RELEVANT:
        return True
    return any(keyword in text for keyword in AI_SAFETY_KEYWORDS)


def dedupe_candidates(candidates: list[CandidateItem]) -> list[CandidateItem]:
    deduped: list[CandidateItem] = []
    seen_urls: set[str] = set()
    normalized_titles: list[str] = []
    for candidate in candidates:
        normalized = normalize_url(candidate.canonical_url)
        if normalized in seen_urls:
            continue
        title = normalize_whitespace(candidate.title.lower())
        if any(fuzz.token_set_ratio(title, existing) >= 95 for existing in normalized_titles):
            continue
        deduped.append(candidate)
        seen_urls.add(normalized)
        normalized_titles.append(title)
    return deduped

