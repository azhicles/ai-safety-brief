"""Ranking, relevance, and dedupe helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math

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
}

MAJOR_NEWS_SOURCES = {
    "anthropic_research",
    "anthropic_news",
    "openai_research",
    "openai_news",
    "deepmind_blog",
    "uk_aisi",
    "cais_blog",
}

ANNOUNCEMENT_TERMS = (
    "announcing",
    "announce",
    "introducing",
    "launch",
    "launched",
    "initiative",
    "partnership",
    "project",
    "system card",
    "preparedness",
    "safety case",
    "policy proposal",
    "security",
    "cybersecurity",
    "vulnerability",
    "evaluation report",
)

HIGH_SIGNAL_TERMS = (
    "frontier",
    "critical infrastructure",
    "zero-day",
    "dangerous capability",
    "deployment",
    "oversight",
    "eval",
    "red team",
    "governance",
    "safeguard",
)

OFFICIAL_NEWS_RELEVANCE_TERMS = (
    "security",
    "secure",
    "cyber",
    "critical software",
    "critical infrastructure",
    "preparedness",
    "deployment",
    "safeguard",
    "risk",
    "responsibility",
    "system card",
    "frontier",
)

ARXIV_STRONG_TERMS = (
    "alignment",
    "deceptive",
    "oversight",
    "red team",
    "interpretability",
    "robustness",
    "monitoring",
    "eval",
    "evaluation",
    "governance",
    "misalignment",
    "safety",
    "control",
)

AI_CONTEXT_TERMS = (
    "ai",
    "model",
    "models",
    "agent",
    "agents",
    "agentic",
    "language model",
    "llm",
    "frontier",
    "machine learning",
    "neural",
    "gpt",
    "claude",
)

HIGH_CONFIDENCE_RELEVANCE_TERMS = (
    "ai safety",
    "alignment",
    "misalignment",
    "deceptive",
    "oversight",
    "red team",
    "interpretability",
    "system card",
    "model welfare",
    "frontier model",
    "constitutional",
    "biosecurity",
    "cybersecurity",
)

GENERIC_RELEVANCE_TERMS = (
    "control",
    "monitoring",
    "robustness",
    "evaluation",
    "eval",
    "governance",
    "policy",
    "security",
    "safeguard",
    "preparedness",
)


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

    announcement_hits = sum(1 for term in ANNOUNCEMENT_TERMS if term in title_text)
    signal_hits = sum(1 for term in HIGH_SIGNAL_TERMS if term in f"{title_text} {body_text}")
    score += announcement_hits * 3.0
    score += signal_hits * 1.3
    if source.key in MAJOR_NEWS_SOURCES:
        score += announcement_hits * 4.5
        score += signal_hits * 0.8

    if candidate.content_type == "paper":
        score -= 1.5
    elif candidate.content_type == "opinion":
        score += 0.5
    else:
        score += 3.0

    if candidate.published_at is not None:
        age_hours = max(
            0.0,
            (now - candidate.published_at.astimezone(timezone.utc)).total_seconds() / 3600,
        )
        score += max(0.0, 10.0 - age_hours / 10.0)

    if source.key in TRUSTED_ALWAYS_RELEVANT:
        score += 2.0
    if source.key in MAJOR_NEWS_SOURCES and candidate.content_type == "news":
        score += 6.0
    if candidate.content_type == "paper" and source.key.startswith("arxiv_"):
        score -= 1.0

    return score


def is_relevant(candidate: CandidateItem, source: SourceDefinition) -> bool:
    text = f"{candidate.title} {candidate.excerpt} {candidate.raw_text[:600]}".lower()
    if source.key in TRUSTED_ALWAYS_RELEVANT:
        return True
    if source.key.startswith("arxiv_"):
        return _matches_any(text, ARXIV_STRONG_TERMS) and _matches_any(text, AI_CONTEXT_TERMS)
    if source.key in MAJOR_NEWS_SOURCES and _matches_any(text, OFFICIAL_NEWS_RELEVANCE_TERMS):
        return True
    if _matches_any(text, HIGH_CONFIDENCE_RELEVANCE_TERMS):
        return True
    return _matches_any(text, GENERIC_RELEVANCE_TERMS) and _matches_any(text, AI_CONTEXT_TERMS)


def _matches_any(text: str, terms: tuple[str, ...]) -> bool:
    tokens = set(words(text))
    for term in terms:
        if " " in term:
            if term in text:
                return True
        elif term in tokens:
            return True
    return False


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


def adjusted_selection_score(
    candidate: CandidateItem,
    source: SourceDefinition,
    selected: list[CandidateItem],
    remaining_by_type: Counter[str],
    target_top_k: int,
) -> float:
    score = candidate.score
    same_source_count = sum(1 for item in selected if item.source_key == candidate.source_key)
    same_type_count = sum(1 for item in selected if item.content_type == candidate.content_type)
    score -= same_source_count * 5.0
    score -= same_type_count * 2.0

    desired_papers = max(1, math.ceil(target_top_k / 4))
    desired_news = max(2, math.ceil(target_top_k / 2))
    if candidate.content_type == "paper" and same_type_count >= desired_papers and remaining_by_type["news"] > 0:
        score -= 8.0
    if candidate.content_type == "news" and same_type_count < desired_news:
        score += 4.0
    if source.key in MAJOR_NEWS_SOURCES and candidate.content_type == "news":
        score += 3.0
    if candidate.content_type == "opinion" and remaining_by_type["news"] == 0:
        score += 1.0
    return score
