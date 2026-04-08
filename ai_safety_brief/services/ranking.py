"""Ranking, relevance, and dedupe helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math

from rapidfuzz import fuzz

from ai_safety_brief.models import CandidateItem, ChatSettings, SourceDefinition
from ai_safety_brief.personalization import infer_topics
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


def score_candidate(
    candidate: CandidateItem,
    source: SourceDefinition,
    now: datetime,
    chat: ChatSettings | None = None,
) -> float:
    title_text = normalize_whitespace(candidate.title.lower())
    body_text = normalize_whitespace(
        f"{candidate.excerpt} {candidate.raw_text[:1200]}".lower()
    )
    topics, strongest_topic, topic_scores = infer_topics(candidate.title, candidate.excerpt, candidate.raw_text)
    matched_focus_topics = [topic for topic in topics if chat and topic in chat.focus_topics]

    keyword_hits = sum(1 for keyword in AI_SAFETY_KEYWORDS if keyword in body_text)
    title_hits = sum(1 for keyword in AI_SAFETY_KEYWORDS if keyword in title_text)
    title_words = Counter(words(title_text))
    body_words = Counter(words(body_text))
    overlap = sum(min(title_words[token], body_words[token]) for token in title_words)

    source_authority_boost = source.authority_weight * 10.0
    title_keyword_boost = title_hits * 4.0
    body_keyword_boost = keyword_hits * 1.1
    overlap_boost = overlap * 0.25

    score = source_authority_boost
    score += title_keyword_boost
    score += body_keyword_boost
    score += overlap_boost

    announcement_hits = sum(1 for term in ANNOUNCEMENT_TERMS if term in title_text)
    signal_hits = sum(1 for term in HIGH_SIGNAL_TERMS if term in f"{title_text} {body_text}")
    announcement_boost = announcement_hits * 3.0
    signal_boost = signal_hits * 1.3
    major_news_boost = 0.0

    score += announcement_boost
    score += signal_boost
    if source.key in MAJOR_NEWS_SOURCES:
        major_news_boost += announcement_hits * 4.5
        major_news_boost += signal_hits * 0.8
        score += major_news_boost

    content_type_boost = 0.0
    if candidate.content_type == "paper":
        content_type_boost -= 1.5
    elif candidate.content_type == "opinion":
        content_type_boost += 0.5
    else:
        content_type_boost += 3.0
    score += content_type_boost

    recency_boost = 0.0
    age_hours = None
    if candidate.published_at is not None:
        age_hours = max(
            0.0,
            (now - candidate.published_at.astimezone(timezone.utc)).total_seconds() / 3600,
        )
        recency_boost = max(0.0, 10.0 - age_hours / 10.0)
        score += recency_boost

    trusted_source_boost = 0.0
    if source.key in TRUSTED_ALWAYS_RELEVANT:
        trusted_source_boost += 2.0
        score += trusted_source_boost
    major_source_news_boost = 0.0
    if source.key in MAJOR_NEWS_SOURCES and candidate.content_type == "news":
        major_source_news_boost += 6.0
        score += major_source_news_boost
    arxiv_penalty = 0.0
    if candidate.content_type == "paper" and source.key.startswith("arxiv_"):
        arxiv_penalty -= 1.0
        score += arxiv_penalty

    topic_focus_boost = 0.0
    content_mix_boost = 0.0
    if chat:
        if chat.focus_topics:
            if matched_focus_topics:
                topic_focus_boost += 5.0 + 1.5 * len(matched_focus_topics)
            else:
                topic_focus_boost -= 2.0
        content_mix_boost += _content_mix_boost(candidate, topics, chat.content_mix)
        score += topic_focus_boost + content_mix_boost

    candidate.metadata.update(
        {
            "topics": topics,
            "strongest_topic": strongest_topic or "",
            "topic_scores": topic_scores,
            "matched_focus_topics": matched_focus_topics,
            "source_authority": round(source.authority_weight, 3),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "announcement_hits": announcement_hits,
            "signal_hits": signal_hits,
            "major_news_source": source.key in MAJOR_NEWS_SOURCES,
            "source_authority_boost": round(source_authority_boost, 2),
            "title_keyword_boost": round(title_keyword_boost, 2),
            "body_keyword_boost": round(body_keyword_boost, 2),
            "overlap_boost": round(overlap_boost, 2),
            "announcement_boost": round(announcement_boost, 2),
            "signal_boost": round(signal_boost, 2),
            "major_news_boost": round(major_news_boost + major_source_news_boost, 2),
            "content_type_boost": round(content_type_boost, 2),
            "recency_boost": round(recency_boost, 2),
            "topic_focus_boost": round(topic_focus_boost, 2),
            "content_mix_boost": round(content_mix_boost, 2),
            "trusted_source_boost": round(trusted_source_boost, 2),
            "arxiv_penalty": round(arxiv_penalty, 2),
        }
    )
    candidate.metadata["score_reasons"] = build_score_reasons(candidate, chat)
    candidate.metadata["final_score"] = round(score, 2)
    candidate.score = score

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
    chat: ChatSettings | None = None,
) -> float:
    score = candidate.score
    same_source_count = sum(1 for item in selected if item.source_key == candidate.source_key)
    same_type_count = sum(1 for item in selected if item.content_type == candidate.content_type)
    score -= same_source_count * 5.0
    score -= same_type_count * 2.0

    desired_papers, desired_news, desired_opinion = _desired_mix_counts(target_top_k, chat.content_mix if chat else "balanced")
    if candidate.content_type == "paper" and same_type_count >= desired_papers and remaining_by_type["news"] > 0:
        score -= 8.0
    if candidate.content_type == "news" and same_type_count < desired_news:
        score += 4.0
    if candidate.content_type == "opinion" and same_type_count < desired_opinion:
        score += 2.0
    if source.key in MAJOR_NEWS_SOURCES and candidate.content_type == "news":
        score += 3.0
    if candidate.content_type == "opinion" and remaining_by_type["news"] == 0:
        score += 1.0
    return score


def passes_alert_threshold(
    candidate: CandidateItem,
    source: SourceDefinition,
    chat: ChatSettings,
    now: datetime,
) -> bool:
    if chat.alert_mode == "off":
        return False
    if candidate.published_at is None:
        return False

    age_hours = max(
        0.0,
        (now - candidate.published_at.astimezone(timezone.utc)).total_seconds() / 3600,
    )
    announcement_hits = int(candidate.metadata.get("announcement_hits", 0))
    signal_hits = int(candidate.metadata.get("signal_hits", 0))
    combined_signal = announcement_hits + signal_hits

    if chat.alert_mode == "strict":
        return (
            candidate.score >= 40
            and age_hours <= 24
            and (source.key in MAJOR_NEWS_SOURCES or combined_signal >= 2)
        )

    if chat.alert_mode == "moderate":
        if candidate.score < 34 or age_hours > 24:
            return False
        if candidate.content_type in {"paper", "opinion"}:
            return source.authority_weight >= 1.2 or combined_signal >= 2
        return True

    return candidate.score >= 30 and age_hours <= 36


def build_score_reasons(candidate: CandidateItem, chat: ChatSettings | None = None) -> list[str]:
    metadata = candidate.metadata
    reasons: list[str] = []
    source_authority = float(metadata.get("source_authority", 0.0) or 0.0)
    if source_authority:
        reasons.append(f"source authority {source_authority:.2f}")
    topics = list(metadata.get("topics", []))
    if topics:
        reasons.append(f"topic match: {', '.join(topics[:2])}")
    matched_focus = list(metadata.get("matched_focus_topics", []))
    if matched_focus:
        reasons.append(f"your focus topics: {', '.join(matched_focus[:2])}")
    if float(metadata.get("recency_boost", 0.0) or 0.0) > 0:
        reasons.append("very recent")
    if float(metadata.get("major_news_boost", 0.0) or 0.0) > 0:
        reasons.append("major-news boost")
    if chat and float(metadata.get("content_mix_boost", 0.0) or 0.0) > 0:
        reasons.append(f"{chat.content_mix} mix boost")
    return reasons


def build_item_explanation(candidate_or_item, chat: ChatSettings | None = None) -> str:
    metadata = getattr(candidate_or_item, "metadata", {}) or {}
    reasons = metadata.get("score_reasons") or build_score_reasons(candidate_or_item, chat)
    score = metadata.get("final_score")
    lines = ["why this was picked:"]
    if reasons:
        lines.extend(f"- {reason}" for reason in reasons[:5])
    else:
        lines.append("- source authority, recency, and ai-safety relevance")
    if score is not None:
        lines.append(f"- final score: {score}")
    return "\n".join(lines)


def _content_mix_boost(candidate: CandidateItem, topics: list[str], content_mix: str) -> float:
    boost = 0.0
    if content_mix == "news-heavy":
        if candidate.content_type == "news":
            boost += 2.5
        elif candidate.content_type == "paper":
            boost -= 1.5
        else:
            boost -= 0.5
    elif content_mix == "papers-heavy":
        if candidate.content_type == "paper":
            boost += 2.5
        elif candidate.content_type == "news":
            boost -= 0.5
    elif content_mix == "policy-heavy":
        if candidate.content_type == "opinion":
            boost += 2.5
        if any(topic in {"governance", "security"} for topic in topics):
            boost += 1.5
        if candidate.content_type == "paper":
            boost -= 0.5
    return boost


def _desired_mix_counts(target_top_k: int, content_mix: str) -> tuple[int, int, int]:
    if content_mix == "news-heavy":
        return max(1, math.ceil(target_top_k / 6)), max(2, math.ceil(target_top_k * 0.7)), 1
    if content_mix == "papers-heavy":
        return max(1, math.ceil(target_top_k / 2)), max(1, math.ceil(target_top_k / 3)), 1
    if content_mix == "policy-heavy":
        return max(1, math.ceil(target_top_k / 4)), max(1, math.ceil(target_top_k / 3)), max(1, math.ceil(target_top_k / 3))
    return max(1, math.ceil(target_top_k / 4)), max(2, math.ceil(target_top_k / 2)), 1
