"""Personalization, topic, and alert helpers."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ai_safety_brief.models import ChatSettings
from ai_safety_brief.utils.text import normalize_whitespace, words

FOCUS_TOPICS: tuple[str, ...] = (
    "alignment",
    "evals",
    "interpretability",
    "governance",
    "security",
    "labs",
)

CONTENT_MIXES: tuple[str, ...] = (
    "balanced",
    "news-heavy",
    "papers-heavy",
    "policy-heavy",
)

ALERT_MODES: tuple[str, ...] = (
    "off",
    "strict",
    "moderate",
    "broad",
)

QUIET_HOUR_PRESETS: tuple[str, ...] = (
    "off",
    "22:00-07:00",
    "23:00-07:00",
    "00:00-08:00",
)

TOPIC_TERMS: dict[str, tuple[str, ...]] = {
    "alignment": (
        "alignment",
        "misalignment",
        "deceptive alignment",
        "deception",
        "control",
        "scalable oversight",
        "constitutional",
        "agent foundations",
    ),
    "evals": (
        "eval",
        "evaluation",
        "benchmark",
        "red team",
        "red teaming",
        "preparedness",
        "safety case",
        "monitoring",
        "oversight",
        "capability discovery",
    ),
    "interpretability": (
        "interpretability",
        "mechanistic",
        "circuits",
        "features",
        "transparency",
        "explainability",
        "probe",
        "latent",
    ),
    "governance": (
        "governance",
        "policy",
        "regulation",
        "standards",
        "audit",
        "auditing",
        "law",
        "treaty",
        "export control",
        "licensing",
    ),
    "security": (
        "security",
        "secure",
        "cyber",
        "cybersecurity",
        "biosecurity",
        "critical software",
        "critical infrastructure",
        "safeguard",
        "vulnerability",
        "zero-day",
    ),
    "labs": (
        "anthropic",
        "openai",
        "deepmind",
        "google deepmind",
        "frontier lab",
        "frontier model",
        "project",
        "system card",
        "model spec",
    ),
}


def parse_topics_csv(raw: str) -> list[str]:
    topics: list[str] = []
    for part in raw.split(","):
        topic = part.strip().lower()
        if not topic:
            continue
        if topic not in FOCUS_TOPICS:
            raise ValueError(f"Unknown topic: {topic}")
        if topic not in topics:
            topics.append(topic)
    return topics


def coerce_content_mix(value: str) -> str:
    mix = value.strip().lower()
    if mix not in CONTENT_MIXES:
        raise ValueError(f"Unknown content mix: {value}")
    return mix


def coerce_alert_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in ALERT_MODES:
        raise ValueError(f"Unknown alert mode: {value}")
    return mode


def parse_quiet_hours(value: str) -> tuple[str | None, str | None]:
    clean = value.strip().lower()
    if clean in {"off", "none", ""}:
        return None, None
    if "-" not in clean:
        raise ValueError("Quiet hours must use HH:MM-HH:MM format.")
    start, end = clean.split("-", 1)
    start = _validate_hhmm(start)
    end = _validate_hhmm(end)
    if start == end:
        raise ValueError("Quiet hours start and end cannot be identical.")
    return start, end


def format_quiet_hours(chat: ChatSettings) -> str:
    if not chat.quiet_hours_start or not chat.quiet_hours_end:
        return "off"
    return f"{chat.quiet_hours_start}-{chat.quiet_hours_end}"


def within_quiet_hours(chat: ChatSettings, now_utc: datetime) -> bool:
    if not chat.quiet_hours_start or not chat.quiet_hours_end:
        return False
    local_now = now_utc.astimezone(ZoneInfo(chat.timezone))
    start_hour, start_minute = _parse_hhmm(chat.quiet_hours_start)
    end_hour, end_minute = _parse_hhmm(chat.quiet_hours_end)
    now_minutes = local_now.hour * 60 + local_now.minute
    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute
    if start_minutes < end_minutes:
        return start_minutes <= now_minutes < end_minutes
    return now_minutes >= start_minutes or now_minutes < end_minutes


def topic_scores(title: str, excerpt: str, raw_text: str) -> dict[str, int]:
    text = normalize_whitespace(f"{title} {excerpt} {raw_text}").lower()
    tokens = set(words(text))
    scores: dict[str, int] = {}
    for topic, terms in TOPIC_TERMS.items():
        hits = 0
        for term in terms:
            if " " in term:
                if term in text:
                    hits += 2
            elif term in tokens:
                hits += 1
        scores[topic] = hits
    return scores


def infer_topics(title: str, excerpt: str, raw_text: str) -> tuple[list[str], str | None, dict[str, int]]:
    scores = topic_scores(title, excerpt, raw_text)
    topics = [topic for topic, score in scores.items() if score > 0]
    topics.sort(key=lambda topic: (-scores[topic], topic))
    strongest = topics[0] if topics else None
    return topics, strongest, scores


def topic_label(topic: str) -> str:
    return topic.replace("-", " ")


def _validate_hhmm(value: str) -> str:
    hour, minute = _parse_hhmm(value)
    return f"{hour:02d}:{minute:02d}"


def _parse_hhmm(value: str) -> tuple[int, int]:
    if ":" not in value:
        raise ValueError("Time must use HH:MM format.")
    hour_str, minute_str = value.split(":", 1)
    hour = int(hour_str)
    minute = int(minute_str)
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("Time must be a valid 24-hour clock value.")
    return hour, minute
