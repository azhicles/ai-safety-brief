"""Text helpers."""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_markdown(text: str) -> str:
    return text.replace("*", "").replace("_", "").replace("`", "")


def split_sentences(text: str) -> list[str]:
    clean = normalize_whitespace(text)
    if not clean:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(clean) if part.strip()]


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text or "")]


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def shorten(text: str, limit: int) -> str:
    clean = normalize_whitespace(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1].rstrip() + "…"


def split_for_telegram(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit * 0.5:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def lowercase_sentence_start(text: str) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return clean
    match = re.search(r"[A-Za-z]+", clean)
    if not match:
        return clean
    start, end = match.span()
    return clean[:start] + clean[start:end].lower() + clean[end:]
