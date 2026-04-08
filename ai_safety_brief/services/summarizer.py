"""Summarization helpers."""

from __future__ import annotations

from collections import Counter

from ai_safety_brief.models import CandidateItem
from ai_safety_brief.sources import AI_SAFETY_KEYWORDS
from ai_safety_brief.utils.text import normalize_whitespace, shorten, split_sentences, words

THEME_TEMPLATES = (
    ("governance", "Could shape how frontier systems are governed, audited, or rolled out."),
    ("policy", "Changes the policy backdrop around advanced AI, which can move the whole field."),
    ("eval", "Makes risky model behavior easier to catch before anyone hits deploy."),
    ("red team", "Gives teams a sharper way to probe for failure modes and misuse."),
    ("interpretability", "Could make model behavior a bit less mysterious and a lot more inspectable."),
    ("oversight", "Strengthens the practical toolkit for keeping humans in the loop."),
    ("alignment", "Sits close to the core alignment question, not just the edges of it."),
    ("control", "Adds concrete ideas for keeping powerful systems inside safer bounds."),
    ("robustness", "Helps test whether systems stay dependable when the pressure goes up."),
)


class Summarizer:
    def summarize(self, item: CandidateItem) -> tuple[str, str]:
        summary = self._polish_summary(self._extractive_summary(item))
        why = self._polish_why(self._why_it_matters(item))
        return summary, why

    def _extractive_summary(self, item: CandidateItem) -> str:
        text = normalize_whitespace(f"{item.excerpt} {item.raw_text}")
        if not text:
            return shorten(item.title, 280)

        sentences = split_sentences(text)
        unique_sentences: list[str] = []
        for sentence in sentences:
            key = self._comparison_key(sentence)
            if any(
                key == self._comparison_key(existing)
                or key in self._comparison_key(existing)
                or self._comparison_key(existing) in key
                for existing in unique_sentences
            ):
                continue
            unique_sentences.append(sentence)
        sentences = unique_sentences
        if not sentences:
            return shorten(text, 280)

        title_words = set(words(item.title))
        corpus_words = Counter(words(text))
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_words = words(sentence)
            if len(sentence_words) < 5:
                continue
            score = 0.0
            score += sum(corpus_words[word] for word in sentence_words if len(word) > 3) / max(
                len(sentence_words), 1
            )
            score += sum(2.0 for word in sentence_words if word in title_words)
            score += sum(2.5 for keyword in AI_SAFETY_KEYWORDS if keyword in sentence.lower())
            if index == 0:
                score += 2.0
            if len(sentence) > 360:
                score -= 1.5
            scored.append((score, index, normalize_whitespace(sentence)))

        if not scored:
            return shorten(text, 280)

        best = sorted(scored, reverse=True)[:2]
        ordered = [sentence for _, _, sentence in sorted(best, key=lambda item: item[1])]
        return shorten(" ".join(ordered), 320)

    def _why_it_matters(self, item: CandidateItem) -> str:
        text = f"{item.title} {item.excerpt} {item.raw_text[:800]}".lower()
        for keyword, template in THEME_TEMPLATES:
            if keyword in text:
                return template
        if item.content_type == "paper":
            return "Adds evidence or tools that other AI safety work can actually build on."
        if item.content_type == "opinion":
            return "Sharpens the case for where attention and effort should go next."
        return "Could change how frontier AI gets built, tested, or governed in practice."

    def _comparison_key(self, sentence: str) -> str:
        return normalize_whitespace(sentence).lower().rstrip(".!?")

    def _polish_summary(self, summary: str) -> str:
        cleaned = normalize_whitespace(summary)
        cleaned = cleaned.replace("This paper", "The paper").replace("This post", "The post")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _polish_why(self, why: str) -> str:
        cleaned = normalize_whitespace(why)
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned
