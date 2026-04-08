"""Summarization helpers."""

from __future__ import annotations

from collections import Counter

from ai_safety_brief.models import CandidateItem
from ai_safety_brief.sources import AI_SAFETY_KEYWORDS
from ai_safety_brief.utils.text import normalize_whitespace, shorten, split_sentences, words

THEME_TEMPLATES = (
    ("governance", "This matters because it could influence how frontier models are governed or deployed."),
    ("policy", "This matters because it changes the policy environment around advanced AI systems."),
    ("eval", "This matters because better evaluations make dangerous model behavior easier to detect before deployment."),
    ("red team", "This matters because it improves how teams probe models for failure modes and misuse risks."),
    ("interpretability", "This matters because it could make model behavior easier to inspect, explain, or control."),
    ("oversight", "This matters because it strengthens how humans supervise powerful systems at scale."),
    ("alignment", "This matters because it bears directly on aligning advanced systems with human goals."),
    ("control", "This matters because it informs concrete ways to keep increasingly capable systems within safe bounds."),
    ("robustness", "This matters because it affects whether systems remain dependable under stress or adversarial pressure."),
)


class Summarizer:
    def summarize(self, item: CandidateItem) -> tuple[str, str]:
        summary = self._extractive_summary(item)
        why = self._why_it_matters(item)
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
            return "This matters because it adds evidence or methods that other AI safety work can build on."
        if item.content_type == "opinion":
            return "This matters because it sharpens the arguments people are using to prioritize AI safety work."
        return "This matters because it could change how frontier AI is built, evaluated, or governed."

    def _comparison_key(self, sentence: str) -> str:
        return normalize_whitespace(sentence).lower().rstrip(".!?")
