"""Summarization helpers."""

from __future__ import annotations

from collections import Counter

from ai_safety_brief.models import CandidateItem
from ai_safety_brief.sources import AI_SAFETY_KEYWORDS
from ai_safety_brief.utils.text import (
    lowercase_sentence_start,
    normalize_whitespace,
    shorten,
    split_sentences,
    words,
)

THEME_TEMPLATES = (
    (
        "governance",
        "what stands out is that it turns safety into real deployment choices, audits, and rules instead of vague good intentions.",
    ),
    (
        "policy",
        "it could quietly shape the rules of the road for advanced ai, and that often matters more than the splashier headlines.",
    ),
    (
        "eval",
        "what makes this feel useful is that it turns safety into something teams can actually test before they deploy.",
    ),
    (
        "red team",
        "this feels useful in a grounded way: better probing means a better shot at catching ugly surprises early.",
    ),
    (
        "interpretability",
        "what makes this notable is that it could make model behavior a little less opaque and a lot easier to inspect.",
    ),
    (
        "oversight",
        "this is the kind of work that helps humans stay meaningfully in the loop instead of just nominally in charge.",
    ),
    (
        "alignment",
        "it stays close to the real alignment problem, not just the easier edges around it.",
    ),
    (
        "control",
        "it pushes on the unglamorous but essential question of how to keep capable systems inside safer bounds.",
    ),
    (
        "robustness",
        "it is trying to answer a very practical question: do these systems stay dependable once the pressure goes up?",
    ),
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
            return "if it holds up, it gives the field something more solid to build on than intuition alone."
        if item.content_type == "opinion":
            return "even when it is speculative, it can still sharpen where attention and effort should go next."
        return "what makes this worth watching is that it could change how frontier ai gets built, tested, or governed in practice."

    def _comparison_key(self, sentence: str) -> str:
        return normalize_whitespace(sentence).lower().rstrip(".!?")

    def _polish_summary(self, summary: str) -> str:
        cleaned = normalize_whitespace(summary)
        cleaned = cleaned.replace("This paper", "a new paper").replace("This post", "a new post")
        cleaned = cleaned.replace("The paper studies", "a new paper looks at")
        cleaned = cleaned.replace("The paper shows", "a new paper shows")
        cleaned = cleaned.replace("The paper introduces", "a new paper introduces")
        cleaned = cleaned.replace("The post argues", "a new post argues")
        cleaned = cleaned.replace("The post lays out", "a new post lays out")
        cleaned = lowercase_sentence_start(cleaned)
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    def _polish_why(self, why: str) -> str:
        cleaned = normalize_whitespace(why)
        cleaned = lowercase_sentence_start(cleaned)
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned
