"""Optional free-tier LLM refinement."""

from __future__ import annotations

import json
import logging

from groq import AsyncGroq

from ai_safety_brief.models import CandidateItem

logger = logging.getLogger(__name__)


class GroqRefiner:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key) if api_key else None
        self._model = model

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def maybe_refine(self, items: list[CandidateItem]) -> list[CandidateItem]:
        if not self._client or not items:
            return items

        prompt_payload = [
            {
                "index": index,
                "title": item.title,
                "content_type": item.content_type,
                "summary": item.summary,
                "why_it_matters": item.why_it_matters,
            }
            for index, item in enumerate(items)
        ]
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite AI safety news summaries for clarity. Return JSON only: "
                            '[{"index":0,"summary":"...","why_it_matters":"..."}]. Keep each field under 220 characters.'
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload)},
                ],
            )
            message = response.choices[0].message.content or "[]"
            updates = json.loads(message)
        except Exception as exc:
            logger.warning("Groq refinement failed, falling back to extractive summaries: %s", exc)
            return items

        by_index = {int(entry["index"]): entry for entry in updates if "index" in entry}
        for index, item in enumerate(items):
            refined = by_index.get(index)
            if not refined:
                continue
            item.summary = refined.get("summary", item.summary) or item.summary
            item.why_it_matters = refined.get("why_it_matters", item.why_it_matters) or item.why_it_matters
        return items

