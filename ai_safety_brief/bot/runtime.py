"""Runtime container shared by handlers and scheduler."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ai_safety_brief.config import Settings
from ai_safety_brief.db import Database
from ai_safety_brief.services import DigestPipeline, SourceCollector


@dataclass
class Runtime:
    settings: Settings
    db: Database
    collector: SourceCollector
    pipeline: DigestPipeline
    locks: dict[int, asyncio.Lock] = field(default_factory=dict)

    def chat_lock(self, chat_id: int) -> asyncio.Lock:
        if chat_id not in self.locks:
            self.locks[chat_id] = asyncio.Lock()
        return self.locks[chat_id]

