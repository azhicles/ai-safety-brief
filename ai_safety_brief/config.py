"""Runtime configuration for AI Safety Brief."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    groq_api_key: str
    groq_model: str
    data_dir: Path
    db_path: Path
    default_timezone: str
    default_top_k: int
    default_send_hour: int
    default_send_minute: int
    default_repeat_window_days: int
    lookback_hours: int
    scheduler_poll_seconds: int
    item_fetch_timeout_seconds: int
    x_rss_base_url: str
    x_accounts: tuple[str, ...]
    max_candidate_fetch: int = 40
    item_preview_chars: int = 1200
    digest_message_limit: int = 3500


def load_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    db_path = Path(os.getenv("DB_PATH", str(data_dir / "bot.db")))
    accounts = tuple(
        account.strip()
        for account in os.getenv("X_ACCOUNTS", "").split(",")
        if account.strip()
    )
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        data_dir=data_dir,
        db_path=db_path,
        default_timezone=os.getenv("DEFAULT_TIMEZONE", "Asia/Singapore"),
        default_top_k=_env_int("DEFAULT_TOP_K", 5),
        default_send_hour=_env_int("DEFAULT_SEND_HOUR", 19),
        default_send_minute=_env_int("DEFAULT_SEND_MINUTE", 0),
        default_repeat_window_days=_env_int("DEFAULT_REPEAT_WINDOW_DAYS", 7),
        lookback_hours=_env_int("LOOKBACK_HOURS", 72),
        scheduler_poll_seconds=_env_int("SCHEDULER_POLL_SECONDS", 60),
        item_fetch_timeout_seconds=_env_int("ITEM_FETCH_TIMEOUT_SECONDS", 20),
        x_rss_base_url=os.getenv("X_RSS_BASE_URL", "").rstrip("/"),
        x_accounts=accounts,
    )

