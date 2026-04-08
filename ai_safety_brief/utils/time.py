"""Time and schedule helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ai_safety_brief.models import ChatSettings

WEEKDAY_LOOKUP = {
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timezone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def parse_weekdays(value: str) -> list[int]:
    if not value:
        return []
    days: list[int] = []
    for part in value.split(","):
        key = part.strip().lower()[:3]
        if key not in WEEKDAY_LOOKUP:
            raise ValueError(f"Unknown weekday: {part}")
        days.append(WEEKDAY_LOOKUP[key])
    return sorted(set(days))


def compute_next_run(chat: ChatSettings, now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or utc_now()
    local_tz = parse_timezone(chat.timezone)
    local_now = now_utc.astimezone(local_tz)

    if chat.schedule_type == "hourly":
        interval = max(1, int(chat.schedule_value or "24"))
        if chat.last_digest_at is not None:
            return chat.last_digest_at + timedelta(hours=interval)
        aligned = local_now.replace(minute=chat.send_minute, second=0, microsecond=0)
        if aligned <= local_now:
            aligned += timedelta(hours=interval)
        return aligned.astimezone(timezone.utc)

    target = local_now.replace(
        hour=chat.send_hour,
        minute=chat.send_minute,
        second=0,
        microsecond=0,
    )

    if chat.schedule_type == "daily":
        if target <= local_now:
            target += timedelta(days=1)
        return target.astimezone(timezone.utc)

    weekdays = parse_weekdays(chat.schedule_value)
    if not weekdays:
        weekdays = [target.weekday()]

    for offset in range(8):
        candidate = target + timedelta(days=offset)
        if candidate.weekday() not in weekdays:
            continue
        if offset == 0 and candidate <= local_now:
            continue
        return candidate.astimezone(timezone.utc)

    return (target + timedelta(days=7)).astimezone(timezone.utc)


def format_schedule(chat: ChatSettings) -> str:
    if chat.schedule_type == "hourly":
        return f"Every {chat.schedule_value or '24'} hour(s) at minute {chat.send_minute:02d}"
    if chat.schedule_type == "weekly":
        days = chat.schedule_value or "Mon"
        return f"Weekly on {days} at {chat.send_hour:02d}:{chat.send_minute:02d}"
    return f"Daily at {chat.send_hour:02d}:{chat.send_minute:02d}"

