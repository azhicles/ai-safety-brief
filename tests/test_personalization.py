from datetime import datetime, timezone

from ai_safety_brief.models import ChatSettings
from ai_safety_brief.personalization import (
    format_quiet_hours,
    parse_quiet_hours,
    parse_topics_csv,
    within_quiet_hours,
)


def test_parse_topics_csv_validates_and_dedupes():
    assert parse_topics_csv("alignment,security,alignment") == ["alignment", "security"]


def test_parse_quiet_hours_supports_off_and_ranges():
    assert parse_quiet_hours("off") == (None, None)
    assert parse_quiet_hours("23:00-07:00") == ("23:00", "07:00")


def test_within_quiet_hours_handles_overnight_window():
    chat = ChatSettings(
        chat_id=1,
        timezone="Asia/Singapore",
        quiet_hours_start="23:00",
        quiet_hours_end="07:00",
    )
    assert format_quiet_hours(chat) == "23:00-07:00"
    assert within_quiet_hours(chat, datetime(2026, 4, 8, 15, 30, tzinfo=timezone.utc)) is True
    assert within_quiet_hours(chat, datetime(2026, 4, 8, 2, 30, tzinfo=timezone.utc)) is False

