from datetime import datetime, timezone

from ai_safety_brief.models import ChatSettings
from ai_safety_brief.utils.time import compute_next_run


def test_compute_next_run_daily():
    chat = ChatSettings(chat_id=1, timezone="Asia/Singapore", send_hour=19, send_minute=0)
    now = datetime(2026, 4, 8, 8, 30, tzinfo=timezone.utc)
    next_run = compute_next_run(chat, now)
    assert next_run.isoformat() == "2026-04-08T11:00:00+00:00"


def test_compute_next_run_hourly():
    chat = ChatSettings(
        chat_id=1,
        timezone="UTC",
        schedule_type="hourly",
        schedule_value="6",
        send_minute=15,
        last_digest_at=datetime(2026, 4, 8, 2, 0, tzinfo=timezone.utc),
    )
    next_run = compute_next_run(chat, datetime(2026, 4, 8, 3, 0, tzinfo=timezone.utc))
    assert next_run.isoformat() == "2026-04-08T08:00:00+00:00"


def test_compute_next_run_weekly():
    chat = ChatSettings(
        chat_id=1,
        timezone="UTC",
        schedule_type="weekly",
        schedule_value="Mon,Wed,Fri",
        send_hour=10,
        send_minute=30,
    )
    now = datetime(2026, 4, 8, 11, 0, tzinfo=timezone.utc)
    next_run = compute_next_run(chat, now)
    assert next_run.isoformat() == "2026-04-10T10:30:00+00:00"

