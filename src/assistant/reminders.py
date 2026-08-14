"""Nudge the phone shortly before a meeting starts.

Deliberately NOT an agent capability. A reminder is arithmetic — read the
calendar, compare two timestamps, send a line of text — and running a language
model every two minutes to do it would be expensive, slower, and less
predictable than the thing it replaced. It also keeps the property the rest of
this system is built on: sending lives here, in a plain function the model
cannot call, so the component that can be prompt-injected still has no way to
message anyone.

Two failure modes shaped the design:

* **Never twice.** Reminded event ids are recorded in the state repo alongside
  the approval queue, so a job that runs every two minutes does not send the
  same nudge fifteen times before the meeting starts.

* **Never a backlog.** Only events starting between now and the lead window are
  considered. If the job stops for a day, it wakes up quiet rather than firing
  thirty reminders for meetings that already happened — the second-worst thing
  a reminder can do, after not arriving.

The lead time must be LONGER than the interval the job runs on, or a meeting
can fall between two runs and be missed entirely: at a fifteen-minute cadence
with a ten-minute lead, an event starting fourteen minutes after a run is too
far away to remind and already over by the next one.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_LEAD_MINUTES = 20

# How long a reminded id is remembered. Long enough that a slow-running job
# cannot forget and re-remind, short enough that the file stays small.
_FORGET_AFTER = timedelta(days=2)


def _lead_minutes() -> int:
    raw = (os.environ.get("ASSISTANT_REMINDER_MINUTES") or "").strip()
    if not raw:
        return DEFAULT_LEAD_MINUTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_LEAD_MINUTES


def _parse(value: str) -> datetime | None:
    """RFC3339 to datetime. 'Z' is spelled out for Python 3.10, which cannot."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in dict(data).items()}
    except Exception:
        # A corrupt file must not stop the meeting reminder AND the approval
        # queue that shares this job. Start over instead.
        return {}


def _save(path: Path, sent: dict[str, str]) -> None:
    now = datetime.now(timezone.utc)
    kept = {}
    for event_id, started in sent.items():
        moment = _parse(started)
        if moment is None or now - moment < _FORGET_AFTER:
            kept[event_id] = started
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept, indent=2), encoding="utf-8")


def due(events: list[dict[str, Any]], already_sent: dict[str, str], lead: int) -> list[dict[str, Any]]:
    """Which of these events should be nudged about right now.

    All-day events are skipped: "in 20 minutes" means nothing for something
    that has no time, and a birthday does not need a nudge.
    """
    if lead <= 0:
        return []
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(minutes=lead)

    upcoming = []
    for event in events:
        event_id = str(event.get("id", ""))
        if not event_id or event.get("status") == "cancelled":
            continue
        start = (event.get("start") or {}).get("dateTime")
        if not start:  # all-day
            continue
        moment = _parse(start)
        # Strictly between now and the horizon: anything already started is
        # water under the bridge, and reminding about it is worse than silence.
        if moment is None or not (now <= moment <= horizon):
            continue
        # Keyed on the id AND the start, never the id alone. Google keeps the
        # id when an event is rescheduled, so an id-only check would suppress
        # the nudge for the NEW time — exactly the case that most needs one, a
        # meeting postponed at the last minute after the first reminder went
        # out. Compared as instants, so a reformatted offset is not a new time.
        if event_id in already_sent:
            seen = _parse(already_sent[event_id])
            if seen is None or seen == moment:
                continue
        upcoming.append(event)
    return upcoming


def _line(event: dict[str, Any], now: datetime) -> str:
    start = _parse((event.get("start") or {}).get("dateTime", "")) or now
    minutes = max(0, round((start - now).total_seconds() / 60))
    # Escaped: the title is written by whoever created the event, which is not
    # necessarily the user, and this is sent with parse_mode=HTML. A '<' in a
    # meeting name would fail the send and lose the reminder with it.
    summary = html.escape((event.get("summary") or "(untitled)").strip())
    location = (event.get("location") or "").strip()
    text = f"⏰ <b>{summary}</b> in {minutes} min ({start.strftime('%H:%M %Z') or start.isoformat()})"
    if location:
        text += f"\n{html.escape(location)}"
    return text


def run(settings: Any) -> list[str]:
    """Send any reminders that are due. Returns one line per reminder sent."""
    lead = _lead_minutes()
    if lead <= 0 or not os.environ.get("GOOGLE_CALENDAR_REFRESH_TOKEN"):
        # An unconfigured integration is an absent one, not a broken one.
        return []

    from . import notify
    from .servers.calendar import server as calendar

    now = datetime.now(timezone.utc)
    payload = calendar._get(
        "/events",
        timeMin=now.isoformat(),
        timeMax=(now + timedelta(minutes=lead)).isoformat(),
        singleEvents="true",
        orderBy="startTime",
        maxResults=20,
    )

    path = settings.reminded_path
    already = _load(path)
    sent: list[str] = []
    for event in due(list(payload.get("items", [])), already, lead):
        notify.send(_line(event, now))
        already[str(event["id"])] = (event.get("start") or {}).get("dateTime", "")
        # Recorded after EACH send, not once at the end. With several meetings
        # due together, a failure on the second would otherwise lose the record
        # of the first, and the next run two minutes later would send it again.
        _save(path, already)
        sent.append(f"reminded: {event.get('summary', event['id'])}")

    return sent
