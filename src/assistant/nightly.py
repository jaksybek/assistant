"""Unattended sweep: triage the mailbox, log it, and email you the briefing.

Run by cron. There is no human at the keyboard, so it runs in autonomous mode —
gated actions queue for review instead of blocking, and the agent carries on.

The delivery half is deliberately NOT an agent capability. Sending happens here,
in a plain function the model cannot call, using credentials it never sees. The
component that can be prompt-injected has no ability to send; the component that
sends cannot be injected. That is why there is still no send tool anywhere in
the mail server, and why adding one would be a real loosening rather than a
convenience.
"""

from __future__ import annotations

import asyncio
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage

from dotenv import load_dotenv

from .agent import Assistant
from .approvals import load_pending
from .config import default_settings

SWEEP = (
    "Read the note 'todo' if it exists, and surface anything due on or before "
    "today at the TOP of your reply under the heading 'Needs you today'. Say "
    "nothing about items not yet due.\n\n"
    "Then sweep my mail from the last 24 hours. Search your notes first for "
    "context on anything recurring or already settled, so you do not re-raise "
    "questions I have answered. Tell me what actually needs me and what is "
    "noise, and append the triage to the note 'assistant/mail/inbox-log'. Be "
    "brief and concrete."
)


def _send(subject: str, body: str) -> None:
    """Email the briefing. Never exposed as a tool."""
    user = os.environ["MAIL_IMAP_USER"]
    # Same non-breaking-space hazard as IMAP; see servers/mail/server.py.
    password = "".join(c for c in os.environ["MAIL_IMAP_PASSWORD"] if not c.isspace())
    recipient = os.environ.get("MAIL_DIGEST_TO")
    if not recipient:
        raise RuntimeError("Set MAIL_DIGEST_TO in .env to receive the briefing.")

    message = EmailMessage()
    message["From"] = user
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    host = os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("MAIL_SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)


async def _run() -> str:
    settings = default_settings()
    # Nobody is watching: queue gated actions rather than waiting on a prompt.
    settings.mode = "autonomous"

    async with Assistant(settings) as assistant:
        briefing = await assistant.send(SWEEP)

    pending = load_pending(settings)
    if pending:
        lines = "\n".join(
            f"  {e['id']}  {e['tool']}  {e['reason']}" for e in pending.values()
        )
        from . import notify

        where = (
            "Approve or deny them in Telegram"
            if notify.is_configured()
            else "Run `assistant` and use /pending to review"
        )
        briefing += (
            f"\n\n---\n{len(pending)} action(s) waiting on your approval. "
            f"{where}:\n{lines}"
        )
    return briefing


def main() -> None:
    load_dotenv()
    today = datetime.now(timezone.utc).strftime("%a %d %b")
    try:
        briefing = asyncio.run(_run())
    except Exception as exc:
        # A failed sweep must still reach you — silence is the worst outcome
        # for something that runs unattended.
        _send(f"Assistant briefing FAILED — {today}", f"{type(exc).__name__}: {exc}")
        raise
    _send(f"Morning briefing — {today}", briefing)
    print(f"sent briefing ({len(briefing)} chars)")


if __name__ == "__main__":
    sys.exit(main())
