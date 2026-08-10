"""The Telegram channel: how a queued action reaches a human, and how their
answer gets back.

This module is the same shape as the briefing sender in nightly.py, and for the
same reason. Delivery is NOT an agent capability. Sending happens here, in plain
functions the model cannot call, using a token it never sees — the environment
handed to every MCP server has TELEGRAM_* stripped out of it (agent.py). The
component that can be prompt-injected cannot message anyone; the component that
messages cannot be injected.

Two properties matter more than anything else here:

1. **Only one chat may approve.** A bot username is public and anyone can start
   a conversation with it, so a callback carrying `approve:a1b2c3d4` proves
   nothing on its own. Every update is checked against TELEGRAM_CHAT_ID and
   discarded otherwise. Without that check, the approval gate — the thing the
   whole design rests on — would be bypassable by any stranger who guessed an
   eight-character id.

2. **Queued actions are untrusted text.** What gets rendered into the approval
   message is a tool input, and a tool input can contain content the agent read
   out of an email. So every interpolated value is HTML-escaped and shown inside
   a <pre> block: a note body reading "<b>✅ approved automatically</b>" arrives
   as those literal characters, not as formatting that could dress an attacker's
   payload up as the bot's own words.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import httpx

API = "https://api.telegram.org"

# Telegram caps message text at 4096 characters, and a queued tool input can be
# a whole note. Truncate the input, not the framing around it.
MAX_INPUT_CHARS = 800


def _token() -> str:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")
    return token


def _chat_id() -> str:
    chat = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    if not chat:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set.")
    return chat


def is_configured() -> bool:
    """Telegram is optional. An unconfigured integration is an absent one, not a
    broken one — the same rule the mail server follows."""
    return bool(
        (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        and (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    )


def _call(method: str, **payload: Any) -> dict[str, Any]:
    response = httpx.post(f"{API}/bot{_token()}/{method}", json=payload, timeout=20.0)
    body = response.json()
    if not body.get("ok"):
        # Never include the token; the URL carries it.
        raise RuntimeError(f"Telegram {method} failed: {body.get('description')!r}")
    return body.get("result") or {}


# -- outbound ----------------------------------------------------------------


def send(text: str) -> None:
    """A plain notification with no buttons."""
    _call("sendMessage", chat_id=_chat_id(), text=text, parse_mode="HTML")


def request_approval(entry: dict[str, Any]) -> None:
    """Ask for a decision on one queued action, with the buttons to answer it."""
    tool = html.escape(str(entry["tool"]))
    reason = html.escape(str(entry.get("reason", "")))
    raw = json.dumps(entry.get("input", {}), ensure_ascii=False, indent=2)
    if len(raw) > MAX_INPUT_CHARS:
        raw = raw[:MAX_INPUT_CHARS] + "\n… truncated"

    text = (
        f"🔒 <b>Approval needed</b>\n\n"
        f"<b>{tool}</b>\n{reason}\n\n"
        f"<pre>{html.escape(raw)}</pre>\n"
        f"<code>{html.escape(entry['id'])}</code>"
    )
    _call(
        "sendMessage",
        chat_id=_chat_id(),
        text=text,
        parse_mode="HTML",
        reply_markup={
            "inline_keyboard": [
                [
                    # callback_data is capped at 64 bytes; an 8-char id fits easily.
                    {"text": "✅ Approve", "callback_data": f"ok:{entry['id']}"},
                    {"text": "✖️ Deny", "callback_data": f"no:{entry['id']}"},
                ]
            ]
        },
    )


def acknowledge(callback_id: str, text: str) -> None:
    """Stop the button spinning and show the outcome as a toast."""
    _call("answerCallbackQuery", callback_query_id=callback_id, text=text[:200])


def settle(chat_id: Any, message_id: Any, text: str) -> None:
    """Rewrite the original message so the buttons are gone and the record shows
    what happened. Otherwise an approved action still reads as pending, and the
    buttons stay pressable."""
    _call(
        "editMessageText",
        chat_id=chat_id,
        message_id=message_id,
        text=text,
        parse_mode="HTML",
    )


# -- inbound -----------------------------------------------------------------


def _offset(path: Path) -> int:
    """getUpdates replays everything from the last 24 hours until you confirm
    receipt with an offset. Without persisting it across runs, a stateless
    container would re-execute every approval it had already executed."""
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["offset"])
    except (ValueError, KeyError, TypeError):
        return 0


def _save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"offset": offset}), encoding="utf-8")


def is_authorised(update: dict[str, Any]) -> bool:
    """Did this come from the one chat allowed to approve things?

    Anyone can find a bot and press nothing — but anyone can also send it a
    callback-shaped message, and an eight-character id is not a secret. The
    sender is the only thing that authorises a decision here.
    """
    callback = update.get("callback_query") or {}
    sender = str((callback.get("from") or {}).get("id", ""))
    return bool(sender) and sender == _chat_id()


def poll_decisions(offset_path: Path) -> list[dict[str, Any]]:
    """Collect button presses since the last run.

    Returns one dict per authorised decision: entry_id, verdict ('approve' or
    'deny'), and the callback/message ids needed to answer it.
    """
    updates = _call(
        "getUpdates",
        offset=_offset(offset_path),
        timeout=0,
        allowed_updates=["callback_query"],
    )
    if not isinstance(updates, list):
        return []

    decisions: list[dict[str, Any]] = []
    highest = 0
    for update in updates:
        highest = max(highest, int(update.get("update_id", 0)))
        callback = update.get("callback_query")
        if not callback:
            continue
        if not is_authorised(update):
            # Deliberately silent: do not tell an unknown sender that the id
            # they guessed exists, or that this bot approves anything.
            continue
        data = str(callback.get("data", ""))
        verb, _, entry_id = data.partition(":")
        if verb not in {"ok", "no"} or not entry_id:
            continue
        message = callback.get("message") or {}
        decisions.append(
            {
                "id": entry_id,
                "verdict": "approve" if verb == "ok" else "deny",
                "callback_id": callback.get("id"),
                "chat_id": (message.get("chat") or {}).get("id"),
                "message_id": message.get("message_id"),
            }
        )

    if highest:
        # Confirm receipt of everything seen, including updates we ignored —
        # otherwise an unauthorised press would be replayed on every future run.
        _save_offset(offset_path, highest + 1)
    return decisions
