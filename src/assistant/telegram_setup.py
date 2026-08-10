"""One-time helper: find your Telegram chat id.

TELEGRAM_CHAT_ID is not just a delivery address — it is the allowlist. It is the
only thing standing between "a human approved this" and "a stranger pressed a
button", so it is worth getting right without guesswork or copying numbers out
of a screenshot.

    1. Message @BotFather, /newbot, put the token in .env as TELEGRAM_BOT_TOKEN
    2. Send your new bot any message, so it is allowed to reply to you
    3. uv run assistant-telegram-setup

Prints the line to paste into .env. Never prints the token.
"""

from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

from .notify import API


def main() -> None:
    load_dotenv()
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env first (get it from @BotFather).")
        raise SystemExit(1)

    me = httpx.get(f"{API}/bot{token}/getMe", timeout=20.0).json()
    if not me.get("ok"):
        print(f"The token was rejected: {me.get('description')!r}")
        raise SystemExit(1)
    print(f"bot: @{me['result'].get('username')}")

    updates = httpx.get(f"{API}/bot{token}/getUpdates", timeout=20.0).json()
    chats = {}
    for update in updates.get("result", []):
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        if chat.get("id"):
            name = chat.get("username") or chat.get("first_name") or chat.get("title") or "?"
            chats[chat["id"]] = name

    if not chats:
        print(
            "\nNo messages yet. Open Telegram, send your bot any message "
            "(just 'hi'), then run this again."
        )
        raise SystemExit(1)

    print("\nAdd this to .env:\n")
    for chat_id, name in chats.items():
        print(f"TELEGRAM_CHAT_ID={chat_id}    # {name}")
    if len(chats) > 1:
        print("\nSeveral chats have messaged the bot — pick your own.")


if __name__ == "__main__":
    sys.exit(main())
