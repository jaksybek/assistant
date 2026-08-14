"""The chat end of the assistant: button presses, and messages as instructions.

The nightly sweep runs on a stateless container that exits when it is done, so
there is nobody left listening when a button is finally pressed. This is the
other half of the loop, and it is deliberately a separate short job rather than
a long-lived service: it wakes up, asks Telegram what was decided since last
time, acts on it, pushes the state, and exits. Run it on whatever interval you
want approvals to land on.

The bypass here is narrow and intentional. `call_tool_directly` skips the
approval gate — it is the one path allowed to, because a human has just looked
at the action and said yes. Everything that makes that safe happens before it:
the decision must arrive from the one authorised chat (notify.is_authorised),
and it must match an id that the gate itself queued. An id that is not in the
pending file executes nothing.

The same job also carries instructions. A message in the chat is run as a
prompt and answered there, which is what makes the phone a usable front end:
dictate into the message box, and the text you already read is what the agent
acts on. That rests on the same allowlist — a bot username is public, so
without it a stranger would be prompting an agent that can read the mailbox
and the whole vault. Instructions are trusted because of who sent them; what
the agent reads while carrying one out is not, and taint handles that.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from . import notify
from .approvals import load_pending, save_pending
from .audit import AuditLog
from .config import default_settings


def _summarise(entry: dict) -> str:
    return f"{entry['tool']}  {entry.get('reason', '')}".strip()


async def _apply(decisions: list[dict]) -> list[str]:
    settings = default_settings()
    audit = AuditLog(settings.audit_path)
    pending = load_pending(settings)
    outcomes: list[str] = []

    # Resolve every decision against the queue first, so the expensive part —
    # starting the MCP servers — only happens if something actually needs to run.
    approved, resolved = [], []
    for decision in decisions:
        entry = pending.get(decision["id"])
        if entry is None:
            # Already handled, or an id that was never queued. Say so plainly
            # rather than silently doing nothing.
            audit.record("approval_stale", id=decision["id"], verdict=decision["verdict"])
            _settle(decision, "This action is no longer pending.", "already handled")
            outcomes.append(f"{decision['id']}: no longer pending")
            continue
        if decision["verdict"] == "deny":
            audit.record("approval_denied", id=entry["id"], tool=entry["tool"], channel="telegram")
            _settle(decision, f"✖️ Denied\n\n{_summarise(entry)}", "denied")
            outcomes.append(f"{decision['id']}: denied")
            resolved.append(decision["id"])
        else:
            approved.append((decision, entry))

    if approved:
        from .agent import Assistant

        async with Assistant(settings) as assistant:
            for decision, entry in approved:
                try:
                    result = await assistant.call_tool_directly(entry["tool"], entry["input"])
                    audit.record(
                        "approved_execution_channel",
                        id=entry["id"],
                        tool=entry["tool"],
                        channel="telegram",
                    )
                    _settle(decision, f"✅ Done\n\n{_summarise(entry)}", "done")
                    outcomes.append(f"{decision['id']}: executed — {result[:120]}")
                    resolved.append(decision["id"])
                except Exception as exc:
                    # The approval stands; only the execution failed. Leave it in
                    # the queue so it can be retried rather than silently lost.
                    audit.record("approved_execution_failed", id=entry["id"], error=str(exc))
                    _settle(decision, f"⚠️ Failed\n\n{_summarise(entry)}\n\n{exc}", "failed")
                    outcomes.append(f"{decision['id']}: FAILED — {exc}")

    if resolved:
        pending = load_pending(settings)
        for entry_id in resolved:
            pending.pop(entry_id, None)
        save_pending(settings, pending)

    return outcomes


async def _answer(instructions: list[dict]) -> list[str]:
    """Run what the chat asked for, and reply with the result.

    This is the whole of "voice control": dictation happens on the device, so
    what arrives here is text the user has already seen and could correct. The
    agent never hears audio, and so never acts on words nobody read.

    Autonomous mode, deliberately. A human sent the message, but no human is at
    a terminal to answer a prompt — so gated actions queue and come back as
    buttons in the same chat, rather than blocking a job that is about to exit.

    The instruction itself is trusted: it passed the sender allowlist, so it
    came from the one chat allowed to drive this bot. What the agent then READS
    while carrying it out is not, and the taint model handles that exactly as
    it does during a sweep.
    """
    settings = default_settings()
    settings.mode = "autonomous"
    outcomes: list[str] = []

    from .agent import Assistant

    async with Assistant(settings) as assistant:
        for instruction in instructions:
            if instruction.get("unsupported") == "voice":
                notify.send_reply(
                    "I can't listen to voice notes. Dictate into the message box "
                    "instead — then you see the text before it reaches me."
                )
                outcomes.append("voice note: declined")
                continue
            try:
                reply = await assistant.send(instruction["text"])
                notify.send_reply(reply)
                outcomes.append(f"answered: {instruction['text'][:60]}")
            except Exception as exc:
                # Never leave a message unanswered: silence in a chat is
                # indistinguishable from the bot being dead.
                notify.send_reply(f"That failed: {type(exc).__name__}: {exc}")
                outcomes.append(f"FAILED: {exc}")

    return outcomes


def _settle(decision: dict, text: str, toast: str) -> None:
    """Close the loop in the chat. Never fatal: the action has already happened
    (or not), and failing to redraw a message must not change that.

    The two calls are guarded separately, and that separation is the point.
    `answerCallbackQuery` is a toast on a live button press, and Telegram expires
    the query within seconds — while this job wakes on a fifteen-minute cron, so
    by design it almost always arrives too late. `editMessageText` never expires
    and is the half that matters: it strips the buttons and records the outcome.
    Sharing one try block let the doomed call skip the useful one, leaving every
    executed action displaying live Approve/Deny buttons and reading as pending.
    """
    if decision.get("callback_id"):
        try:
            notify.acknowledge(decision["callback_id"], toast)
        except Exception:
            # Expected whenever the press is older than the query's lifetime,
            # which on a cron is nearly always. Not worth reporting.
            pass

    if decision.get("chat_id") and decision.get("message_id"):
        try:
            notify.settle(decision["chat_id"], decision["message_id"], text)
        except Exception as exc:
            print(f"[approve] could not update the message: {exc}", flush=True)


def main() -> None:
    load_dotenv()
    if not notify.is_configured():
        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. "
            "Run `assistant-telegram-setup` to find the chat id."
        )

    # On a stateless host the queue lives in the state repo, so it has to be
    # fetched before it can be read and pushed after it changes. Locally there
    # is no state repo and the files are simply on disk.
    stateful = bool(os.environ.get("ASSISTANT_STATE_REPO"))
    if stateful:
        from .cron import WORKDIR, pull_state, push_state

        pull_state()
        os.environ.setdefault("ASSISTANT_DATA_DIR", str(WORKDIR))
        os.environ.setdefault("ASSISTANT_SANDBOX_DIR", str(WORKDIR / "sandbox"))

    settings = default_settings()
    updates = notify.poll_updates(settings.telegram_offset_path)
    decisions, instructions = updates["decisions"], updates["instructions"]

    if not decisions and not instructions:
        print("nothing waiting")
        # The offset may still have moved, so push before leaving.
        if stateful:
            from .cron import push_state

            print(push_state())
        return

    try:
        # Decisions first: an approval the user already gave should not wait
        # behind a question they asked afterwards.
        if decisions:
            for line in asyncio.run(_apply(decisions)):
                print(line)
        if instructions:
            for line in asyncio.run(_answer(instructions)):
                print(line)
    finally:
        if stateful:
            from .cron import push_state

            print(push_state())


if __name__ == "__main__":
    sys.exit(main())
