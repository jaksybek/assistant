"""Terminal entry point: a REPL over the agent, plus approval-queue review."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from .agent import Assistant
from .approvals import load_pending, save_pending
from .config import default_settings

BANNER = """assistant — /pending  /approve <id>  /deny <id>  /mode  /quit"""


async def _repl() -> None:
    settings = default_settings()
    # The scratch server resolves its sandbox from the environment.
    os.environ.setdefault("ASSISTANT_SANDBOX_DIR", str(settings.sandbox_dir))

    async with Assistant(settings) as assistant:
        print(BANNER)
        print(f"model={settings.model}  mode={settings.mode}  sandbox={settings.sandbox_dir}\n")

        while True:
            try:
                line = input("you  ▸ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue

            if line in {"/quit", "/exit"}:
                return
            if line == "/mode":
                settings.mode = "autonomous" if settings.mode == "interactive" else "interactive"
                print(f"       mode → {settings.mode}\n")
                continue
            if line == "/pending":
                _show_pending(settings)
                continue
            if line.startswith(("/approve ", "/deny ")):
                await _resolve_pending(assistant, settings, line)
                continue

            reply = await assistant.send(line)
            print(f"\nclaude ▸ {reply}\n")


def _show_pending(settings) -> None:
    pending = load_pending(settings)
    if not pending:
        print("       nothing awaiting approval\n")
        return
    for entry in pending.values():
        print(f"       {entry['id']}  {entry['tool']}  {entry['input']}")
        print(f"                 {entry['reason']}  (queued {entry['queued_at']})")
    print()


async def _resolve_pending(assistant: Assistant, settings, line: str) -> None:
    verb, _, entry_id = line.partition(" ")
    entry_id = entry_id.strip()
    pending = load_pending(settings)
    entry = pending.get(entry_id)
    if entry is None:
        print(f"       no pending action '{entry_id}'\n")
        return

    del pending[entry_id]
    save_pending(settings, pending)

    if verb == "/deny":
        assistant.audit.record("approval_denied", id=entry_id, tool=entry["tool"])
        print(f"       denied {entry_id}\n")
        return

    # Approved by a human, just now, with the action in front of them — this is
    # the one path allowed to bypass the gate.
    result = await assistant.call_tool_directly(entry["tool"], entry["input"])
    print(f"       executed {entry_id}: {result}\n")


def main() -> None:
    load_dotenv()
    try:
        asyncio.run(_repl())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
