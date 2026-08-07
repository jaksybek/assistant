"""Scheduled entry point for a stateless host (Render, Fly, a container).

The agent's whole persistent state — notes, audit log, pending approvals —
lives in a private git repository rather than on a disk. The container clones
it, runs the sweep, commits whatever changed, and pushes.

That buys three things a mounted disk would not:

  * It survives a host with no persistent storage, which is what makes a cron
    container viable at all. Without this the second brain forgets every night.
  * Every sweep becomes a diff. A note the agent got wrong, or a "fact" a
    prompt injection talked it into recording, shows up in `git log` and
    reverts like any other bad commit.
  * The same repository opens as an Obsidian vault via its Git plugin, so the
    knowledge base syncs to your phone for free.

The token here is scoped to the state repository only — contents:write on that
one repo, nothing else. It is never exposed as a tool, so the agent cannot
reach it; only this module uses it, after the sweep has finished.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

WORKDIR = Path(os.environ.get("ASSISTANT_STATE_DIR", "/tmp/assistant-state"))


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        # Never echo the remote URL — it carries the token.
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def _remote() -> str:
    """Build the authenticated push URL. Kept out of logs and out of git config."""
    repo = (os.environ.get("ASSISTANT_STATE_REPO") or "").strip().strip("/")
    token = (os.environ.get("ASSISTANT_STATE_TOKEN") or "").strip()
    if not (repo and token):
        raise RuntimeError(
            "Set ASSISTANT_STATE_REPO (owner/name) and ASSISTANT_STATE_TOKEN."
        )
    # A secret pasted into a dashboard field is the likeliest thing to be wrong,
    # and every failure mode looks identical from git's error message. Report
    # the shape of what we were handed — never the value.
    print(
        f"[state] repo={repo!r}  "
        f"token: {len(token)} chars, starts {token[:11]!r}, ends {token[-4:]!r}",
        flush=True,
    )
    if token.startswith("PASTE") or "_HERE" in token:
        raise RuntimeError("ASSISTANT_STATE_TOKEN is still the placeholder value.")
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _pull() -> None:
    remote = _remote()
    if (WORKDIR / ".git").exists():
        _git("remote", "set-url", "origin", remote, cwd=WORKDIR)
        _git("fetch", "--depth", "1", "origin", "main", cwd=WORKDIR)
        _git("reset", "--hard", "origin/main", cwd=WORKDIR)
    else:
        WORKDIR.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", remote, str(WORKDIR))
    _git("config", "user.name", "assistant", cwd=WORKDIR)
    _git("config", "user.email", "assistant@localhost", cwd=WORKDIR)


def _push() -> str:
    if not _git("status", "--porcelain", cwd=WORKDIR):
        return "no changes to commit"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _git("add", "-A", cwd=WORKDIR)
    _git("commit", "-m", f"Sweep {stamp}", cwd=WORKDIR)
    _git("push", "origin", "HEAD:main", cwd=WORKDIR)
    return _git("log", "-1", "--stat", "--format=%h %s", cwd=WORKDIR)


def main() -> None:
    load_dotenv()
    _pull()

    # Point the agent at the checkout. Set before anything reads Settings,
    # which resolve these at construction.
    os.environ["ASSISTANT_DATA_DIR"] = str(WORKDIR)
    os.environ.setdefault("ASSISTANT_SANDBOX_DIR", str(WORKDIR / "sandbox"))

    from .nightly import main as sweep

    try:
        sweep()
    finally:
        # Push whatever the sweep managed to record, even if it then failed.
        # Losing a night's notes to an unrelated error would be worse than a
        # partial commit.
        print(_push())


if __name__ == "__main__":
    sys.exit(main())
