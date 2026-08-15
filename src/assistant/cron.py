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


def pull_state() -> None:
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


def push_state() -> str:
    if not _git("status", "--porcelain", cwd=WORKDIR):
        return "no changes to commit"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _git("add", "-A", cwd=WORKDIR)
    _git("commit", "-m", f"Sweep {stamp}", cwd=WORKDIR)
    _git("push", "origin", "HEAD:main", cwd=WORKDIR)
    return _git("log", "-1", "--stat", "--format=%h %s", cwd=WORKDIR)


def _preflight() -> None:
    """Report the shape of every credential before using any of them.

    Secrets are set by hand in a dashboard, and a mis-paste is the single most
    likely failure. Worse, each one surfaces as a different opaque error deep in
    a library — a git auth failure, an IMAP login refusal, an SMTP 535 — so
    diagnosing them one deploy at a time is slow. Print the shape of all of
    them up front, values never included.
    """
    expected = {
        "ANTHROPIC_API_KEY": "sk-ant-",
        "MAIL_IMAP_HOST": "",
        "MAIL_IMAP_USER": "",
        "MAIL_IMAP_PASSWORD": "",
        "MAIL_DIGEST_TO": "",
        "ASSISTANT_STATE_REPO": "",
        "ASSISTANT_STATE_TOKEN": "github_pat_",
    }
    problems = []
    for name, prefix in expected.items():
        raw = os.environ.get(name)
        if not raw:
            problems.append(f"{name} is not set")
            print(f"[preflight] {name:24} MISSING", flush=True)
            continue
        value = raw.strip()
        note = ""
        if value != raw:
            note = "  (had surrounding whitespace)"
        if value.endswith("...") or "_HERE" in value or value.startswith("PASTE"):
            note += "  <-- looks truncated or still a placeholder"
            problems.append(f"{name} looks truncated or is a placeholder")
        elif prefix and not value.startswith(prefix):
            note += f"  <-- expected it to start {prefix!r}"
            problems.append(f"{name} has an unexpected prefix")
        shown = value if name in ("MAIL_IMAP_HOST", "MAIL_IMAP_USER", "MAIL_DIGEST_TO",
                                  "ASSISTANT_STATE_REPO") else f"{len(value)} chars"
        print(f"[preflight] {name:24} {shown}{note}", flush=True)

    if problems:
        raise RuntimeError("Bad configuration: " + "; ".join(problems))

    _check_logins()
    _check_drive()
    _check_calendar()


def _check_calendar() -> None:
    """Prove the calendar credentials work before the sweep starts.

    This exists because they stopped working and nobody found out for a day.
    Google began answering `401 invalid_client`; the sweep carried on, the
    briefing was produced and looked complete, and the only symptom was an
    absence — no schedule at the top, no meeting reminders, and an approved
    calendar change that would have failed at the moment it was least expected.
    Exactly the failure `_check_drive` was written for, on the other integration.

    The rules are the same, for the same reasons:

    * Not configured is not broken. The server registry starts the calendar only
      when all three credentials are present, so their absence is a choice.

    * Half-configured IS broken, and silently: two of three means no calendar
      server, no schedule section, and a briefing that reads as complete while
      missing the part it was meant to open with.

    * A refused credential is fatal — it will not fix itself, and the failure
      mail is how it gets noticed. A network blip is not: mail is the point of
      the sweep, and the calendar is not worth losing a morning briefing over.

      That last line is easy to write and easy to get wrong, and the first cut
      of this function did: `httpx.post` RETURNS for 429 and 503 instead of
      raising, so a throttled Google arrived as the same exception as a wrong
      client secret, and the blip aborted the sweep after all. Hence
      TokenRefused carrying the status — see oauth.py.
    """
    names = ("GOOGLE_CALENDAR_CLIENT_ID", "GOOGLE_CALENDAR_CLIENT_SECRET",
             "GOOGLE_CALENDAR_REFRESH_TOKEN")
    present = [name for name in names if (os.environ.get(name) or "").strip()]

    if not present:
        return

    if len(present) < len(names):
        missing = ", ".join(name for name in names if name not in present)
        raise RuntimeError(
            f"Calendar is half-configured: {missing} not set. The calendar server "
            "will not start, so the briefing loses its schedule and no approved "
            "calendar change can run. Set them, or clear all three."
        )

    import httpx

    from .oauth import TokenRefused
    from .servers.calendar import server as calendar

    try:
        calendar._access_token()
    except httpx.HTTPError as exc:
        # Transport-level: Google unreachable. Say so and let the sweep run.
        print(f"[preflight] Calendar token      UNREACHABLE ({exc}) — continuing", flush=True)
        return
    except TokenRefused as exc:
        # A REPLY from Google, which httpx does not raise on — 429 and 503 arrive
        # here looking exactly like a wrong secret. Only the status tells them
        # apart, and getting it wrong means a rate limit costs the briefing.
        if exc.transient:
            print(
                f"[preflight] Calendar token      Google returned {exc.status} — continuing",
                flush=True,
            )
            return
        raise RuntimeError(f"Calendar credentials rejected. {exc}") from None
    except Exception as exc:
        # Anything else — a malformed key, a missing field in the reply. Will not
        # fix itself, so it stops the run.
        raise RuntimeError(f"Calendar credentials rejected. {exc}") from None

    print("[preflight] Calendar token      ok", flush=True)


def _check_drive() -> None:
    """Prove the recordings folder is reachable before the sweep starts.

    This exists because of how the Drive integration failed the first time it
    ran for real: every token exchange returned 400, the agent treated it as an
    ordinary tool error and carried on, and the job still reported success. A
    dead integration looked exactly like a working one with nothing to do — it
    could have stayed broken for weeks, visible only as notes never appearing.

    Two deliberate choices:

    * It fetches the FOLDER, not a listing of it. A listing cannot tell "no
      recordings yet" from "shared with the wrong address" — both come back
      empty. Fetching the folder answers 404 when the service account cannot
      see it, and that is the likeliest mistake in the whole setup: every other
      step is visible in a dashboard, but a share with a mistyped address looks
      identical to a correct one.

    * Configuration failures are fatal; transient ones are not. A wrong key or
      an unshared folder will not fix itself and must stop the run — the failure
      mail is how it reaches him. But Drive is optional and mail is not, so a
      network blip must never cost the morning briefing.
    """
    key = os.environ.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
    folder = (os.environ.get("GOOGLE_DRIVE_FOLDER_ID") or "").strip()

    # Not configured is not broken — the same rule the server registry follows.
    if not key and not folder:
        return

    # Half-configured IS broken, and silently so: config.py starts the server
    # only when both halves are present, so the agent would simply have no
    # recordings and never say why.
    if not (key and folder):
        missing = "GOOGLE_DRIVE_FOLDER_ID" if key else "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"
        raise RuntimeError(
            f"Drive is half-configured: {missing} is not set, so the Drive server will "
            "not start and no recording will ever be read. Set it, or clear both."
        )

    import urllib.parse

    import httpx

    from .oauth import TokenRefused
    from .servers.drive.server import _access_token, _service_account

    try:
        address = _service_account().get("client_email", "the service account")
        token = _access_token()
    except httpx.HTTPError as exc:
        print(f"[preflight] Drive token         UNREACHABLE ({exc}) — continuing", flush=True)
        return
    except TokenRefused as exc:
        # Same trap as the calendar: httpx returns a 429 or a 503 rather than
        # raising, so without the status a bad five minutes at Google is
        # indistinguishable from a deleted key — and kills the briefing.
        if exc.transient:
            print(
                f"[preflight] Drive token         Google returned {exc.status} — continuing",
                flush=True,
            )
            return
        raise RuntimeError(
            f"Drive credentials rejected: {exc}. The service-account key is wrong, "
            "malformed, or has been deleted in the Google Cloud console."
        ) from None
    except Exception as exc:
        raise RuntimeError(
            f"Drive credentials rejected: {exc}. The service-account key is wrong, "
            "malformed, or has been deleted in the Google Cloud console."
        ) from None
    print("[preflight] Drive token         ok", flush=True)

    try:
        response = httpx.get(
            f"https://www.googleapis.com/drive/v3/files/{urllib.parse.quote(folder, safe='')}",
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "id,name", "supportsAllDrives": "true"},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        # Transient. Say so and let the sweep run: the mail half still works, and
        # a briefing without recordings beats no briefing at all.
        print(f"[preflight] Drive folder        UNREACHABLE ({exc}) — continuing", flush=True)
        return

    if response.status_code == 200:
        print(f"[preflight] Drive folder        {response.json().get('name')!r} ok", flush=True)
        return

    if response.status_code in (403, 404):
        raise RuntimeError(
            f"The Drive folder {folder} is not visible to {address}. Share that one "
            "folder with that address (Viewer is enough), or the id is wrong. Until "
            "then no recording can be read, however well everything else works."
        )

    if response.status_code >= 500:
        print(
            f"[preflight] Drive folder        Google returned {response.status_code}"
            " — continuing",
            flush=True,
        )
        return

    raise RuntimeError(
        f"Drive folder check failed ({response.status_code}): {response.text[:200]}"
    )


def _check_logins() -> None:
    """Actually log in to the mailbox before doing any work.

    Shape checks cannot catch a credential that is the right length and simply
    wrong — which is precisely what happened: a mistyped 16-character app
    password passed every static check, then burned a full sweep before failing,
    and the failure notification could not be sent either, because it needed the
    same broken credential. Prove both logins work first; a wrong password now
    costs seconds and says so plainly.
    """
    import imaplib
    import smtplib

    user = os.environ["MAIL_IMAP_USER"]
    # .strip() misses spaces BETWEEN the groups of a Gmail app password.
    password = "".join(c for c in os.environ["MAIL_IMAP_PASSWORD"] if not c.isspace())

    try:
        imap = imaplib.IMAP4_SSL(os.environ.get("MAIL_IMAP_HOST", "imap.gmail.com"), 993)
        imap.login(user, password)
        imap.logout()
        print("[preflight] IMAP login          ok", flush=True)
    except Exception as exc:
        raise RuntimeError(
            f"IMAP login failed for {user}: {exc}. The app password is wrong or revoked."
        ) from None

    try:
        smtp = smtplib.SMTP_SSL(
            os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com"),
            int(os.environ.get("MAIL_SMTP_PORT", "465")),
        )
        smtp.login(user, password)
        smtp.quit()
        print("[preflight] SMTP login          ok", flush=True)
    except Exception as exc:
        raise RuntimeError(
            f"SMTP login failed for {user}: {exc}. The briefing could not be delivered."
        ) from None


def main() -> None:
    load_dotenv()
    _preflight()
    pull_state()

    # Point the agent at the checkout, unless the environment already says
    # where to look. setdefault rather than assignment: when the state repo is
    # the user's own Obsidian vault, the sandbox is the repo ROOT (so the agent
    # sees every note) and the audit log belongs in a subfolder rather than
    # scattered at the top of a knowledge base someone else reads.
    os.environ.setdefault("ASSISTANT_DATA_DIR", str(WORKDIR))
    os.environ.setdefault("ASSISTANT_SANDBOX_DIR", str(WORKDIR / "sandbox"))

    from .nightly import main as sweep

    try:
        sweep()
    finally:
        # Push whatever the sweep managed to record, even if it then failed.
        # Losing a night's notes to an unrelated error would be worse than a
        # partial commit.
        print(push_state())


if __name__ == "__main__":
    sys.exit(main())
