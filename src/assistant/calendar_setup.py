"""One-time Google OAuth for the calendar server: get a refresh token.

Google's installed-application flow, with a loopback redirect. The browser
sends the authorisation code to a throwaway HTTP server on localhost rather
than to any third party, so the code never leaves this machine and there is
nothing to copy by hand — the step that has already gone wrong twice in this
project with credentials pasted from screenshots.

Run once. The refresh token it prints goes in .env and is long-lived; the
short-lived access tokens are fetched from it at runtime and never stored.

The consent screen will say the agent can see AND edit events — the scope is
`calendar.events`, because creating, moving and cancelling now exist. It should
NOT offer calendar settings, sharing, or deleting whole calendars; if it does,
something is wrong, so stop and say so.

That widening retires a safety property worth naming out loud: under the old
read-only token, no bug anywhere in this system could have changed the
calendar. Now one credential can, and the approval gate is what stands in the
way — every write is EXTERNAL, gated in every mode, and the tools that touch an
existing event refuse unless the caller names that event's real title.

Anyone re-running this after the read-only version must RE-CONSENT: a refresh
token keeps the scope it was granted with, so calendar writes go on failing
with 403 until a new token is issued.
"""

from __future__ import annotations

import http.server
import os
import secrets
import socket
import sys
import threading
import urllib.parse
import webbrowser

import httpx
from dotenv import load_dotenv

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/calendar.events"

HOW_TO_GET_A_CLIENT = """\
No client credentials found.

In the Google Cloud console (console.cloud.google.com):

  1. Create a project, or pick an existing one.
  2. APIs & Services -> Library -> enable "Google Calendar API".
  3. APIs & Services -> OAuth consent screen -> External, add yourself as a
     test user. It does not need verifying for personal use.
  4. APIs & Services -> Credentials -> Create credentials ->
     OAuth client ID -> Desktop app.
  5. Copy the client id and client secret into ~/assistant/.env as:

       GOOGLE_CALENDAR_CLIENT_ID=...
       GOOGLE_CALENDAR_CLIENT_SECRET=...

Then run this command again.
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Catch the single redirect, then get out of the way."""

    code: str | None = None
    state: str | None = None
    error: str | None = None

    def do_GET(self) -> None:  # noqa: N802 - name fixed by the stdlib
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        _Handler.code = (params.get("code") or [None])[0]
        _Handler.state = (params.get("state") or [None])[0]
        _Handler.error = (params.get("error") or [None])[0]

        body = (
            b"Authorised. You can close this tab and return to the terminal."
            if _Handler.code
            else b"Authorisation failed. Return to the terminal."
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence the default request logging — it would print the code."""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> None:
    load_dotenv()
    client_id = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET")
    if not (client_id and client_secret):
        print(HOW_TO_GET_A_CLIENT)
        sys.exit(1)

    port = _free_port()
    redirect_uri = f"http://127.0.0.1:{port}"
    # Binds the callback to this run, so a stray or replayed redirect from
    # anywhere else is rejected rather than silently accepted.
    state = secrets.token_urlsafe(24)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        # offline + consent is what makes Google return a refresh token at all;
        # without them you get an access token that dies in an hour and a setup
        # that appears to work until tomorrow.
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("Opening your browser to authorise calendar access (see and edit events).")
    print("If it does not open, paste this into your browser:\n")
    print(url, "\n")
    webbrowser.open(url)

    thread.join(timeout=300)
    server.server_close()

    if _Handler.error:
        print(f"Authorisation was refused: {_Handler.error}")
        sys.exit(1)
    if not _Handler.code:
        print("Timed out waiting for the browser. Run the command again.")
        sys.exit(1)
    if _Handler.state != state:
        print("State mismatch — the response did not come from the request we made.")
        sys.exit(1)

    response = httpx.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": _Handler.code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if response.status_code != 200:
        print(f"Token exchange failed ({response.status_code}): {response.text[:300]}")
        sys.exit(1)

    payload = response.json()
    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        print(
            "Google returned no refresh token. This happens when the app was "
            "already authorised: revoke it at myaccount.google.com/permissions "
            "and run this again."
        )
        sys.exit(1)

    print("\nAdd this to ~/assistant/.env:\n")
    print(f"GOOGLE_CALENDAR_REFRESH_TOKEN={refresh_token}")
    print(
        "\nOptionally set GOOGLE_CALENDAR_ID to read a calendar other than "
        "your default one (it accepts the calendar's address)."
    )


if __name__ == "__main__":
    main()
