"""What a refusal from a Google token endpoint means, and whether it will pass.

Both the calendar and Drive servers exchange a credential for an access token,
and both hit the same trap: `httpx.post` returns a response for 429 and 503 as
happily as for 200, so a non-200 check turns Google having a bad five minutes
into the same exception as a wrong client secret. The preflight then reads that
as "credentials rejected", aborts, and the morning briefing is lost to a rate
limit that would have cleared by itself — the exact opposite of the stated rule
that transient Google failures must never cost the briefing.

Carrying the status on the exception is what lets a caller tell those apart.
The alternative, matching on the message text, breaks the first time a message
is reworded.
"""

from __future__ import annotations

# 408 request timeout, 429 rate limited, and anything 5xx: Google's problem,
# not the credential's. Everything else — notably 400 `invalid_grant` and 401
# `invalid_client` — is a statement about the credential itself and will still
# be true on the next run, so it must stop the job rather than be slept off.
RETRYABLE = (408, 429)


def retryable(status: int) -> bool:
    """Will this status plausibly be different on the next run?

    One function rather than an inline `>= 500` at each call site, because the
    rule was already stated twice and the two copies already disagreed: the
    Drive folder check treated 5xx as transient and let 429 fall through to a
    fatal error, so a rate limit there still cost the briefing after the token
    exchange had been fixed. A rule about retrying belongs in one place.
    """
    return status in RETRYABLE or status >= 500


class TokenRefused(RuntimeError):
    """A non-200 from a Google token endpoint, with the status kept."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status

    @property
    def transient(self) -> bool:
        return retryable(self.status)
