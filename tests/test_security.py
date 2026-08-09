"""The security model, pinned.

These are the invariants that must survive every future change. A refactor that
quietly turns a gated action into an automatic one would be invisible in review
and catastrophic in production, so each property gets a test that fails loudly.

Nothing here touches the network or a real mailbox.
"""

from __future__ import annotations

import importlib

import pytest

from assistant.approvals import ApprovalGate, Decision, load_pending
from assistant.audit import AuditLog
from assistant.config import Capability, default_settings


@pytest.fixture
def settings(tmp_path, monkeypatch):
    # Settings read the environment at construction, so no module reload is
    # needed — which matters, because reloading would rebuild the Capability
    # enum and break identity comparisons against the one imported here.
    monkeypatch.setenv("ASSISTANT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ASSISTANT_SANDBOX_DIR", str(tmp_path / "sandbox"))
    monkeypatch.delenv("MAIL_IMAP_HOST", raising=False)
    return default_settings()


@pytest.fixture
def gate(settings):
    return ApprovalGate(settings, AuditLog(settings.audit_path))


@pytest.fixture
def notes(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSISTANT_SANDBOX_DIR", str(tmp_path / "sandbox"))
    from assistant.servers.notes import server

    importlib.reload(server)
    return server


# --- capability tiers -------------------------------------------------------


def test_reads_never_gate(gate):
    assert gate.evaluate("notes_read").decision is Decision.ALLOW
    gate.note_output("mail_read_recent")
    assert gate.evaluate("notes_read").decision is Decision.ALLOW


def test_append_survives_taint(gate):
    """The bug that motivated the APPEND tier: without this, an unattended
    triage can never record what it read."""
    gate.note_output("mail_read_recent")
    assert gate.evaluate("notes_append").decision is Decision.ALLOW


def test_write_gates_once_tainted(gate, settings):
    """Tests the tier, not a particular tool — which tools sit in WRITE changes
    as the sandbox moves between a scratch directory and a real vault."""
    settings.capabilities["demo_write"] = Capability.WRITE
    assert gate.evaluate("demo_write").decision is Decision.ALLOW
    gate.note_output("mail_read_recent")
    assert gate.evaluate("demo_write").decision is not Decision.ALLOW


def test_vault_mutations_always_gate(gate, settings):
    """The sandbox now points at hundreds of notes the user wrote themselves,
    behind iCloud with no undo. Overwriting, moving or deleting one must never
    be automatic, however clean the context looks."""
    for tool in ("notes_save", "notes_move", "notes_delete"):
        for mode in ("interactive", "autonomous"):
            settings.mode = mode
            assert gate.evaluate(tool).decision is not Decision.ALLOW, tool


def test_external_always_gates(gate, settings):
    for mode in ("interactive", "autonomous"):
        settings.mode = mode
        assert gate.evaluate("notes_delete").decision is not Decision.ALLOW


def test_unknown_tool_is_treated_as_external(gate):
    verdict = gate.evaluate("mail_send_message")
    assert verdict.capability is Capability.EXTERNAL
    assert verdict.decision is not Decision.ALLOW


def test_budget_exhaustion_denies(gate, settings):
    gate.tool_calls = settings.max_tool_calls
    assert gate.evaluate("notes_read").decision is Decision.DENY


# --- taint ------------------------------------------------------------------


def test_only_untrusted_tools_taint(gate):
    gate.note_output("notes_now")
    assert gate.tainted_by is None
    gate.note_output("mail_read_message")
    assert gate.tainted_by == "mail_read_message"


def test_every_mail_tool_is_untrusted(settings):
    """Mail is the hostile-input boundary; missing one would silently reopen it."""
    mail_tools = {t for t in settings.capabilities if t.startswith("mail_")}
    assert mail_tools, "mail tools should be classified"
    assert mail_tools <= settings.untrusted_output


# --- deferral ---------------------------------------------------------------


def test_autonomous_defers_and_keeps_working(gate, settings):
    settings.mode = "autonomous"
    permitted, message = gate.authorize("notes_delete", {"path": "x"})
    assert permitted is False
    assert "NOT been performed" in message
    assert "Continue" in message  # the agent must not stall
    assert len(load_pending(settings)) == 1


# --- containment ------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["../escape", "../../etc/passwd", "/etc/passwd", "a/../../../out"]
)
def test_note_paths_cannot_escape_the_sandbox(notes, path):
    with pytest.raises(ValueError):
        notes._resolve(path)


def test_nested_paths_are_allowed(notes):
    notes.append("projects/demo/log", "hello")
    assert "hello" in notes.read("projects/demo/log")


def test_move_refuses_to_overwrite(notes):
    notes.save("a", "first")
    notes.save("b", "second")
    assert "refusing to overwrite" in notes.move("a", "b")
    assert notes.read("b") == "second"


# --- mail -------------------------------------------------------------------


def test_mail_exposes_no_sending_capability():
    """Read-only is enforced by absence, not policy: there must be no tool to call."""
    from assistant.servers.mail import server

    exported = {n for n in dir(server) if not n.startswith("_")}
    forbidden = {"send", "reply", "forward", "delete", "move", "send_message"}
    assert not (exported & forbidden)


@pytest.mark.parametrize(
    "payload", ['x"\r\nA001 DELETE INBOX', 'a" OR "1', "line\nbreak"]
)
def test_imap_search_quoting_neutralises_injection(payload):
    """imaplib sends SEARCH criteria raw, so an unescaped query could smuggle
    in extra IMAP commands."""
    from assistant.servers.mail.server import _quote

    quoted = _quote(payload)
    assert "\r" not in quoted and "\n" not in quoted
    assert quoted.startswith('"') and quoted.endswith('"')
    assert '\\"' in quoted or '"' not in quoted[1:-1]


# --- audit ------------------------------------------------------------------


def test_refusals_are_audited(gate, settings):
    settings.mode = "autonomous"
    gate.authorize("notes_delete", {"path": "x"})
    log = settings.audit_path.read_text()
    assert "queued_for_approval" in log
    assert "tool_decision" in log


@pytest.mark.parametrize(
    "raw", ["abcd efgh ijkl mnop", "abcd\xa0efgh\xa0ijkl\xa0mnop", "  abcdefghijklmnop  "]
)
def test_app_password_whitespace_is_stripped(raw):
    """Gmail renders app passwords in groups of four, and copying from that page
    yields NON-BREAKING spaces. They survive .replace(' ', '') and, when they sit
    between groups, .strip() too — then fail as an opaque ascii codec error."""
    from assistant.servers.mail.server import _clean

    assert _clean(raw) == "abcdefghijklmnop"


def test_non_ascii_search_does_not_crash(monkeypatch):
    """A Cyrillic query used to raise UnicodeEncodeError inside imaplib before
    reaching the server, silently making the Russian half of a bilingual
    mailbox unsearchable — and an empty result reads as "nothing there"."""
    import contextlib
    from email.header import Header

    from assistant.servers.mail import server

    # Real mail encodes non-ASCII headers per RFC 2047, which is what the
    # decoder in _header expects; a raw UTF-8 header would not be realistic.
    subject = Header("Календарь", "utf-8").encode()
    raw = f"From: a@b.com\r\nSubject: {subject}\r\n\r\n".encode()

    class FakeIMAP:
        def uid(self, command, *args):
            if command == "SEARCH":
                return "OK", [b"1"]
            return "OK", [(b"1", raw)]

    @contextlib.contextmanager
    def fake_mailbox():
        yield FakeIMAP()

    monkeypatch.setattr(server, "_mailbox", fake_mailbox)
    result = server.search_messages("календарь")  # lower case: match is case-insensitive
    assert "non-ASCII query" in result  # took the local-scan path
    assert "uid=1" in result            # and actually matched


def test_search_matches_note_titles(notes):
    """People title notes with the thing they're about. A note called
    'procrastination and productivity' must answer a search for
    'procrastination' even if the body never repeats the word."""
    notes.save("Ideas/procrastination and productivity", "Всё про откладывание дел.")
    result = notes.search("procrastination")
    assert "Ideas/procrastination and productivity" in result
