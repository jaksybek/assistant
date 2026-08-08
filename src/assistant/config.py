"""Configuration: settings, the MCP server registry, and the capability model.

The capability model is the heart of the security design. Every tool the agent
can call is classified into one of three tiers, in code the model cannot talk
its way past:

    READ      — observes the world, changes nothing. Runs freely.
    APPEND    — adds to the sandbox without replacing anything. Cannot destroy,
                so it runs freely even on untrusted context. This is what lets
                the agent keep a durable log of what it read.
    WRITE     — replaces or removes sandbox state. Reversible in principle, but
                it can destroy. Runs freely on trusted context; gated once
                untrusted content has been read (see approvals.py).
    EXTERNAL  — irreversible, or leaves this machine. ALWAYS gated, in every
                mode, no exceptions.

The APPEND/WRITE split exists because taint was gating the wrong thing. Reading
mail necessarily taints the session, so an unattended "triage my inbox and note
what matters" could never finish — the note write queued forever and the agent
forgot everything by morning. Appending is strictly additive: the worst a
prompt injection achieves is noise in a log the user can read, bounded by
max_tool_calls. Overwriting is a different matter and stays gated.

That split is what lets the agent be genuinely independent without being
dangerous: it acts on its own across the entire reversible surface, and the
human is only ever asked about the small set of actions that can actually
cause harm.

Unknown tools classify as EXTERNAL — the most restricted tier. Fail safe.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

def _data_dir() -> Path:
    """Runtime data (audit log, notes, pending approvals). Gitignored.

    Read at construction rather than import, so the environment still decides
    after this module has been imported — which tests rely on, and which avoids
    a module-level constant that silently freezes configuration.
    """
    return Path(os.environ.get("ASSISTANT_DATA_DIR", "./data")).expanduser().resolve()


def _sandbox_dir() -> Path:
    override = os.environ.get("ASSISTANT_SANDBOX_DIR")
    return Path(override).expanduser().resolve() if override else _data_dir() / "sandbox"


class Capability(str, Enum):
    """What a tool can do to the world. Drives the approval decision."""

    READ = "read"
    APPEND = "append"
    WRITE = "write"
    EXTERNAL = "external"


@dataclass
class MCPServer:
    """A scoped integration, launched as its own isolated subprocess."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)


@dataclass
class Settings:
    # Claude Opus 5 thinks by default; we set it explicitly for clarity.
    model: str = field(default_factory=lambda: os.environ.get("ASSISTANT_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=lambda: os.environ.get("ASSISTANT_EFFORT", "high"))
    max_tokens: int = 16000

    data_dir: Path = field(default_factory=_data_dir)
    audit_path: Path = field(default_factory=lambda: _data_dir() / "audit.jsonl")
    pending_path: Path = field(default_factory=lambda: _data_dir() / "pending.json")
    # The ONLY directory the agent may write to. Never your whole disk.
    # Point this at a subfolder of an Obsidian vault to keep one knowledge base
    # while still confining the agent to its own corner of it.
    sandbox_dir: Path = field(default_factory=_sandbox_dir)

    # interactive: a human is present, so gated actions prompt on the terminal.
    # autonomous:  nobody is watching, so gated actions queue for later review
    #              instead of blocking. The agent keeps working either way.
    mode: str = field(default_factory=lambda: os.environ.get("ASSISTANT_MODE", "interactive"))

    servers: list[MCPServer] = field(default_factory=list)

    # Namespaced tool name -> capability tier.
    capabilities: dict[str, Capability] = field(default_factory=dict)

    # Tools whose OUTPUT is attacker-influenceable. Reading one of these taints
    # the session: writes stop being automatic. Mail, web fetch, and any tool
    # returning third-party content belong here.
    untrusted_output: set[str] = field(default_factory=set)

    # Runaway protection: an autonomous loop can't spend forever.
    max_tool_calls: int = 40

    # Anything not classified above is treated as the most dangerous tier.
    default_capability: Capability = Capability.EXTERNAL


def default_settings() -> Settings:
    """The starter configuration: one trivial MCP server, one of each tier."""
    settings = Settings(
        servers=[
            MCPServer(
                name="notes",
                # Launch the server with the same interpreter running the agent,
                # so it shares this project's virtualenv.
                command=sys.executable,
                args=["-m", "assistant.servers.notes.server"],
            ),
        ],
        capabilities={
            "notes_now": Capability.READ,
            "notes_outline": Capability.READ,
            "notes_list_notes": Capability.READ,
            "notes_read": Capability.READ,
            "notes_search": Capability.READ,
            "notes_backlinks": Capability.READ,
            # Strictly additive — safe even after reading untrusted content.
            "notes_append": Capability.APPEND,
            # Once the sandbox points at a real Obsidian vault, these stop
            # being cheap. Overwriting or relocating one of 656 notes the user
            # wrote themselves is not the same as editing one the agent made,
            # and there is no undo behind iCloud. Gate all three, always —
            # reading and appending stay free, which is where the value is.
            "notes_save": Capability.EXTERNAL,
            "notes_move": Capability.EXTERNAL,
            "notes_delete": Capability.EXTERNAL,
            # Mail is read-only by construction — there is no send tool to
            # classify, and the server opens the mailbox readonly.
            "mail_list_messages": Capability.READ,
            "mail_search_messages": Capability.READ,
            "mail_read_message": Capability.READ,
            "mail_read_recent": Capability.READ,
        },
        # Output written by someone other than the user. Reading any of these
        # taints the session: writes stop being automatic. A note counts —
        # it can contain text pasted out of an email.
        untrusted_output={
            "notes_read",
            "notes_search",
            "mail_list_messages",
            "mail_search_messages",
            "mail_read_message",
            "mail_read_recent",
        },
    )

    # Mail only starts if it has been configured. An unconfigured integration
    # is an absent one, not a broken one.
    if os.environ.get("MAIL_IMAP_HOST"):
        settings.servers.append(
            MCPServer(
                name="mail",
                command=sys.executable,
                args=["-m", "assistant.servers.mail.server"],
            )
        )

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.sandbox_dir.mkdir(parents=True, exist_ok=True)
    return settings
