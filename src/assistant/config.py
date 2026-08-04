"""Configuration: settings, the MCP server registry, and the capability model.

The capability model is the heart of the security design. Every tool the agent
can call is classified into one of three tiers, in code the model cannot talk
its way past:

    READ      — observes the world, changes nothing. Runs freely.
    WRITE     — changes state, but reversibly and inside the sandbox. Runs
                freely on trusted context; gated once untrusted content has
                been read (see approvals.py).
    EXTERNAL  — irreversible, or leaves this machine. ALWAYS gated, in every
                mode, no exceptions.

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

# Runtime data (audit log, notes, pending approvals). Gitignored.
DATA_DIR = Path(os.environ.get("ASSISTANT_DATA_DIR", "./data")).resolve()


class Capability(str, Enum):
    """What a tool can do to the world. Drives the approval decision."""

    READ = "read"
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

    data_dir: Path = DATA_DIR
    audit_path: Path = DATA_DIR / "audit.jsonl"
    pending_path: Path = DATA_DIR / "pending.json"
    # The ONLY directory the agent may write to. Never your whole disk.
    sandbox_dir: Path = DATA_DIR / "sandbox"

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
                name="scratch",
                # Launch the server with the same interpreter running the agent,
                # so it shares this project's virtualenv.
                command=sys.executable,
                args=["-m", "assistant.servers.scratch.server"],
            ),
        ],
        capabilities={
            "scratch_now": Capability.READ,
            "scratch_list_notes": Capability.READ,
            "scratch_read_note": Capability.READ,
            # Reversible and sandboxed — the agent does this on its own.
            "scratch_save_note": Capability.WRITE,
            # Irreversible. Gated every single time, in every mode.
            "scratch_delete_note": Capability.EXTERNAL,
            # Mail is read-only by construction — there is no send tool to
            # classify, and the server opens the mailbox readonly.
            "mail_list_messages": Capability.READ,
            "mail_search_messages": Capability.READ,
            "mail_read_message": Capability.READ,
        },
        # Output written by someone other than the user. Reading any of these
        # taints the session: writes stop being automatic. A note counts —
        # it can contain text pasted out of an email.
        untrusted_output={
            "scratch_read_note",
            "mail_list_messages",
            "mail_search_messages",
            "mail_read_message",
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
