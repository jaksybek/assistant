"""A scratch-notes MCP server — the smallest thing that exercises every tier.

`now` and the list/read tools are READ. `save_note` is a reversible sandboxed
WRITE. `delete_note` is EXTERNAL — irreversible, so it is always gated. The
tiers are assigned in config.py, not here: a server never decides its own
privileges.

Every path is confined to the sandbox directory. The server refuses any name
that would escape it, so even a fully compromised agent cannot reach the rest
of the disk through this server.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

SANDBOX = Path(
    os.environ.get("ASSISTANT_SANDBOX_DIR")
    or Path(os.environ.get("ASSISTANT_DATA_DIR", "./data")).resolve() / "sandbox"
).resolve()
SANDBOX.mkdir(parents=True, exist_ok=True)

mcp = MCPServer("scratch")


def _resolve(name: str) -> Path:
    """Map a note name to a path inside the sandbox, or refuse.

    This is the containment boundary. `name` is model-supplied and untrusted:
    resolve it and confirm it is still under SANDBOX before touching the disk.
    """
    candidate = (SANDBOX / f"{name}.md").resolve()
    if not candidate.is_relative_to(SANDBOX):
        raise ValueError(f"'{name}' escapes the sandbox directory")
    return candidate


@mcp.tool()
def now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def list_notes() -> str:
    """List the names of all saved notes."""
    names = sorted(p.stem for p in SANDBOX.glob("*.md"))
    return "\n".join(names) if names else "(no notes yet)"


@mcp.tool()
def read_note(name: str) -> str:
    """Read the contents of a saved note by name."""
    path = _resolve(name)
    if not path.exists():
        return f"No note named '{name}'."
    return path.read_text(encoding="utf-8")


@mcp.tool()
def save_note(name: str, content: str) -> str:
    """Save a note under the given name, replacing it if it already exists."""
    path = _resolve(name)
    path.write_text(content, encoding="utf-8")
    return f"Saved '{name}' ({len(content)} characters)."


@mcp.tool()
def delete_note(name: str) -> str:
    """Permanently delete a saved note. This cannot be undone."""
    path = _resolve(name)
    if not path.exists():
        return f"No note named '{name}'."
    path.unlink()
    return f"Deleted '{name}'."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
