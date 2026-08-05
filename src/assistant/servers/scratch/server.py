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
def search_notes(query: str, limit: int = 30) -> str:
    """Search across every note for a phrase, returning the matching lines and
    which note each came from. Use this to recall prior context before starting
    a task, rather than reading notes one by one."""
    needle = query.strip().lower()
    if not needle:
        return "Provide something to search for."
    limit = max(1, min(limit, 100))

    hits: list[str] = []
    for path in sorted(SANDBOX.glob("*.md")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if needle in line.lower():
                hits.append(f"{path.stem}:{number}: {line.strip()}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n[stopped at {limit} matches]"
    return "\n".join(hits) if hits else f"No notes mention {query!r}."


@mcp.tool()
def save_note(name: str, content: str) -> str:
    """Save a note under the given name, REPLACING it if it already exists.
    To add to a note without losing what is there, use append_note."""
    path = _resolve(name)
    path.write_text(content, encoding="utf-8")
    return f"Saved '{name}' ({len(content)} characters)."


@mcp.tool()
def append_note(name: str, content: str) -> str:
    """Add to the end of a note under a timestamped heading, creating it if it
    does not exist. Preferred over save_note for anything accumulating over
    time — a running log, a digest, notes on a project."""
    path = _resolve(name)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n\n## {stamp}\n\n{content.strip()}\n"
    existed = path.exists()
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    verb = "Appended to" if existed else "Created"
    return f"{verb} '{name}' (+{len(entry)} characters)."


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
