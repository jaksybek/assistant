"""A folder-structured note store — the knowledge base.

Notes are plain markdown files in a directory tree, so the same store opens as
an Obsidian or Logseq vault with no export step. Nothing here is a proprietary
format; the agent and a human editor are peers over the same files.

The tools are shaped for PROGRESSIVE DISCLOSURE, because context is the scarce
resource. Reading everything to answer one question is what makes a knowledge
base expensive. Three cheapening steps, in order:

    outline()          the shape of the store — folder names, note names,
                       sizes. No content at all. Costs almost nothing and is
                       usually enough to decide where to look.
    search(q, folder)  matching LINES, not whole notes, and scoped to one
                       folder so a big archive does not cost more than a
                       small one.
    read(path)         the full note, only once something is worth reading.

Every path is confined to the sandbox directory. `path` is model-supplied and
untrusted: it is resolved and checked before any disk access, so no note name
can escape the tree.
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

MAX_MATCHES = 40
MAX_OUTLINE = 300

mcp = MCPServer("notes")


def _resolve(path: str) -> Path:
    """Map a note path to a file inside the sandbox, or refuse.

    This is the containment boundary. Accepts nested paths like
    'projects/elfathlon/baseline'; rejects anything that escapes the tree.
    """
    cleaned = path.strip().strip("/")
    if not cleaned:
        raise ValueError("A note path is required.")
    if "\x00" in cleaned:
        raise ValueError("Invalid note path.")
    candidate = (SANDBOX / f"{cleaned}.md").resolve()
    if not candidate.is_relative_to(SANDBOX):
        raise ValueError(f"'{path}' escapes the notes directory")
    return candidate


def _folder(folder: str) -> Path:
    cleaned = folder.strip().strip("/")
    target = (SANDBOX / cleaned).resolve() if cleaned else SANDBOX
    if not target.is_relative_to(SANDBOX):
        raise ValueError(f"'{folder}' escapes the notes directory")
    return target


def _rel(path: Path) -> str:
    return str(path.relative_to(SANDBOX).with_suffix(""))


@mcp.tool()
def now() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@mcp.tool()
def outline(folder: str = "", depth: int = 2) -> str:
    """Show the SHAPE of the knowledge base — folders, note names and sizes,
    with no content. Start here: it is by far the cheapest way to find out what
    exists and decide where to look, before spending anything on reading."""
    root = _folder(folder)
    if not root.exists():
        return f"No folder '{folder}'."
    depth = max(1, min(depth, 6))

    lines: list[str] = []

    def walk(directory: Path, level: int) -> None:
        if level > depth or len(lines) >= MAX_OUTLINE:
            return
        entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        for entry in entries:
            if len(lines) >= MAX_OUTLINE:
                lines.append("[outline truncated]")
                return
            indent = "  " * (level - 1)
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
                walk(entry, level + 1)
            elif entry.suffix == ".md":
                kb = max(1, entry.stat().st_size // 1024)
                lines.append(f"{indent}{entry.stem}  ({kb}k)")

    walk(root, 1)
    if not lines:
        return f"'{folder or 'notes'}' is empty."
    return f"{folder or 'notes'}/\n" + "\n".join(lines)


@mcp.tool()
def list_notes(folder: str = "") -> str:
    """List the immediate contents of one folder — subfolders and notes, not
    recursive. Use outline for the wider shape."""
    root = _folder(folder)
    if not root.exists():
        return f"No folder '{folder}'."
    items = []
    for entry in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if entry.is_dir():
            items.append(f"{entry.name}/")
        elif entry.suffix == ".md":
            items.append(entry.stem)
    return "\n".join(items) if items else f"'{folder or 'notes'}' is empty."


@mcp.tool()
def search(query: str, folder: str = "", limit: int = 25) -> str:
    """Search notes for a phrase, returning the matching LINES and which note
    each came from — not whole notes. Pass `folder` to scope the search to one
    part of the tree, which keeps the cost flat as the store grows."""
    needle = query.strip().lower()
    if not needle:
        return "Provide something to search for."
    root = _folder(folder)
    if not root.exists():
        return f"No folder '{folder}'."
    limit = max(1, min(limit, MAX_MATCHES))

    hits: list[str] = []
    for file in sorted(root.rglob("*.md")):
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if needle in line.lower():
                hits.append(f"{_rel(file)}:{number}: {line.strip()[:200]}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n[stopped at {limit} matches]"
    scope = f" under '{folder}'" if folder else ""
    return "\n".join(hits) if hits else f"No notes{scope} mention {query!r}."


@mcp.tool()
def read(path: str) -> str:
    """Read one note in full, by path (e.g. 'mail/inbox-log'). Reach for this
    only once outline or search says the note is worth the context."""
    file = _resolve(path)
    if not file.exists():
        return f"No note at '{path}'."
    return file.read_text(encoding="utf-8")


@mcp.tool()
def append(path: str, content: str) -> str:
    """Add to the end of a note under a timestamped heading, creating it and
    any parent folders if needed. Preferred for anything accumulating over
    time — a running log, notes on a project."""
    file = _resolve(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n\n## {stamp}\n\n{content.strip()}\n"
    existed = file.exists()
    with file.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f"{'Appended to' if existed else 'Created'} '{path}' (+{len(entry)} chars)."


@mcp.tool()
def save(path: str, content: str) -> str:
    """Write a note, REPLACING it entirely if it exists. Use append to add to a
    note without losing what is already there."""
    file = _resolve(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return f"Saved '{path}' ({len(content)} chars)."


@mcp.tool()
def delete(path: str) -> str:
    """Permanently delete a note. This cannot be undone."""
    file = _resolve(path)
    if not file.exists():
        return f"No note at '{path}'."
    file.unlink()
    return f"Deleted '{path}'."


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
