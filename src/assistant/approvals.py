"""The approval gate — where independence and safety are traded off.

Every tool call the agent makes passes through `ApprovalGate.authorize`. The
gate is deliberately boring, deterministic code: no model output influences the
decision, so no prompt injection can talk its way past it.

Three ideas do the work:

1. Capability tiers (config.Capability). Reads and reversible sandboxed writes
   run without asking. Irreversible or outward-facing actions never do.

2. Taint. Once the agent reads attacker-influenceable content, `tainted_by` is
   set and WRITE drops from allow to ask. The agent is fully autonomous while
   its context is clean, and cautious the moment it isn't. This is the defence
   against "the email told me to delete your files".

3. Deferral instead of denial. In autonomous mode nobody is at the keyboard, so
   a gated action is queued rather than refused, and the agent is told to carry
   on with everything else. Denial would stall the agent; queuing keeps it
   working. You review the queue later and approve, and only then does it run.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .audit import AuditLog
from .config import Capability, Settings


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DEFER = "defer"
    DENY = "deny"


@dataclass
class Verdict:
    decision: Decision
    capability: Capability
    reason: str


class ApprovalGate:
    def __init__(self, settings: Settings, audit: AuditLog) -> None:
        self.settings = settings
        self.audit = audit
        # Name of the first tool that brought untrusted content into context.
        self.tainted_by: str | None = None
        self.tool_calls = 0

    # -- taint ---------------------------------------------------------------

    def note_output(self, tool: str) -> None:
        """Call after a tool returns. Marks the session tainted if the tool's
        output is attacker-influenceable."""
        if tool in self.settings.untrusted_output and self.tainted_by is None:
            self.tainted_by = tool
            self.audit.record("context_tainted", tool=tool)

    # -- policy --------------------------------------------------------------

    def capability_of(self, tool: str) -> Capability:
        return self.settings.capabilities.get(tool, self.settings.default_capability)

    def evaluate(self, tool: str) -> Verdict:
        """Pure policy. No side effects, no I/O, no model input."""
        cap = self.capability_of(tool)

        if self.tool_calls >= self.settings.max_tool_calls:
            return Verdict(Decision.DENY, cap, "tool-call budget exhausted for this run")

        if cap is Capability.READ:
            return Verdict(Decision.ALLOW, cap, "read-only")

        if cap is Capability.APPEND:
            # Additive and sandboxed: it cannot destroy anything, so taint does
            # not change the answer. Without this, an unattended agent could
            # never record what it read — the note write would queue forever.
            return Verdict(Decision.ALLOW, cap, "additive, cannot destroy")

        if cap is Capability.WRITE:
            if self.tainted_by is None:
                return Verdict(Decision.ALLOW, cap, "reversible write, context is clean")
            return Verdict(
                self._gated(),
                cap,
                f"reversible write, but context was tainted by '{self.tainted_by}'",
            )

        # EXTERNAL: irreversible or leaves the machine. Never automatic.
        return Verdict(self._gated(), cap, "irreversible or outward-facing")

    def _gated(self) -> Decision:
        return Decision.ASK if self.settings.mode == "interactive" else Decision.DEFER

    # -- enforcement ---------------------------------------------------------

    def authorize(self, tool: str, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Returns (permitted, message). `message` explains a refusal to the
        agent — it is the tool result the model will read."""
        verdict = self.evaluate(tool)
        self.tool_calls += 1
        self.audit.record(
            "tool_decision",
            tool=tool,
            input=tool_input,
            capability=verdict.capability.value,
            decision=verdict.decision.value,
            reason=verdict.reason,
            tainted_by=self.tainted_by,
        )

        if verdict.decision is Decision.ALLOW:
            return True, ""

        if verdict.decision is Decision.DENY:
            return False, f"Denied: {verdict.reason}. Stop and report this to the user."

        if verdict.decision is Decision.ASK:
            return self._prompt(tool, tool_input, verdict)

        return self._defer(tool, tool_input, verdict)

    def _prompt(self, tool: str, tool_input: dict[str, Any], verdict: Verdict) -> tuple[bool, str]:
        print(f"\n  ⚠  {tool} — {verdict.reason}")
        print(f"     {json.dumps(tool_input, ensure_ascii=False)}")
        try:
            answer = input("     Allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        approved = answer in {"y", "yes"}
        self.audit.record("human_response", tool=tool, approved=approved)
        if approved:
            return True, ""
        return False, "The user declined this action. Do not retry it; continue with the rest of the task."

    def _defer(self, tool: str, tool_input: dict[str, Any], verdict: Verdict) -> tuple[bool, str]:
        entry_id = uuid.uuid4().hex[:8]
        pending = self._load_pending()
        pending[entry_id] = {
            "id": entry_id,
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "input": tool_input,
            "capability": verdict.capability.value,
            "reason": verdict.reason,
        }
        self._save_pending(pending)
        self.audit.record("queued_for_approval", id=entry_id, tool=tool, input=tool_input)
        # This wording matters: it keeps the agent productive instead of stuck.
        return False, (
            f"Queued for human approval as '{entry_id}' — the action has NOT been performed. "
            "Do not retry it and do not attempt a workaround. Continue with every other part "
            "of the task, and tell the user at the end what is waiting on their approval."
        )

    # -- the pending queue ---------------------------------------------------

    def _load_pending(self) -> dict[str, Any]:
        return load_pending(self.settings)

    def _save_pending(self, pending: dict[str, Any]) -> None:
        save_pending(self.settings, pending)


def load_pending(settings: Settings) -> dict[str, Any]:
    if not settings.pending_path.exists():
        return {}
    with settings.pending_path.open(encoding="utf-8") as f:
        return json.load(f)


def save_pending(settings: Settings, pending: dict[str, Any]) -> None:
    settings.pending_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.pending_path.open("w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
