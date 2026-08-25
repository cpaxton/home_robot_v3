# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared tool-outcome shape for CHAT and EQA_EPISODE orchestrators.

Both loops should report ``ok`` / ``status`` / ``note`` the same way and optionally
write structured attempts into the graph-memory action-outcome ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutcome:
    """Normalized tool result for prompts, traces, and the attempt ledger."""

    ok: bool
    status: str = ""
    note: str = ""
    tool: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ok": bool(self.ok),
            "status": str(self.status or ""),
            "note": str(self.note or ""),
        }
        if self.tool:
            d["tool"] = self.tool
        if self.payload:
            d.update(self.payload)
        return d

    def render(self) -> str:
        """Human/LLM-facing one-liner for chat transcripts and tool-result blocks."""
        name = self.tool or "tool"
        head = f"[{name}] ok" if self.ok else f"[{name}] failed"
        bits = [head]
        if self.status:
            bits.append(f"status={self.status}")
        if self.note:
            bits.append(self.note)
        msg = self.payload.get("message") or self.payload.get("error")
        if msg and str(msg) not in (self.note, self.status):
            bits.append(str(msg)[:200])
        return " ".join(bits)

    @classmethod
    def from_exception(cls, tool: str, exc: BaseException) -> ToolOutcome:
        return cls(ok=False, status="exception", note=str(exc)[:200], tool=tool, payload={"error": str(exc)})

    @classmethod
    def from_eqa_dict(cls, tool: str, out: dict[str, Any] | None) -> ToolOutcome:
        """Wrap an existing EQA ``handle_tool`` dict without dropping fields."""
        d = dict(out or {})
        ok = bool(d.get("ok", False))
        status = str(d.get("status") or d.get("decision") or "")
        note = str(d.get("note") or d.get("error") or d.get("nav_note") or "")
        return cls(ok=ok, status=status, note=note, tool=tool, payload=d)

    @classmethod
    def coerce(cls, tool: str, result: Any) -> ToolOutcome:
        """Accept ToolOutcome, dict-with-ok, or free-form return values."""
        if isinstance(result, ToolOutcome):
            if not result.tool:
                result.tool = tool
            return result
        if isinstance(result, dict) and "ok" in result:
            return cls.from_eqa_dict(tool, result)
        text = "" if result is None else str(result)
        lowered = text.lower()
        failed = (lowered.startswith("tool ") and " failed" in lowered) or (
            "was interrupted or failed" in lowered or "not available" in lowered
        )
        return cls(
            ok=not failed,
            status="" if not failed else "failed",
            note=text[:240],
            tool=tool,
            payload={"message": text} if text else {},
        )


def maybe_record_tool_attempt(
    graph_memory: Any,
    outcome: ToolOutcome,
    *,
    step: int | None = None,
    source: str = "unknown",
) -> None:
    """Best-effort write of a tool outcome into ``graph_memory``'s attempt ledger."""
    if graph_memory is None or not hasattr(graph_memory, "record_attempt"):
        return
    tool = (outcome.tool or "").strip().lower()
    payload = outcome.payload or {}
    kind_map = {
        "investigate": "investigate",
        "navigate_to_obs": "navigate",
        "explore": "explore",
        "explore_frontier": "explore",
        "verify_siglip": "verify",
        "aim_arm_at": "closer_look",
        "take_ee_picture": "closer_look",
        "pick_place": "pick",
        "pickup": "pick",
        "place": "place",
    }
    kind = kind_map.get(tool)
    if kind is None:
        return
    if tool == "pick_place" and str(payload.get("action_kind") or "").lower() in {"pick", "place"}:
        kind = str(payload.get("action_kind")).lower()
    obs_id = payload.get("obs_id")
    node_id = payload.get("node_id") or payload.get("target_node_id")
    xyz = payload.get("target_xyz") or payload.get("xyz")
    xyz_t: tuple[float, float, float] | None = None
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
        xyz_t = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    phrase = str(payload.get("phrase") or "")
    target_kind = str(payload.get("target_kind") or "")
    target_id = str(payload.get("target_id") or payload.get("frontier_id") or "")
    view_id = str(payload.get("view_id") or "")
    room = str(payload.get("room") or "")
    status = str(outcome.status or payload.get("nav_note") or payload.get("error") or "")
    if kind == "verify":
        st = status.upper()
        if st == "ABSENT":
            outcome_str = "absent"
        elif st == "PRESENT":
            outcome_str = "present"
        elif st == "CANDIDATE":
            outcome_str = "candidate"
        else:
            outcome_str = "ok" if outcome.ok else "failed"
    else:
        outcome_str = "ok" if outcome.ok else "failed"
        if not outcome.ok and status:
            low = status.lower()
            if any(x in low for x in ("no_path", "navmesh", "blocked", "clearance")):
                outcome_str = "unreachable"
            elif any(x in low for x in ("timeout", "cancel", "abort")):
                outcome_str = "aborted"
    try:
        graph_memory.record_attempt(
            action_kind=kind,
            outcome=outcome_str,
            status_code=(status or outcome_str)[:80],
            note=outcome.note or status,
            step=step,
            target_node_id=int(node_id) if node_id is not None else None,
            obs_id=int(obs_id) if obs_id is not None else None,
            xyz=xyz_t,
            source=source,
            phrase=phrase,
            room=room,
            target_kind=target_kind,
            target_id=target_id,
            view_id=view_id,
        )
    except Exception:
        pass
