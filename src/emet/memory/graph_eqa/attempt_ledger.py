# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Structured action-outcome ledger for GraphEQA / Dynagraph memory.

Records navigate / investigate / verify / closer_look / pick / place attempts so
planners can avoid repeating failed actions. Lives next to the scene graph;
per-node ``nav_attempts`` / ``nav_failures`` remain dual-written compatibility
views. Opt-in via ``eqa.attempt_ledger`` / ``EMET_EQA_ATTEMPT_LEDGER`` (default off
so HM-EQA / OVMM paper paths are unchanged).

See ``docs/plans/2026-08-08_embodied_agent_planning.md`` Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AttemptSource = Literal["chat", "eqa", "unknown"]

ACTION_KINDS = frozenset(
    {
        "navigate",
        "investigate",
        "verify",
        "closer_look",
        "pick",
        "place",
        "explore",
    }
)

OUTCOMES = frozenset(
    {
        "ok",
        "failed",
        "aborted",
        "absent",
        "unreachable",
        "candidate",
        "present",
    }
)

# Substrings in planner notes → status_code (first match wins).
_NAV_NOTE_STATUS: tuple[tuple[str, str], ...] = (
    ("navmesh_no_path", "navmesh_no_path"),
    ("no_path", "no_path"),
    ("no path", "no_path"),
    ("rejected_low_clearance", "rejected_low_clearance"),
    ("rejected low clearance", "rejected_low_clearance"),
    ("low clearance", "rejected_low_clearance"),
    ("aborted_waypoint_timeout", "timeout"),
    ("waypoint timeout", "timeout"),
    ("timeout", "timeout"),
    ("user_cancelled", "user_cancelled"),
    ("cancelled", "user_cancelled"),
    ("blocked", "blocked"),
    ("failed_move", "failed_move"),
)

_UNREACHABLE_CODES = frozenset(
    {
        "navmesh_no_path",
        "no_path",
        "blocked",
        "unreachable",
        "rejected_low_clearance",
    }
)
_ABORTED_CODES = frozenset({"timeout", "user_cancelled", "aborted"})


def infer_nav_status_code(*, success: bool, note: str = "") -> str:
    """Map a navigation success flag + free-form note onto a stable status_code."""
    if success:
        return "ok"
    text = str(note or "").strip().lower()
    for needle, code in _NAV_NOTE_STATUS:
        if needle in text:
            return code
    return "failed_move" if text else "failed"


def infer_nav_outcome(*, success: bool, status_code: str = "") -> str:
    """Map success + status_code onto a ledger ``outcome``."""
    if success or status_code == "ok":
        return "ok"
    code = str(status_code or "").strip().lower()
    if code in _UNREACHABLE_CODES:
        return "unreachable"
    if code in _ABORTED_CODES:
        return "aborted"
    return "failed"


def _as_xyz_tuple(xyz: Any) -> tuple[float, float, float] | None:
    if xyz is None:
        return None
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 2:
        z = float(xyz[2]) if len(xyz) >= 3 else 0.0
        return (float(xyz[0]), float(xyz[1]), z)
    try:
        import numpy as np

        arr = np.asarray(xyz, dtype=float).reshape(-1)
        if arr.size < 2:
            return None
        z = float(arr[2]) if arr.size >= 3 else 0.0
        return (float(arr[0]), float(arr[1]), z)
    except Exception:
        return None


@dataclass(frozen=True)
class AttemptRecord:
    """One recorded action attempt and its outcome."""

    action_kind: str
    outcome: str
    status_code: str
    note: str = ""
    step: int = 0
    target_node_id: int | None = None
    obs_id: int | None = None
    xyz: tuple[float, float, float] | None = None
    source: AttemptSource = "unknown"
    question_id: str | None = None
    phrase: str = ""
    # Canonical room label when known (schema v2). Empty for v1 imports / unknown.
    room: str = ""
    # Stable graph identity fields (schema v3); legacy node/obs ids remain adapters.
    target_kind: str = ""
    target_id: str = ""
    view_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "action_kind": self.action_kind,
            "outcome": self.outcome,
            "status_code": self.status_code,
            "note": self.note,
            "step": int(self.step),
            "target_node_id": self.target_node_id,
            "obs_id": self.obs_id,
            "xyz": list(self.xyz) if self.xyz is not None else None,
            "source": self.source,
            "question_id": self.question_id,
            "phrase": self.phrase,
            "room": self.room,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "view_id": self.view_id,
        }

    def summary_bit(self) -> str:
        """Compact tag for place cards / diagnostics, e.g. ``navigate:failed(no_path)``."""
        kind = self.action_kind or "?"
        out = self.outcome or "?"
        code = (self.status_code or "").strip()
        if code and code not in {out, "ok", "failed"}:
            return f"{kind}:{out}({code})"
        if code and code != out:
            return f"{kind}:{out}({code})"
        return f"{kind}:{out}"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | AttemptRecord) -> AttemptRecord:
        if isinstance(data, AttemptRecord):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"AttemptRecord.from_dict expected dict, got {type(data)!r}")
        kind = str(data.get("action_kind") or "").strip().lower()
        if kind not in ACTION_KINDS:
            raise ValueError(f"invalid action_kind={kind!r}; expected one of {sorted(ACTION_KINDS)}")
        outcome = str(data.get("outcome") or "").strip().lower()
        if outcome not in OUTCOMES:
            # Tolerate aliases from older traces.
            if outcome in {"true", "success", "succeeded"}:
                outcome = "ok"
            elif outcome in {"false", "error", "fail"}:
                outcome = "failed"
            else:
                raise ValueError(f"invalid outcome={outcome!r}; expected one of {sorted(OUTCOMES)}")
        status = str(data.get("status_code") or "").strip() or outcome
        src = str(data.get("source") or "unknown").strip().lower()
        if src not in ("chat", "eqa", "unknown"):
            src = "unknown"
        nid = data.get("target_node_id")
        oid = data.get("obs_id")
        room = str(data.get("room") or "").strip().lower()[:40]
        return cls(
            action_kind=kind,
            outcome=outcome,
            status_code=status[:80],
            note=str(data.get("note") or "")[:240],
            step=int(data.get("step") or 0),
            target_node_id=int(nid) if nid is not None else None,
            obs_id=int(oid) if oid is not None else None,
            xyz=_as_xyz_tuple(data.get("xyz")),
            source=src,  # type: ignore[arg-type]
            question_id=str(data["question_id"]) if data.get("question_id") is not None else None,
            phrase=str(data.get("phrase") or "").strip().lower()[:80],
            room=room,
            target_kind=str(data.get("target_kind") or "").strip().lower()[:40],
            target_id=str(data.get("target_id") or "").strip()[:120],
            view_id=str(data.get("view_id") or "").strip()[:120],
        )


def records_to_dicts(rows: list[AttemptRecord] | None) -> list[dict[str, Any]]:
    return [r.to_dict() for r in (rows or [])]


def records_from_dicts(items: list[Any] | None) -> list[AttemptRecord]:
    out: list[AttemptRecord] = []
    for item in items or []:
        if isinstance(item, AttemptRecord):
            out.append(item)
        else:
            out.append(AttemptRecord.from_dict(item))
    return out


def summary_bits_for_obs(rows: list[AttemptRecord], obs_id: int, *, max_bits: int = 4) -> str:
    """Newest-first compact summary for place cards."""
    oid = int(obs_id)
    matched = [r for r in rows if r.obs_id is not None and int(r.obs_id) == oid]
    if not matched:
        return ""
    bits: list[str] = []
    seen: set[str] = set()
    for r in reversed(matched):
        bit = r.summary_bit()
        if bit in seen:
            continue
        seen.add(bit)
        bits.append(bit)
        if len(bits) >= int(max_bits):
            break
    return "; ".join(bits)
