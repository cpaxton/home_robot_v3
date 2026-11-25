# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Shared tool-outcome shape for CHAT and EQA_EPISODE orchestrators.

Both loops should report ``ok`` / ``status`` / ``note`` the same way and optionally
write structured attempts into the graph-memory action-outcome ledger.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from emet.controller.task.tamp.api import SCHEMA_VERSION, TAMP_TOOLS


def _valid_versioned_result(tool: str, value: Any) -> bool:
    """Validate the shared TAMP/relative-motion envelope without serialization."""
    return (
        isinstance(value, dict)
        and value.get("schema_version") == SCHEMA_VERSION
        and value.get("tool") == tool
        and isinstance(value.get("status"), str)
        and value["status"] in {"ok", "error", "partial"}
        and isinstance(value.get("code"), str)
        and isinstance(value.get("data"), dict)
    )


@dataclass
class ToolOutcome:
    """Normalized tool result for prompts, traces, and the attempt ledger.

    Graded-conservatism fields (all optional, additive, default-neutral) let a
    refusal carry *why* it happened and *how far* it was from the threshold, so the
    agent can choose to relax, retry, or escalate instead of dead-ending:

    - ``refused`` / ``refusal_code`` — a machine-readable refusal (not just ``ok=False``).
    - ``margin_m`` — signed distance to the checked threshold (negative = failed by this much).
    - ``threshold_m`` — the threshold that was checked.
    - ``relax_allowed`` — a bounded relaxation retry is permitted.
    """

    ok: bool
    status: str = ""
    note: str = ""
    tool: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    margin_m: float | None = None
    threshold_m: float | None = None
    refused: bool = False
    refusal_code: str = ""
    relax_allowed: bool = False

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
        # Graded-conservatism fields win over any same-named payload key.
        if self.refused:
            d["refused"] = True
        if self.refusal_code:
            d["refusal_code"] = self.refusal_code
        if self.margin_m is not None:
            d["margin_m"] = float(self.margin_m)
        if self.threshold_m is not None:
            d["threshold_m"] = float(self.threshold_m)
        if self.relax_allowed:
            d["relax_allowed"] = True
        return d

    def render(self) -> str:
        """Render public TAMP JSON intact; other outcomes use compact labeled prose."""
        if self.tool in TAMP_TOOLS and self.payload.get("schema_version") == SCHEMA_VERSION:
            # Continuation and historical task summaries consume the same public
            # envelope. Prose rendering would lose plan handles/partial progress.
            return f"[{self.tool}] " + json.dumps(self.payload, allow_nan=False)
        name = self.tool or "tool"
        head = f"[{name}] ok" if self.ok else f"[{name}] failed"
        bits = [head]
        if self.status:
            bits.append(f"status={self.status}")
        if self.note:
            bits.append(self.note)
        if self.refused:
            bits.append(f"refused={self.refusal_code or 'unknown'}")
            if self.margin_m is not None and self.threshold_m is not None:
                bits.append(f"margin={self.margin_m:+.3f}m vs {self.threshold_m:.3f}m")
                if self.relax_allowed:
                    bits.append("(relaxable)")
        msg = self.payload.get("message") or self.payload.get("error")
        if msg and str(msg) not in (self.note, self.status):
            bits.append(str(msg)[:200])
        # Actionable details must reach the model, not just the trace ledger.
        details = {
            key: self.payload[key]
            for key in (
                "manipulation",
                "navigation",
                "observation",
                "recovery_tools",
                "suggested_action",
                "capture",
                "camera_motion",
                "visual",
                "geometry",
                "task_view",
            )
            if key in self.payload
        }
        if self.tool in {"rotate_base", "move_forward"} and "data" in self.payload:
            details["motion"] = self.payload["data"]
        verification = self.payload.get("verify")
        if isinstance(verification, dict):
            # Keep acquisition/geometry failures visible to the coordinator
            # without copying auxiliary prompts or embedding vectors.
            details["verify"] = {
                key: verification[key]
                for key in ("ok", "status", "obs_id", "view_id", "visual", "geometry")
                if key in verification
            }
        if details:
            bits.append(json.dumps(details, default=str))
        return " ".join(bits)

    @classmethod
    def from_refusal(
        cls,
        tool: str,
        *,
        refusal_code: str,
        margin_m: float | None = None,
        threshold_m: float | None = None,
        relax_allowed: bool = False,
        note: str = "",
        status: str = "refused",
    ) -> ToolOutcome:
        """Build a refused outcome carrying margin/threshold for graded conservatism."""
        return cls(
            ok=False,
            status=status,
            note=note,
            tool=tool,
            refused=True,
            refusal_code=refusal_code,
            margin_m=margin_m,
            threshold_m=threshold_m,
            relax_allowed=relax_allowed,
        )

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
    def from_versioned_result(cls, tool: str, value: Any, *, wrapper: ToolOutcome | None = None) -> ToolOutcome:
        """Own and validate a TAMP/motion receipt without a JSON round trip.

        Wrapper success must agree with the embedded receipt. In particular, a
        partial receipt cannot be promoted, or a failed wrapper silently erased.
        Copy nested data so a provider's later updates cannot rewrite history.
        """
        envelope = deepcopy(value) if isinstance(value, dict) else value
        if not _valid_versioned_result(tool, envelope):
            return cls(
                False,
                status="invalid_tool_result",
                tool=tool,
                note=wrapper.note if wrapper is not None else "",
                payload={"recovery": "inspect"},
            )
        success = envelope["status"] == "ok" and envelope["code"] == "ok"
        if wrapper is not None and (wrapper.tool != tool or wrapper.ok is not success):
            return cls(False, status="invalid_tool_result", tool=tool, payload={"recovery": "inspect"})
        return cls(
            ok=success,
            status=envelope["code"],
            note=(wrapper.note if wrapper is not None else "") or str(envelope.get("message", "")),
            tool=tool,
            payload=envelope,
        )

    @classmethod
    def coerce(cls, tool: str, result: Any) -> ToolOutcome:
        """Normalize versioned TAMP JSON, EQA dictionaries, and legacy prose."""
        if isinstance(result, ToolOutcome):
            if not result.tool:
                result = replace(result, tool=tool)
            if result.tool != tool:
                return cls(False, status="invalid_tool_result", tool=tool, payload={"recovery": "inspect"})
            if tool not in TAMP_TOOLS and tool not in {"rotate_base", "move_forward"}:
                return result
            return cls.from_versioned_result(tool, result.payload, wrapper=result)
        if tool in TAMP_TOOLS or tool in {"rotate_base", "move_forward"}:
            try:
                envelope = json.loads(result) if isinstance(result, str) else result
            except (ValueError, TypeError):
                envelope = None
            return cls.from_versioned_result(tool, envelope)
        if isinstance(result, dict) and "ok" in result:
            return cls.from_eqa_dict(tool, result)
        text = "" if result is None else str(result)
        # Legacy prose is information, not an action-success receipt. New CHAT
        # handlers return explicit outcomes; do not infer success from wording.
        return cls(
            ok=False,
            status="unverified_result",
            note=text[:240],
            tool=tool,
            payload={"message": text} if text else {},
        )


def retained_outcomes(results: list[ToolOutcome] | list[str]) -> list[ToolOutcome]:
    """Accept typed receipts, or decode historical rendered inputs at the boundary.

    The production loop passes typed receipts. String support keeps existing
    callers of the historical dialogue/continuation helpers working.
    """
    outcomes = []
    for result in results:
        if isinstance(result, ToolOutcome):
            if result.tool in TAMP_TOOLS:
                outcomes.append(ToolOutcome.from_versioned_result(result.tool, result.payload, wrapper=result))
            else:
                outcomes.append(result)
            continue
        prefix, separator, body = result.partition("] ")
        if separator and prefix.startswith("["):
            outcomes.append(ToolOutcome.coerce(prefix[1:], body))
        else:
            outcomes.append(ToolOutcome(False, status="invalid_tool_result"))
    return outcomes


@dataclass
class DispatchResult:
    """Typed receipts plus their model/log rendering for one ordered dispatch.

    Runtime continuation/history consume ``outcomes`` directly. Tuple unpacking
    preserves the old private dispatcher interface for integrations and tests.
    Rendering is captured once; it is never reparsed by the CHAT runtime.
    """

    continue_running: bool = True
    has_info: bool = False
    outcomes: list[ToolOutcome] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def append(self, outcome: ToolOutcome, text: str | None = None) -> None:
        self.outcomes.append(outcome)
        self.messages.append(outcome.render() if text is None else text)

    def __iter__(self):
        yield self.continue_running
        yield self.messages
        yield self.has_info


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
        "execute_pick_place_plan": "pick",
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
    if status == "unverified_result":
        outcome_str = "unknown"
    elif kind == "verify":
        st = status.upper()
        if st in {"VISUAL_ABSENT", "ABSENT"}:
            outcome_str = "absent"
        elif st in {"VISUAL_PRESENT", "PRESENT"}:
            outcome_str = "present"
        elif st in {"CANDIDATE", "NOT_ASSESSED"}:
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
