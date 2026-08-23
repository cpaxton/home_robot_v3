# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Semantic action history and deterministic retry decisions.

The grounded EQA router needs two different notions of sameness:

``work_key``
    The semantic work being attempted, such as inspecting one stable place for
    the question target.

``equivalence_key``
    One concrete way to perform that work, such as a particular approach slot
    or immutable verification view.

A retry is suppressed only when the same concrete action has already started
from the same action-specific progress token and produced no useful progress.
The policy is intentionally conservative and benchmark-scoped; dynamic-world
expiry and event invalidation belong in a later policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

ACTION_PROGRESS_MODES = frozenset({"off", "shadow", "enforce"})
ACTION_HISTORY_SCHEMA_VERSION = 1
ACTION_POLICY_VERSION = "static-v1"

OutcomeClass = Literal[
    "terminal_ok",
    "progress",
    "negative_evidence",
    "no_progress",
    "transient",
    "operator_abort",
    "capability_absent",
]

_TRANSIENT_STATUS_PARTS = (
    "timeout",
    "controller_busy",
    "temporarily",
    "rate_limit",
)
_OPERATOR_ABORT_STATUS_PARTS = ("user_cancel", "operator_cancel", "interrupted")
_CAPABILITY_STATUS_PARTS = ("not_implemented", "unavailable", "unsupported")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s-]+", flags=re.UNICODE)


def resolve_action_progress_mode(raw: Any) -> str:
    """Normalize the independent retry-policy mode."""
    value = str(raw or "").strip().lower()
    return value if value in ACTION_PROGRESS_MODES else "off"


def normalize_intent(value: Any) -> str:
    """Conservative, versioned text normalization for action identity."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def quantized_xy(value: Any, *, cell_m: float = 0.25) -> tuple[int, int] | None:
    """Return a deterministic planar grid cell, or ``None``."""
    if value is None:
        return None
    try:
        vals = list(value)
        x, y = float(vals[0]), float(vals[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    cell = max(1e-3, float(cell_m))
    return int(round(x / cell)), int(round(y / cell))


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        rows = [_canonical(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, 6)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def stable_digest(namespace: str, payload: Any) -> str:
    """Hash canonical JSON; never use process-randomized Python hashes."""
    body = {
        "namespace": str(namespace),
        "payload": _canonical(payload),
        "schema_version": ACTION_HISTORY_SCHEMA_VERSION,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _frozen_items(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            str(key),
            json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
        for key, value in sorted((values or {}).items(), key=lambda item: str(item[0]))
    )


def _item_value(items: tuple[tuple[str, str], ...], key: str) -> Any:
    raw = next((value for name, value in items if name == key), None)
    if raw is None:
        return None
    return json.loads(raw)


@dataclass(frozen=True)
class ActionTarget:
    """Stable target identity plus a frozen human-facing description."""

    kind: str
    stable_id: str
    labels: tuple[str, ...] = ()
    room: str = "unknown"
    adapter_id: int | None = None
    view_id: str | None = None
    revision: int | None = None
    xyz: tuple[float, float, float] | None = None

    @property
    def display_name(self) -> str:
        labels = tuple(label for label in self.labels if label and label != "unknown")
        return "/".join(labels[:3]) if labels else self.stable_id or self.kind or "unknown"

    def identity_dict(self) -> dict[str, str]:
        return {
            "kind": str(self.kind or "unknown"),
            "stable_id": str(self.stable_id or ""),
        }


@dataclass(frozen=True)
class ActionSignature:
    """One semantic task and one concrete executable variant."""

    schema_version: int
    policy_version: str
    tool_name: str
    family: str
    intent: str
    work_intent: str
    target: ActionTarget
    variant: tuple[tuple[str, str], ...]
    work_key: str
    equivalence_key: str

    @classmethod
    def build(
        cls,
        *,
        tool_name: str,
        family: str,
        intent: Any,
        target: ActionTarget,
        work_intent: Any | None = None,
        variant: Mapping[str, Any] | None = None,
        policy_version: str = ACTION_POLICY_VERSION,
    ) -> ActionSignature:
        intent_key = normalize_intent(intent)
        work_intent_key = normalize_intent(intent if work_intent is None else work_intent)
        family_key = str(family or tool_name or "unknown").strip().lower()
        work_payload = {
            "policy_version": policy_version,
            "family": family_key,
            "intent": work_intent_key,
            "target": target.identity_dict(),
        }
        work_key = stable_digest("action-work", work_payload)
        frozen_variant = _frozen_items(variant)
        equivalence_key = stable_digest(
            "action-equivalence",
            {
                "policy_version": policy_version,
                "work_key": work_key,
                "variant": dict(frozen_variant),
            },
        )
        return cls(
            schema_version=ACTION_HISTORY_SCHEMA_VERSION,
            policy_version=policy_version,
            tool_name=str(tool_name or "").strip().lower(),
            family=family_key,
            intent=intent_key,
            work_intent=work_intent_key,
            target=target,
            variant=frozen_variant,
            work_key=work_key,
            equivalence_key=equivalence_key,
        )

    def variant_value(self, key: str) -> Any:
        return _item_value(self.variant, key)


@dataclass(frozen=True)
class ProgressToken:
    """Action-specific material state relevant to retrying one variant."""

    schema_version: int
    components: tuple[tuple[str, str], ...]
    digest: str

    @classmethod
    def build(cls, components: Mapping[str, Any] | None = None) -> ProgressToken:
        frozen = _frozen_items(components)
        return cls(
            schema_version=ACTION_HISTORY_SCHEMA_VERSION,
            components=frozen,
            digest=stable_digest("action-progress", dict(frozen)),
        )

    def value(self, key: str) -> Any:
        return _item_value(self.components, key)


@dataclass(frozen=True)
class ActionHistoryEntry:
    """One completed top-level action with pre/post progress state."""

    schema_version: int
    round_index: int
    selected_by: str
    signature: ActionSignature
    progress_before: ProgressToken
    progress_after: ProgressToken
    outcome_class: OutcomeClass
    status: str
    ok: bool
    progress_reasons: tuple[str, ...] = ()
    closest_m: float | None = None
    capture_status: str = ""
    verify_status: str = ""
    nav_outcome: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    """Read-only decision for one candidate before routing."""

    allowed: bool
    disposition: str
    reason: str
    signature: ActionSignature
    progress: ProgressToken
    prior_rounds: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def status_outcome_class(
    *,
    family: str,
    ok: bool,
    status: Any,
    progress_reasons: Sequence[str] = (),
) -> OutcomeClass:
    """Classify retry semantics without treating all failures alike."""
    code = str(status or "").strip().lower()
    if any(part in code for part in _OPERATOR_ABORT_STATUS_PARTS):
        return "operator_abort"
    if any(part in code for part in _CAPABILITY_STATUS_PARTS):
        return "capability_absent"
    if any(part in code for part in _TRANSIENT_STATUS_PARTS):
        return "transient"
    if str(family) == "verify_view":
        if code in {"absent", "candidate", "vlm_absent", "skipped_same_view"}:
            return "negative_evidence"
        return "terminal_ok" if ok else "no_progress"
    if progress_reasons:
        return "progress"
    if str(family) in {"submit_answer", "finish"} and ok:
        return "terminal_ok"
    return "no_progress"


def decide_candidate(
    history: Sequence[ActionHistoryEntry],
    signature: ActionSignature,
    progress: ProgressToken,
) -> GateDecision:
    """Allow first/alternate/progressed work; suppress unchanged duplicates.

    A known transient result receives exactly one same-token retry. A completed
    verification of an immutable view is terminal for that view, independent of
    unrelated evidence churn.
    """
    same_work = [entry for entry in history if entry.signature.work_key == signature.work_key]
    if not same_work:
        return GateDecision(True, "allowed_first", "no prior semantic work", signature, progress)

    same_equivalence = [entry for entry in same_work if entry.signature.equivalence_key == signature.equivalence_key]
    if not same_equivalence:
        return GateDecision(
            True,
            "allowed_alternate",
            "new concrete approach/view/goal variant",
            signature,
            progress,
            tuple(entry.round_index for entry in same_work[-4:]),
        )

    if signature.family == "verify_view" and any(
        entry.outcome_class in {"terminal_ok", "negative_evidence"} for entry in same_equivalence
    ):
        return GateDecision(
            False,
            "suppressed_terminal_view",
            "this immutable view and phrase were already evaluated; obtain a new view",
            signature,
            progress,
            tuple(entry.round_index for entry in same_equivalence[-4:]),
        )

    suppressible_outcomes = {
        "terminal_ok",
        "negative_evidence",
        "no_progress",
        "transient",
        "operator_abort",
        "capability_absent",
    }
    same_start = [
        entry
        for entry in same_equivalence
        if entry.outcome_class in suppressible_outcomes
        and (entry.progress_before.digest == progress.digest or entry.progress_after.digest == progress.digest)
    ]
    if not same_start:
        return GateDecision(
            True,
            "allowed_progress",
            "target-local material state changed since the prior attempt",
            signature,
            progress,
            tuple(entry.round_index for entry in same_equivalence[-4:]),
        )

    transient = [entry for entry in same_start if entry.outcome_class == "transient"]
    if transient and len(same_start) == 1:
        return GateDecision(
            True,
            "allowed_transient_retry",
            "one fixed retry is allowed after a transient failure",
            signature,
            progress,
            (same_start[-1].round_index,),
        )

    latest = same_start[-1]
    reason = (
        "transient retry already consumed; state is still unchanged"
        if latest.outcome_class == "transient" or len(transient) >= 2
        else "same action variant already produced no new material state"
    )
    return GateDecision(
        False,
        "would_suppress_duplicate",
        reason,
        signature,
        progress,
        tuple(entry.round_index for entry in same_start[-4:]),
    )


def render_history_entry(entry: ActionHistoryEntry) -> str:
    """Deterministic semantic one-line router history."""
    signature = entry.signature
    target = signature.target
    bits = [
        f"round={int(entry.round_index)}",
        f"action={signature.tool_name}",
        f"intent={json.dumps(signature.intent, ensure_ascii=True)}",
        f"target={json.dumps(target.display_name, ensure_ascii=True)}",
        f"room={target.room or 'unknown'}",
        f"{target.kind or 'target'}={target.stable_id or 'unknown'}",
    ]
    approach = signature.variant_value("approach_index")
    if approach is not None:
        bits.append(f"approach={approach}")
    if target.view_id:
        revision = f"@rev{target.revision}" if target.revision is not None else ""
        bits.append(f"view={target.view_id}{revision}")
    if entry.closest_m is not None:
        bits.append(f"closest={float(entry.closest_m):.1f}m")
    if entry.capture_status:
        bits.append(f"capture={entry.capture_status}")
    if entry.verify_status:
        bits.append(f"verify={entry.verify_status}")
    if entry.nav_outcome:
        bits.append(f"nav={entry.nav_outcome}")
    bits.append(f"result={entry.outcome_class}:{entry.status or ('ok' if entry.ok else 'failed')}")
    bits.append(f"progress={','.join(entry.progress_reasons) or 'none'}")
    if target.adapter_id is not None:
        bits.append(f"adapter={int(target.adapter_id)}")
    return " ".join(bits)


def render_gate_decision(decision: GateDecision) -> str:
    """Human-readable, non-actionable suppression explanation."""
    target = decision.signature.target
    bits = [
        f"action={decision.signature.tool_name}",
        f"target={json.dumps(target.display_name, ensure_ascii=True)}",
        f"room={target.room or 'unknown'}",
        f"stable={target.kind}:{target.stable_id or 'unknown'}",
        f"reason={json.dumps(decision.reason, ensure_ascii=True)}",
        "retry=after target/view/geometry change",
    ]
    if target.adapter_id is not None:
        bits.append(f"adapter={int(target.adapter_id)}")
    return " ".join(bits)
