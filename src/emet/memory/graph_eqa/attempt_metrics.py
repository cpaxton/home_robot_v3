# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Repeat-failure metrics over an :class:`AttemptRecord` ledger (Phase 4).

A repeat is a non-ok attempt whose key matches a prior non-ok attempt. Stable
schema-v3 target/view identities take precedence over mutable node/observation
adapters, with legacy fallbacks for older ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emet.memory.graph_eqa.attempt_ledger import AttemptRecord

_BAD = frozenset({"failed", "aborted", "absent", "unreachable"})


@dataclass(frozen=True)
class RepeatFailureStats:
    n_attempts: int
    n_failures: int
    n_repeat_failures: int
    n_wasted_rounds: int
    by_kind: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_attempts": int(self.n_attempts),
            "n_failures": int(self.n_failures),
            "n_repeat_failures": int(self.n_repeat_failures),
            "n_wasted_rounds": int(self.n_wasted_rounds),
            "by_kind": dict(self.by_kind),
        }


def _key(rec: AttemptRecord, *, xyz_tol_m: float) -> tuple[Any, ...]:
    kind = rec.action_kind
    if rec.target_id:
        view_id = rec.view_id if kind == "verify" else ""
        return (kind, rec.target_kind or "target", rec.target_id, view_id)
    if rec.target_node_id is not None:
        return (kind, "node", int(rec.target_node_id))
    if rec.obs_id is not None:
        return (kind, "obs", int(rec.obs_id))
    if rec.xyz is not None:
        # Quantize planar XY so nearby retries collide.
        q = max(1e-3, float(xyz_tol_m))
        x = round(float(rec.xyz[0]) / q) * q
        y = round(float(rec.xyz[1]) / q) * q
        return (kind, "xy", round(x, 3), round(y, 3))
    phrase = (rec.phrase or "").strip().lower()
    if phrase:
        return (kind, "phrase", phrase)
    return (kind, "anon", rec.status_code or "")


def summarize_repeat_failures(
    records: list[AttemptRecord] | None,
    *,
    xyz_tol_m: float = 0.25,
    kinds: set[str] | None = None,
) -> RepeatFailureStats:
    """Compute repeat-failure counts from an ordered ledger (oldest first)."""
    rows = list(records or [])
    if kinds:
        allowed = {str(k).lower() for k in kinds}
        rows = [r for r in rows if r.action_kind in allowed]
    n_attempts = len(rows)
    n_failures = 0
    n_repeat = 0
    by_kind: dict[str, int] = {}
    seen_failed: set[tuple[Any, ...]] = set()
    for r in rows:
        if r.outcome not in _BAD:
            continue
        n_failures += 1
        k = _key(r, xyz_tol_m=xyz_tol_m)
        if k in seen_failed:
            n_repeat += 1
            by_kind[r.action_kind] = by_kind.get(r.action_kind, 0) + 1
        else:
            seen_failed.add(k)
    return RepeatFailureStats(
        n_attempts=n_attempts,
        n_failures=n_failures,
        n_repeat_failures=n_repeat,
        n_wasted_rounds=n_repeat,
        by_kind=by_kind,
    )


def _xyz_tuple(xyz: Any) -> tuple[float, float, float] | None:
    if xyz is None:
        return None
    try:
        vals = [float(x) for x in list(xyz)[:3]]
    except Exception:
        return None
    if len(vals) < 3:
        return None
    return (vals[0], vals[1], vals[2])


def record_manip_attempt(
    graph_memory: Any,
    *,
    action_kind: str,
    success: bool,
    phrase: str = "",
    status_code: str = "",
    note: str = "",
    xyz: Any = None,
    source: str = "unknown",
) -> None:
    """Best-effort pick/place ledger write (no-op when ledger off / no memory)."""
    if graph_memory is None or not hasattr(graph_memory, "record_attempt"):
        return
    kind = str(action_kind or "").strip().lower()
    if kind not in {"pick", "place"}:
        return
    outcome = "ok" if success else "failed"
    code = (status_code or ("ok" if success else "failed")).strip()
    try:
        graph_memory.record_attempt(
            action_kind=kind,
            outcome=outcome,
            status_code=code[:80],
            note=str(note or "")[:240],
            phrase=str(phrase or ""),
            xyz=_xyz_tuple(xyz),
            source=source,
        )
    except Exception:
        pass
