# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Structured navigation attempt results for planners and the action-outcome ledger.

``NavAttemptResult`` (in ``habitat_nav``) is the return shape of
``navigate_to_target_pose``. This module maps those results onto stable
``status_code`` / ledger fields so CHAT and EQA share one vocabulary.
"""

from __future__ import annotations

from typing import Any

from emet.controller.habitat_nav import NavAttemptResult
from emet.memory.graph_eqa.attempt_ledger import infer_nav_outcome, infer_nav_status_code


def nav_status_code(nav_res: NavAttemptResult | None) -> str:
    """Stable status_code for a nav attempt (ledger / ``_last_nav_plan``)."""
    if nav_res is None:
        return "failed"
    ok = bool(nav_res.success) or bool(nav_res.finished)
    # Prefer an explicit field when callers set it.
    explicit = getattr(nav_res, "status_code", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return infer_nav_status_code(success=ok, note=str(nav_res.note or ""))


def nav_outcome(nav_res: NavAttemptResult | None) -> str:
    if nav_res is None:
        return "failed"
    ok = bool(nav_res.success) or bool(nav_res.finished)
    return infer_nav_outcome(success=ok, status_code=nav_status_code(nav_res))


def apply_status_code(nav_res: NavAttemptResult) -> NavAttemptResult:
    """Ensure ``nav_res.status_code`` is populated (mutates and returns ``nav_res``)."""
    code = nav_status_code(nav_res)
    try:
        nav_res.status_code = code  # type: ignore[attr-defined]
    except Exception:
        pass
    return nav_res


def sync_nav_plan_meta(agent: Any, nav_res: NavAttemptResult | None) -> None:
    """Stamp ``status_code`` / ``outcome`` onto ``agent._last_nav_plan`` when present."""
    if agent is None or nav_res is None:
        return
    meta = getattr(agent, "_last_nav_plan", None)
    if not isinstance(meta, dict):
        meta = {}
    code = nav_status_code(nav_res)
    meta.setdefault("outcome", str(nav_res.note or code))
    meta["status_code"] = code
    meta["nav_success"] = bool(nav_res.success)
    meta["nav_finished"] = bool(nav_res.finished)
    meta["nav_dist_m"] = float(nav_res.dist_m or 0.0)
    meta["nav_method"] = str(nav_res.method or "")
    agent._last_nav_plan = meta


def sync_nav_attempt_to_ledger(
    agent: Any,
    nav_res: NavAttemptResult | None,
    *,
    source: str = "unknown",
) -> None:
    """Sync a nav attempt into ``agent.graph_memory`` (node counters + optional ledger).

    Always updates graph node ``nav_attempts`` / ``nav_failures`` via
    ``record_nav_attempt`` when graph memory is present. Ledger *rows* are
    appended only when ``eqa.attempt_ledger`` / ``EMET_EQA_ATTEMPT_LEDGER`` is on
    (gated inside ``GraphEQAMemory.record_attempt``). Safe no-op when there is
    no graph memory.
    """
    if agent is None or nav_res is None:
        return
    apply_status_code(nav_res)
    sync_nav_plan_meta(agent, nav_res)
    gm = getattr(agent, "graph_memory", None)
    if gm is None or not hasattr(gm, "record_nav_attempt"):
        return
    oid = getattr(nav_res, "target_obs_id", None)
    goal = getattr(nav_res, "goal_xy", None) or getattr(nav_res, "effective_goal_xy", None)
    xyz = None
    if goal is not None and len(goal) >= 2:
        xyz = (float(goal[0]), float(goal[1]), 0.0)
    try:
        gm.record_nav_attempt(
            int(oid) if oid is not None else None,
            success=bool(nav_res.success) or bool(nav_res.finished),
            note=str(nav_res.note or ""),
            dist_m=float(nav_res.dist_m or 0.0),
            status_code=nav_status_code(nav_res),
            source=source,
            target_node_id=None,
        )
        # When obs_id is None, still append a ledger row keyed by xyz via record_attempt.
        if oid is None and hasattr(gm, "record_attempt") and xyz is not None:
            gm.record_attempt(
                action_kind="navigate",
                outcome=nav_outcome(nav_res),
                status_code=nav_status_code(nav_res),
                note=str(nav_res.note or ""),
                xyz=xyz,
                source=source,
            )
    except Exception:
        pass
