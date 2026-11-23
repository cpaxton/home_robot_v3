# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded CHAT continuation policy for structured manipulation outcomes."""

from emet.agent.tool_outcome import ToolOutcome, retained_outcomes
from emet.controller.task.tamp.api import TAMP_TOOLS


def serial_tamp_calls(tool_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    """Defer later proposals in all-TAMP batches until the first result is seen.

    Planning produces fresh handles that cannot be known by a precomputed
    execution call. Deferred proposals are logged, never automatically queued.
    Mixed capability batches retain the dispatcher's existing stop semantics.
    """
    if len(tool_calls) > 1 and all(call.get("name") in TAMP_TOOLS for call in tool_calls):
        return tool_calls[:1], tool_calls[1:]
    return tool_calls, []


def tamp_followup(user_goal: str, results: list[ToolOutcome] | list[str], *, recovery_attempts: int = 0) -> str | None:
    """Continue success or bounded, explicitly unstarted planning recovery.

    Partial or unknown execution never gains autonomous retries.
    """
    if not results:
        return None
    outcomes = retained_outcomes(results)
    for outcome in outcomes:
        if outcome.tool not in TAMP_TOOLS:
            return None
        if not outcome.ok:
            data = outcome.payload.get("data", {})
            recoverable = {
                "geometry_tracking_no_match", "geometry_tracking_budget_exhausted",
                "moving_tracking_time_budget_exhausted", "moving_tracking_lost_reinitialize",
                "moving_multiview_association_unknown", "moving_geometry_ambiguous",
                "object_pose_ambiguous", "support_geometry_unknown",
                "background_scene_changed", "scene_changed_replan", "planning_failed", "ik_failed",
                "no_plan_within_budget", "no_reachable_grasp",
            }
            if (
                len(outcomes) != 1 or recovery_attempts >= 2
                or outcome.tool not in {"plan_pick_place", "plan_task"}
                or outcome.payload.get("status") != "error"
                or outcome.payload.get("code") not in recoverable
                or data.get("execution_started") is not False
                or data.get("completed_ops") or data.get("completed_steps")
            ):
                return None
            return (
                f"Original user request:\n{user_goal}\n\n{outcome.render()}\n\n"
                "Planning failed before any execution. Within the original authorized task, "
                "inspect a fresh camera view or query the VLM to clarify object, support, and "
                "approach geometry; then request a fresh plan. Use at most one tool per round. "
                "Change the view or approach rather than repeating the same failed request. "
                "A language or visual judgment is a hypothesis, not measured clearance. "
                "Use only explicit, recorded runtime overrides; never invent an override or "
                "claim a failed check passed. Joint bounds, fresh feedback, and stop control "
                "remain required. Do not replay consumed handles. If evidence stays ambiguous, "
                "report the blocker. This turn permits at most two planning recovery attempts."
            )
        if outcome.payload.get("data", {}).get("discovery_only") is True:
            # A proposal provider cannot authorize a motion-capable next step.
            return None
    return (
        f"Original user request:\n{user_goal}\n\n[Tool results]\n"
        + "\n".join(outcome.render() for outcome in outcomes)
        + "\n\nContinue only unfinished steps requested by the user. If the request was "
        "only to discover or plan, summarize and stop; do not execute without authorization. "
        "For an authorized multi-step task, return at most one next tool call. Copy handles "
        "exactly; after execution rediscover before planning the next task. Never repeat a "
        "consumed plan or silently drop goal constraints. If all requested work is complete, "
        "summarize the measured outcomes and stop."
    )
