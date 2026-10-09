# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Bounded floor sensing inside one navigation decision, without bypassing planning."""

from emet.controller.dynamem.look import supports_floor_observation
from emet.controller.nav_attempt import nav_status_code


def navigate_with_floor_recovery(executor, navigate):
    """Retry the same goal from measured pose after clearing its checked footprint.

    ``navigate`` must invoke the ordinary planner on every call. Two recovery
    observations (each already bounded by the floor tool) belong to this one
    high-level decision. Occupancy, unchanged/invalid footprints and unsupported
    sensors stop immediately. We never infer route clearance from map growth.
    """
    for attempt in range(3):
        previous = getattr(executor.agent, "_last_nav_attempt", None)
        outcome = navigate()
        current = getattr(executor.agent, "_last_nav_attempt", None)
        if (
            attempt == 2
            or current is None
            or current is previous
            or nav_status_code(current) != "rejected_swept_footprint:unobserved_footprint"
            or not supports_floor_observation(executor.agent)
        ):
            return outcome
        recovery = executor.handle_tool("observe_floor", {"target_blocker": True})
        observation = recovery.get("observation") or {}
        resume = bool(recovery.get("ok") and (observation.get("footprint_after") or {}).get("pose_valid"))
        executor._append_trace(
            {
                "event": "navigation_floor_recovery",
                "attempt": attempt + 1,
                "status": recovery.get("status"),
                "resume_same_goal": resume,
                "checked_pose": observation.get("checked_pose"),
            }
        )
        if not resume:
            return outcome
    return outcome
