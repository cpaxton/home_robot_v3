# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded relative goals through the ordinary validated navigation path."""

from copy import deepcopy

import numpy as np

from emet.controller.task.tamp.api import response


def relative_motion_progress(before, after, tool, commanded, position_tolerance, yaw_tolerance):
    """Measured body-frame displacement and full SE(2) arrival confirmation.

    Forward motion also checks yaw: heading drift changes the resulting pose.
    This reports an unconfirmed arrival; it does not erase measured progress.
    """
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if before.shape != (3,) or after.shape != (3,) or not np.isfinite([before, after]).all():
        return None, False
    delta = after - before
    c, s = np.cos(before[2]), np.sin(before[2])
    measured = np.array(
        [c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], np.arctan2(np.sin(delta[2]), np.cos(delta[2]))]
    )
    expected = (
        np.array([commanded, 0.0, 0.0]) if tool == "move_forward" else np.array([0.0, 0.0, np.deg2rad(commanded)])
    )
    residual = measured - expected
    residual[2] = np.arctan2(np.sin(residual[2]), np.cos(residual[2]))
    arrived = np.linalg.norm(residual[:2]) <= position_tolerance and abs(residual[2]) <= yaw_tolerance
    return measured.tolist(), bool(arrived)


def supports_relative_motion(agent):
    return getattr(getattr(agent, "space", None), "obstacle_map_mode", None) == "physical" and all(
        callable(getattr(agent, name, None))
        for name in ("_current_planning_xyt", "_filter_unsafe_nav_traj", "_execute_validated_waypoints")
    )


def execute_relative_motion(agent, tool, value):
    """Never execute unchecked motion or manufacture a target object."""
    if tool not in {"move_forward", "rotate_base"}:
        return response(tool, code="unsupported_motion")
    limit = 1.5 if tool == "move_forward" else 180.0
    lower = 0.0 if tool == "move_forward" else -limit
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        return response(tool, code="invalid_motion_argument")
    if not lower <= value <= limit:
        return response(tool, code="motion_request_out_of_bounds")
    if value == 0 or (tool == "rotate_base" and abs(value) < 1e-3):
        return response(
            tool, data={"requested": value, "execution_started": False, "no_op": True, "measured_body_delta_xyt": None}
        )
    if not supports_relative_motion(agent):
        return response(tool, code="validated_motion_unavailable")
    # Recovery tools consume the current navigation attempt, not a private
    # receipt. Replace its identity so old floor captures cannot certify it.
    agent._last_nav_plan = {"action": tool, "requested": value, "footprint": {}}
    execution_started = False
    try:
        start = np.asarray(agent._current_planning_xyt(), dtype=float)
        if start.shape != (3,) or not np.isfinite(start).all():
            return response(tool, code="motion_measurement_unavailable")
        goal = start.copy()
        if tool == "move_forward":
            goal[:2] += value * np.array([np.cos(start[2]), np.sin(start[2])])
        else:
            goal[2] += np.deg2rad(value)
        goal[2] = np.arctan2(np.sin(goal[2]), np.cos(goal[2]))
        agent.planner.reset()
        _, rejection, _ = agent._filter_unsafe_nav_traj([goal], start_xyt=start, explore_goal=False)
        if rejection:
            footprint = deepcopy(getattr(agent, "_last_nav_sweep_failure", {}) or {})
            agent._last_nav_plan.update(outcome=rejection, footprint=footprint)
            return response(
                tool,
                code="motion_rejected",
                data={
                    "requested": value,
                    "execution_started": False,
                    "reason": rejection,
                    "footprint": deepcopy(footprint),
                },
            )
        execution_started = True
        ok, reason = agent._execute_validated_waypoints([goal], agent._find_phase_nav_timeout())
        end = np.asarray(agent._current_planning_xyt(), dtype=float)
        measured, arrived = relative_motion_progress(
            start,
            end,
            tool,
            value,
            float(getattr(agent, "pos_err_threshold", 0.02)),
            float(getattr(agent, "rot_err_threshold", 0.03)),
        )
        if ok is True and not arrived:
            ok, reason = False, "motion_arrival_unconfirmed"
        return response(
            tool,
            code="ok" if ok is True and measured is not None else (reason or "motion_arrival_unconfirmed"),
            data={
                "requested": value,
                "execution_started": True,
                "measured_body_delta_xyt": measured,
                "goal_world_xyt": goal.tolist(),
            },
        )
    except Exception as exc:
        return response(
            tool, code="motion_execution_error", data={"execution_started": execution_started, "error": str(exc)[:200]}
        )
