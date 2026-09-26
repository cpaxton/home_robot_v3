# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Sampled footprint validation for executable base-motion segments."""

from dataclasses import dataclass

import numpy as np


def validate_navigation_sweep(space, start, waypoints, *, linear_step_m=0.025, angular_step_rad=0.05):
    """Check translation and the shortest yaw sweep, including the measured start.

    Check every sampled yaw at each sampled XY: controllers may overlap turning
    and translation. This is deliberately conservative, not a dynamics/contact
    proof. Unknown footprint cells remain invalid under the space's contract.
    """
    if not np.isfinite([linear_step_m, angular_step_rad]).all() or min(linear_step_m, angular_step_rad) <= 0:
        raise ValueError("Sweep resolution must be positive")
    previous = np.asarray(start, dtype=float)
    if previous.shape != (3,) or not np.isfinite(previous).all():
        return False, "invalid_navigation_pose"
    if not space.is_valid(previous):
        return False, getattr(space, "last_validity", {}).get("reason", "invalid_footprint")
    for raw in waypoints:
        target = np.asarray(raw, dtype=float)
        if previous.shape != (3,) or target.shape != (3,) or not np.isfinite([previous, target]).all():
            return False, "invalid_navigation_pose"
        yaw_delta = np.arctan2(np.sin(target[2] - previous[2]), np.cos(target[2] - previous[2]))
        n_xy = max(1, int(np.ceil(np.linalg.norm(target[:2] - previous[:2]) / linear_step_m)))
        n_yaw = max(1, int(np.ceil(abs(yaw_delta) / angular_step_rad)))
        for fraction in np.linspace(0, 1, n_xy + 1):
            xy = previous[:2] + fraction * (target[:2] - previous[:2])
            for yaw in np.linspace(previous[2], previous[2] + yaw_delta, n_yaw + 1):
                if not space.is_valid(np.array([*xy, yaw])):
                    return False, getattr(space, "last_validity", {}).get("reason", "invalid_footprint")
        previous = target
    return True, None


@dataclass
class RouteExecutionResult:
    success: bool
    reason: str
    current_pose: list[float]
    goal_pose: list[float]
    position_residual_m: float | None
    yaw_residual_rad: float | None
    replans: int


def execute_measured_route(
    robot,
    *,
    goal,
    measure,
    plan_route,
    space,
    max_replans=2,
    position_tolerance_m=0.05,
    yaw_tolerance_rad=0.1,
    waypoint_timeout_s=30.0,
    navigation_policy=None,
    event=lambda **kwargs: None,
):
    """Follow validated waypoints with shared base control and bounded replans.

    ``measure`` and ``goal`` use the same world frame. ``plan_route`` must use
    current geometry and return XYT waypoints, never a mere reachability bool.
    """
    target = np.asarray(goal, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or max_replans < 0:
        raise ValueError("Finite XYT goal and nonnegative replan budget required")
    tolerances = [position_tolerance_m, yaw_tolerance_rad, waypoint_timeout_s]
    if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
        raise ValueError("Finite positive execution tolerances and timeout required")

    def valid_measurement(pose):
        return pose.shape == (3,) and np.isfinite(pose).all()

    def succeeded(value):
        return isinstance(value, (bool, np.bool_)) and bool(value)

    def read_measurement():
        try:
            return np.asarray(measure(), dtype=float)
        except (RuntimeError, ValueError) as exc:
            event(phase="navigation_state", reason=str(exc))
            return np.array([])

    def invalid_measurement(attempt, *, cancel_motion):
        reason = "invalid_measured_pose"
        if cancel_motion:
            cancel = getattr(robot, "cancel_navigation", None)
            if cancel is None or not succeeded(cancel()):
                reason += ":navigation_cancellation_unconfirmed"
        return RouteExecutionResult(False, reason, [], target.tolist(), None, None, attempt)

    current = read_measurement()
    if not valid_measurement(current):
        return invalid_measurement(0, cancel_motion=False)
    reason = "no_plan_within_budget"
    for attempt in range(max_replans + 1):
        route = plan_route(current.copy(), target.copy())
        if not route:
            detail = getattr(space, "last_validity", {}).get("reason")
            reason = f"no_plan_within_budget:{detail}" if detail and detail != "ok" else "no_plan_within_budget"
            break
        accepted, rejected = validate_navigation_sweep(space, current, route)
        if not accepted:
            reason = f"rejected_swept_footprint:{rejected}"
            break
        diverged = False
        for waypoint in route:
            waypoint = np.asarray(waypoint, dtype=float)
            current = read_measurement()
            if not valid_measurement(current):
                return invalid_measurement(attempt, cancel_motion=True)
            accepted, rejected = validate_navigation_sweep(space, current, [waypoint])
            if not accepted:
                reason = f"rejected_swept_footprint:{rejected}"
                diverged = True
                break
            options = {"navigation_policy": navigation_policy} if navigation_policy is not None else {}
            ok = succeeded(robot.move_base_to(waypoint, blocking=True, world_frame=True, timeout=waypoint_timeout_s, **options))
            current = read_measurement()
            if not valid_measurement(current):
                return invalid_measurement(attempt, cancel_motion=True)
            xy = float(np.linalg.norm(current[:2] - waypoint[:2]))
            yaw = float(abs(np.arctan2(np.sin(current[2] - waypoint[2]), np.cos(current[2] - waypoint[2]))))
            receipt = getattr(robot, "_command_receipt", None)
            controller_result = receipt.get("result") if isinstance(receipt, dict) else None
            event(
                phase="navigation",
                command=waypoint.tolist(),
                measured=current.tolist(),
                position_residual_m=xy,
                yaw_residual_rad=yaw,
                controller_success=bool(ok),
                controller_result=controller_result,
                replan=attempt,
            )
            if not ok or not np.isfinite(current).all() or xy > position_tolerance_m or yaw > yaw_tolerance_rad:
                reason = "base_tracking_failed"
                if isinstance(controller_result, dict) and controller_result.get("reason"):
                    reason += ":" + str(controller_result["reason"])
                diverged = True
                break
        if not diverged:
            xy = float(np.linalg.norm(current[:2] - target[:2]))
            yaw = float(abs(np.arctan2(np.sin(current[2] - target[2]), np.cos(current[2] - target[2]))))
            if xy <= position_tolerance_m and yaw <= yaw_tolerance_rad:
                return RouteExecutionResult(True, "ok", current.tolist(), target.tolist(), xy, yaw, attempt)
            reason = "route_did_not_reach_goal"
        # Stop the controller before observing/replanning after divergence.
        cancel = getattr(robot, "cancel_navigation", None)
        if cancel is None or not succeeded(cancel()):
            reason = "navigation_cancellation_unconfirmed"
            break
        current = read_measurement()
        if not valid_measurement(current):
            return invalid_measurement(attempt, cancel_motion=False)
    xy = float(np.linalg.norm(current[:2] - target[:2]))
    yaw = float(abs(np.arctan2(np.sin(current[2] - target[2]), np.cos(current[2] - target[2]))))
    return RouteExecutionResult(False, reason, current.tolist(), target.tolist(), xy, yaw, attempt)
