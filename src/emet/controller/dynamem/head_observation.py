# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""One measured, stationary head inspection shared by task and floor views."""

import numpy as np

from emet.controller.dynamem.constants import DYNAMEM_HEAD_ARRIVAL_TOL_RAD
from emet.robots.head_capability import HeadCapability


def task_head_capability(robot):
    """Advertise explicit gaze only with angular limits and a capture contract."""
    getter = getattr(robot, "get_head_capability", None)
    try:
        capability = getter() if callable(getter) else None
    except (ConnectionError, TimeoutError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(capability, HeadCapability):
        return None
    capture = callable(getattr(robot, "capture_head_observation", None))
    stream = isinstance(getattr(robot, "_seq_id", None), int)
    methods = ("head_to", "get_pan_tilt", "get_observation", "get_base_pose")
    if not (capture or stream) or not all(callable(getattr(robot, name, None)) for name in methods):
        return None
    return capability if any(hi > lo for lo, hi in (capability.pan, capability.tilt)) else None


def capture_head_view(agent, *, pan_rad=None, tilt_rad=None, bounds=None, require_depth=False):
    """Return (receipt, captured observation); never sweep, clip or move the base.

    Task views require declared angular limits. Floor callers can supply their
    existing narrower bounds, including their conservative legacy fallback.
    Advancing reception is not a claim about sensor acquisition timestamps.
    """
    try:
        return _capture_head_view(agent, pan_rad=pan_rad, tilt_rad=tilt_rad, bounds=bounds, require_depth=require_depth)
    except (ConnectionError, TimeoutError, RuntimeError, ValueError, TypeError) as exc:
        return {"ok": False, "status": "head_observation_failed", "reason": str(exc)}, None


def _capture_head_view(agent, *, pan_rad, tilt_rad, bounds, require_depth):
    import time

    from .look import wait_post_motion_obs

    robot = getattr(agent, "robot", None)
    if bounds is None:
        getter = getattr(robot, "get_head_capability", None)
        capability = getter() if callable(getter) else None
        if not isinstance(capability, HeadCapability):
            return {"ok": False, "status": "head_limits_unavailable"}, None
        bounds = (capability.pan, capability.tilt)
    detail = {"head_inspection_bounds_rad": {"pan": list(bounds[0]), "tilt": list(bounds[1])}}

    def failure(status):
        return {"ok": False, "status": status, "observation": detail}, None

    for name, value, limits in zip(("pan", "tilt"), (pan_rad, tilt_rad), bounds, strict=True):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not limits[0] <= value <= limits[1]
        ):
            return failure(f"invalid_head_{name}")
    if not all(callable(getattr(robot, method, None)) for method in ("head_to", "get_pan_tilt")):
        return failure("unsupported_head_observation")
    native_capture = getattr(robot, "capture_head_observation", None)
    streamed = isinstance(getattr(robot, "_seq_id", None), int)
    if not streamed and not callable(native_capture):
        return failure("observation_freshness_unavailable")
    previous = np.asarray(robot.get_pan_tilt(), dtype=float)
    if previous.shape != (2,) or not np.isfinite(previous).all():
        return failure("head_pose_unconfirmed")
    target = [previous[0] if pan_rad is None else pan_rad, previous[1] if tilt_rad is None else tilt_rad]
    detail["requested_head_pan_tilt_rad"] = list(map(float, target))
    if any(not lo <= value <= hi for value, (lo, hi) in zip(target, bounds, strict=True)):
        return failure("head_pose_outside_limits")
    get_base = getattr(robot, "get_base_pose", None)
    if not callable(get_base):
        return failure("base_pose_unavailable")
    base_before = np.asarray(get_base(), dtype=float).copy()
    if base_before.shape != (3,) or not np.isfinite(base_before).all():
        return failure("base_pose_unavailable")
    if robot.head_to(*map(float, target), blocking=True) is False:
        return failure("head_motion_failed")
    if streamed:
        sequence = robot._seq_id
        wait_post_motion_obs(robot, timeout=5.0)
        if robot._seq_id <= sequence:
            return failure("stale_observation")
    deadline = time.monotonic() + 5.0
    while True:
        measured = np.asarray(robot.get_pan_tilt(), dtype=float)
        detail["measured_head_pan_tilt_rad"] = measured.tolist()
        if measured.shape != (2,) or not np.isfinite(measured).all():
            return failure("head_pose_unconfirmed")
        if np.max(np.abs(measured - target)) <= DYNAMEM_HEAD_ARRIVAL_TOL_RAD:
            break
        if time.monotonic() >= deadline:
            return failure("head_pose_unconfirmed")
        time.sleep(0.05)
    obs = None
    if callable(native_capture):
        obs = native_capture(timeout=5.0)
        if obs is None:
            return failure("stale_observation")
    else:
        sequence = robot._seq_id
        wait_post_motion_obs(robot, timeout=5.0)
        if robot._seq_id <= sequence:
            return failure("stale_observation")
    base_after = np.asarray(get_base(), dtype=float)
    if base_after.shape != (3,) or not np.isfinite(base_after).all():
        return failure("base_pose_unavailable")
    detail.update(base_before_xyt=base_before.tolist(), base_after_xyt=base_after.tolist())
    yaw = np.arctan2(np.sin(base_after[2] - base_before[2]), np.cos(base_after[2] - base_before[2]))
    if np.linalg.norm(base_after[:2] - base_before[:2]) > float(getattr(agent, "pos_err_threshold", 0.02)) or abs(
        yaw
    ) > float(getattr(agent, "rot_err_threshold", 0.03)):
        return failure("base_moved_during_head_observation")
    getter = getattr(robot, "get_observation", None)
    if obs is None:
        obs = getter() if callable(getter) else None
    if obs is None or getattr(obs, "rgb", None) is None:
        return failure("observation_unavailable")
    if require_depth and (obs.depth is None or obs.get_xyz_in_world_frame() is None):
        return failure("calibrated_depth_unavailable")
    detail["received_frame_sequence"] = getattr(robot, "_seq_id", None)
    detail["capture_confirmation"] = "adapter_post_arrival_capture" if callable(native_capture) else "stream_reception"
    detail["image_timing"] = getattr(obs, "image_timing", None)
    return {
        "ok": True,
        "status": "head_view_observed",
        "observation": detail,
        "note": "Measured head arrival and subsequent frame reception; base stationary. Sensor acquisition timing is not independently verified. No route or object geometry is certified.",
    }, obs
