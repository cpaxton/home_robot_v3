# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded head-only recovery for obstructed exploration arrival views."""

import math
import time

import numpy as np

from emet.memory.graph_eqa.agentic.views import captured_view, target_in_view


def aim_arrival_view(executor, capture, target_xyz):
    """Acquire an in-frame proposal view with at most two head corrections.

    Optical projection guides gaze only, never identity or visibility through
    occluders. Missing geometry preserves the legacy path; measured off-screen
    geometry must not be passed to target verification as negative evidence.
    """
    robot = getattr(executor.agent, "robot", None)
    reset_used = False
    reason = "target_outside_view"
    for attempt in range(3):
        view = captured_view(executor, capture.get("obs_id"))
        if not capture.get("ok") or view is None:
            return capture
        projection = target_in_view(view, target_xyz)
        executor._append_trace({"event": "arrival_view_aim", "attempt": attempt, "obs_id": view.obs_id, **projection})
        if projection.get("target_in_frame") is True:
            return capture
        if projection.get("target_in_frame") is None:
            # Preserve nonphysical adapters; physical inspection must have geometry.
            if not isinstance(getattr(robot, "_seq_id", None), int):
                return capture
            reason = "missing_geometry"
            break
        if attempt == 2:
            break
        head_to = getattr(robot, "head_to", None)
        get_angles = getattr(robot, "get_pan_tilt", None)
        if not callable(head_to) or not callable(get_angles):
            reason = "unsupported_head_control"
            break
        if not isinstance(getattr(robot, "_seq_id", None), int):
            reason = "observation_freshness_unavailable"
            break
        try:
            x, y, z = projection["target_camera_xyz"]
            pan, tilt = get_angles()
            delta_pan = np.clip(math.atan2(-x, z), -math.pi / 4, math.pi / 4)
            delta_tilt = np.clip(math.atan2(-y, math.hypot(x, z)), -math.pi / 4, math.pi / 4)
            if not np.isfinite([pan, tilt, delta_pan, delta_tilt]).all():
                reason = "head_pose_unconfirmed"
                break
            if projection["status"] == "behind_camera":
                if reset_used:
                    reason = "target_behind_camera_after_reset"
                    break
                # Existing horizontal forward convention; no base motion.
                commanded = np.array([0.0, 0.0])
                reset_used = True
            else:
                commanded = np.array([pan + delta_pan, tilt + delta_tilt])
            # The adapter owns safety clipping. Require measured arrival at the
            # requested pose; never mistake a clipped/stalled motion for success.
            if head_to(*map(float, commanded), blocking=True) is False:
                reason = "head_motion_failed"
                break
            deadline = time.monotonic() + 5.0
            measured = np.asarray(get_angles(), dtype=float)
            while np.isfinite(measured).all() and np.max(np.abs(measured - commanded)) > 0.12:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
                measured = np.asarray(get_angles(), dtype=float)
            executor._append_trace(
                {
                    "event": "arrival_head_motion",
                    "requested": commanded.tolist(),
                    "measured": measured.tolist(),
                    "forward_reset": reset_used,
                }
            )
            if not np.isfinite(measured).all() or np.max(np.abs(measured - commanded)) > 0.12:
                reason = "head_pose_unconfirmed"
                break
            from emet.controller.dynamem.look import wait_post_motion_obs

            sequence = robot._seq_id
            wait_post_motion_obs(robot, timeout=5.0)
            if robot._seq_id <= sequence:
                reason = "stale_observation"
                break
            previous_id = capture.get("obs_id")
            capture = executor._tool_capture_and_update()
            if not capture.get("ok") or capture.get("obs_id") == previous_id:
                reason = "stale_capture"
                break
        except (RuntimeError, ValueError, TypeError, TimeoutError) as exc:
            executor._append_trace({"event": "arrival_view_aim_failed", "error": str(exc)})
            reason = "head_control_error"
            break
    return {**capture, "ok": False, "status": "TARGET_OUTSIDE_VIEW", "reason": reason}


def exploration_view_quality(depth):
    """Judge central free viewing distance, not target presence or brightness.

    Missing depth is unknown, never evidence of an obstruction. The conservative
    gate triggers only when most central rays hit very close geometry or have
    no valid range; blank walls farther away remain legitimate observations.
    """
    if depth is None:
        return {"obstructed": False, "reason": "depth_unavailable"}
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth)
    if depth.ndim != 2 or min(depth.shape) < 4:
        return {"obstructed": False, "reason": "depth_unavailable"}
    h, w = depth.shape
    center = depth[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    valid = np.isfinite(center) & (center > 0)
    near = valid & (center < 0.6)
    blocked_fraction = float(np.mean(~valid | near))
    return {
        "obstructed": blocked_fraction >= 0.8,
        "blocked_fraction": blocked_fraction,
        "valid_fraction": float(valid.mean()),
        "reason": "central_depth",
    }


def recover_exploration_view(executor, capture):
    """At most two head turns/captures; never translate or rotate the base.

    Return only the current captured view, not an earlier best view whose pose
    has become stale. All attempted RGBs use the normal diagnostic capture.
    """
    robot = getattr(executor.agent, "robot", None)
    head_to = getattr(robot, "head_to", None)
    for attempt in range(3):
        if not capture.get("ok"):
            return capture
        view = captured_view(executor, capture.get("obs_id"))
        frames = getattr(getattr(executor.agent, "voxel_map", None), "observations", ())
        if view is None or view.source_obs_id != len(frames):
            return capture
        quality = exploration_view_quality(getattr(frames[-1], "depth", None))
        executor._append_trace(
            {"event": "exploration_view_quality", "obs_id": view.obs_id, "attempt": attempt, **quality}
        )
        if not quality["obstructed"]:
            return capture
        if attempt == 2 or not callable(head_to):
            break
        # Same level-head convention already used by exploration arrival.
        pan = math.pi / 4 if attempt == 0 else -math.pi / 4
        try:
            result = head_to(pan, 0.0, blocking=True)
            if result is False:
                break
            wait = getattr(robot, "wait_for_obs", None)
            if callable(wait):
                wait(timeout=5.0)
            capture = executor._tool_capture_and_update()
        except (RuntimeError, ValueError, TypeError, TimeoutError) as exc:
            executor._append_trace({"event": "exploration_view_recovery_failed", "error": str(exc)})
            break
    # Navigation success is not visual evidence. Keep the observation for map
    # diagnostics but do not run target verification on an obstructed arrival.
    return {**capture, "ok": False, "status": "OBSTRUCTED_VIEW"}
