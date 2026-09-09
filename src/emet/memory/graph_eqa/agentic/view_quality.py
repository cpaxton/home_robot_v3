# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Bounded head-only recovery for obstructed exploration arrival views."""

import math

import numpy as np

from emet.memory.graph_eqa.agentic.views import captured_view


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
