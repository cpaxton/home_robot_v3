# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Independent endpoint checks for the simulation route probe, not collision proof."""

import math

import numpy as np

from emet.core.navigation_result import NAVIGATION_POLICIES


def score_arrival_dwell(goal, samples, policy_name):
    """Check fresh episode-frame observations after a command has finished.

    Samples contain monotonic client time, observation sequence and measured
    episode-frame pose. Sequence advancement proves new messages, not sensor
    acquisition timing. The server's acquisition/settling contract is checked
    separately; this check can refute a false arrival but cannot certify contact
    safety or timing that a bridge does not publish.
    """
    policy = NAVIGATION_POLICIES[policy_name]
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError("goal must be a finite episode-frame SE(2) pose")
    fresh = []
    for sample in samples:
        sequence = sample.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            return {"status": "incomplete_telemetry", "reason": "missing_observation_sequence"}
        pose = np.asarray(sample.get("pose"), dtype=float)
        stamp = sample.get("time")
        if pose.shape != (3,) or not np.isfinite(pose).all() or stamp is None or not math.isfinite(stamp):
            return {"status": "incomplete_telemetry", "reason": "invalid_measured_pose_or_time"}
        if fresh and (sequence < fresh[-1][0] or stamp <= fresh[-1][1]):
            return {"status": "incomplete_telemetry", "reason": "nonmonotonic_observations"}
        if not fresh or sequence > fresh[-1][0]:
            fresh.append((sequence, stamp, pose))
    if len(fresh) < policy.settle_samples or fresh[-1][1] - fresh[0][1] < policy.settle_seconds:
        return {"status": "incomplete_telemetry", "reason": "insufficient_fresh_dwell"}
    poses = np.stack([s[2] for s in fresh])
    xy_errors = np.linalg.norm(poses[:, :2] - goal[:2], axis=1)
    yaw_errors = np.abs(np.arctan2(np.sin(poses[:, 2] - goal[2]), np.cos(poses[:, 2] - goal[2])))
    elapsed = fresh[-1][1] - fresh[0][1]
    # Use total path/rotation rather than endpoint displacement: returning to the
    # same pose after oscillation must not count as settled.
    speed = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum() / elapsed)
    angles = np.diff(poses[:, 2])
    turn_speed = float(np.abs(np.arctan2(np.sin(angles), np.cos(angles))).sum() / elapsed)
    result = {
        "fresh_samples": len(fresh),
        "dwell_seconds": elapsed,
        "max_xy_error_m": float(xy_errors.max()),
        "max_yaw_error_rad": float(yaw_errors.max()),
        "mean_path_speed_m_s": speed,
        "mean_turn_speed_rad_s": turn_speed,
    }
    if xy_errors.max() > policy.xy_tolerance or yaw_errors.max() > policy.yaw_tolerance:
        return {**result, "status": "failed", "reason": "outside_arrival_tolerance"}
    if speed > 0.01 or turn_speed > 0.03:
        return {**result, "status": "failed", "reason": "post_arrival_motion"}
    return {**result, "status": "passed_endpoint_dwell", "reason": "measured_arrival_and_stability"}
