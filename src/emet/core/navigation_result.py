# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Measured planar arrival, independent of any particular robot or controller."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NavigationPolicy:
    xy_tolerance: float
    yaw_tolerance: float
    settle_seconds: float = 0.5
    settle_samples: int = 3
    stale_seconds: float = 1.0
    progress_seconds: float = 5.0
    max_corrections: int = 2


NAVIGATION_POLICIES = {
    "exploration": NavigationPolicy(0.07, 0.15),
    "precision": NavigationPolicy(0.02, 0.03),
}


class ArrivalMonitor:
    """Per-command measured settling; adapters own motion and stop confirmation.

    ``sample_time`` must advance only on a new pose measurement. It need not
    share an epoch with the monotonic ``now`` used for freshness/deadlines.
    A correction request is not permission to retry ambiguous execution.
    """

    def __init__(self, context, policy, *, now):
        self.context = dict(context)
        self.policy = NAVIGATION_POLICIES[policy]
        self.last_sample_time = None
        self.last_fresh = now
        self.settling = None
        self.progress = None
        self.corrections = 0
        self.outside_since = None

    def update(self, pose, *, sample_time, now, stopped):
        policy = self.policy
        if now - self.last_fresh > policy.stale_seconds:
            return "failed", {"reason": "stale navigation pose"}
        if sample_time is None or not math.isfinite(sample_time):
            return None
        if self.last_sample_time is not None and sample_time <= self.last_sample_time:
            return None
        try:
            status, result = measured_arrival(
                self.context, pose, xy_tolerance=policy.xy_tolerance, yaw_tolerance=policy.yaw_tolerance
            )
        except (ValueError, TypeError, OverflowError):
            return None
        if "xy_error" not in result:
            return None
        self.last_sample_time = sample_time
        self.last_fresh = now
        result["corrections"] = self.corrections
        xy, yaw = result["xy_error"], result["yaw_error"]
        inside = status == "succeeded"
        if inside and stopped:
            self.outside_since = None
            if self.settling is None:
                self.settling = (now, sample_time, result["measured_pose"], 1)
                return None
            start, stamp, first, count = self.settling
            self.settling = (start, stamp, first, count + 1)
            elapsed = sample_time - stamp
            if (
                now - start >= policy.settle_seconds
                and elapsed >= policy.settle_seconds
                and count + 1 >= policy.settle_samples
            ):
                displacement = math.hypot(pose[0] - first[0], pose[1] - first[1])
                rotation = abs(math.atan2(math.sin(pose[2] - first[2]), math.cos(pose[2] - first[2])))
                if displacement / elapsed <= 0.01 and rotation / elapsed <= 0.03:
                    return "succeeded", {**result, "settle_seconds": now - start}
                self.settling = None
            return None
        self.settling = None
        if stopped and not inside:
            if self.outside_since is None:
                self.outside_since = (now, sample_time, 1)
                return None
            start, stamp, count = self.outside_since
            self.outside_since = (start, stamp, count + 1)
            if (
                now - start < policy.settle_seconds
                or sample_time - stamp < policy.settle_seconds
                or count + 1 < policy.settle_samples
            ):
                return None
            self.outside_since = None
            if self.corrections >= policy.max_corrections:
                return "failed", {**result, "reason": "navigation correction budget exhausted"}
            self.corrections += 1
            self.progress = None
            return "correct", {**result, "corrections": self.corrections}
        self.outside_since = None
        if self.progress is None:
            self.progress = (now, xy, yaw)
        elif now - self.progress[0] >= policy.progress_seconds:
            _, old_xy, old_yaw = self.progress
            improved = (old_xy > policy.xy_tolerance and old_xy - xy >= 0.01) or (
                old_yaw > policy.yaw_tolerance and old_yaw - yaw >= 0.02
            )
            if not improved and not inside:
                return "failed", {**result, "reason": "navigation stalled"}
            self.progress = (now, xy, yaw)
        return None


def measured_arrival(context, pose, *, xy_tolerance, yaw_tolerance):
    goal = context["resolved_goal"]
    pose = [float(v) for v in pose]
    if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
        return "failed", {"reason": "invalid measured base pose"}
    xy_error = math.hypot(pose[0] - goal[0], pose[1] - goal[1])
    yaw_error = abs(math.atan2(math.sin(pose[2] - goal[2]), math.cos(pose[2] - goal[2])))
    result = {
        **context,
        "measured_pose": pose,
        "xy_error": xy_error,
        "yaw_error": yaw_error,
        "xy_tolerance": xy_tolerance,
        "yaw_tolerance": yaw_tolerance,
    }
    return ("succeeded" if xy_error <= xy_tolerance and yaw_error <= yaw_tolerance else "failed"), result
