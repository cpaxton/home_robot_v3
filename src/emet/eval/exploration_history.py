# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Task-independent observation collection through the qualified motion adapter.

This is an evaluation collector, not an alternative task agent. It takes no
question, semantic detections, scorer labels, or simulator geometry.
"""

import time

import numpy as np

from emet.eval.grapheqa_navigation import HabitatMotionExecutor


class CollectionBudgetExceeded(RuntimeError):
    """A declared acquisition budget expired before another action started."""


class RecordedSensor:
    """Record every fresh capture, counting all primitives including inspections."""

    def __init__(self, sensor, record, *, max_primitives, max_seconds, clock=time.monotonic):
        if (
            isinstance(max_primitives, bool)
            or not isinstance(max_primitives, int)
            or max_primitives <= 0
            or not np.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ValueError("Positive finite collection budgets required")
        self.sensor, self.record = sensor, record
        self.max_primitives, self.max_seconds = max_primitives, max_seconds
        self.clock, self.started = clock, clock()
        self.primitive_count = 0
        record(sensor.last, None)

    @property
    def last(self):
        return self.sensor.last

    def check_budget(self):
        if self.primitive_count >= self.max_primitives or self.clock() - self.started >= self.max_seconds:
            raise CollectionBudgetExceeded("Declared collection budget exhausted")

    def step(self, action, amount=None):
        self.check_budget()
        # A failed/exceptional attempted primitive still consumes its allowance.
        self.primitive_count += 1
        frame = self.sensor.step(action, amount)
        self.record(frame, {"action": action, "amount": amount})
        return frame

    def capture(self):
        self.check_budget()
        frame = self.sensor.capture()
        self.record(frame, None)
        return frame


def route_prefix(route, start, max_distance_m):
    """Keep an admitted prefix bounded by cumulative translation, including turns."""
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("A positive finite leg distance is required")
    if not route:
        return [], 0.0
    prefix, distance, previous = [], 0.0, np.asarray(start, dtype=float)
    for waypoint in route:
        current = np.asarray(waypoint, dtype=float)
        increment = float(np.linalg.norm(current[:2] - previous[:2]))
        if distance + increment > max_distance_m + 1e-9:
            fraction = (max_distance_m - distance) / increment
            if fraction > 0:
                clipped = previous + fraction * (current - previous)
                clipped[2] = previous[2] + fraction * np.arctan2(
                    np.sin(current[2] - previous[2]), np.cos(current[2] - previous[2])
                )
                prefix.append(clipped.tolist())
                distance = max_distance_m
            break
        prefix.append(waypoint)
        distance += increment
        previous = current
    return prefix, distance


def collect_history(sensor, navigation, event, *, max_legs=2, leg_distance_m=0.75):
    """Scan, calibrate stationary floor, then visit shortest observed frontiers.

    The fixed initial panorama is 360 degrees (72 calibrated 5-degree turns).
    Every arrival adds a fixed 90-degree scan. Candidate routes, failures and
    measured receipts are retained. No failed motion is silently retried.
    Budgets are enforced by RecordedSensor before each primitive/capture; a
    route can finish as a retained partial attempt when the budget expires.
    """
    if isinstance(max_legs, bool) or not isinstance(max_legs, int) or max_legs < 0:
        raise ValueError("Leg count must be a nonnegative integer")
    route_prefix([], navigation.pose, leg_distance_m)  # Validate before motion.
    navigation.update(sensor.last)

    def inspect(actions):
        for action in actions:
            navigation.update(sensor.step(action))

    inspect(["turn_left"] * 9 + ["look_down"] * 3)
    event({"stage": "floor_calibration", **navigation.calibrate_stationary_floor(sensor.last)})
    inspect(["look_up"] * 3 + ["turn_left"] * 63)
    event({"stage": "initial_scan_complete", "sequence": sensor.last["sequence"]})
    executor = HabitatMotionExecutor(sensor, navigation)
    attempted = []
    for leg in range(max_legs):
        sensor.check_budget()
        proposals = []
        for index, point in enumerate(navigation.frontiers()):
            if any(np.linalg.norm(np.asarray(point[:2]) - old[:2]) < 0.35 for old in attempted):
                continue
            route, reason = navigation.plan(point, object_target=False)
            prefix, distance = route_prefix(route, navigation.pose, leg_distance_m)
            event({"stage": "proposal", "leg": leg, "target": point, "reason": reason, "prefix_distance_m": distance})
            if prefix and distance > 0:
                route_distance = float(
                    np.linalg.norm(np.diff(np.array([navigation.pose, *route])[:, :2], axis=0), axis=1).sum()
                )
                proposals.append((route_distance, index, point, prefix))
        if not proposals:
            return {"status": "no_admitted_frontier", "completed_legs": leg}
        _, _, target, route = min(proposals, key=lambda item: item[:2])
        attempted.append(np.asarray(target))
        event({"stage": "motion_started", "leg": leg, "target": target, "route": route})
        try:
            receipt = executor.execute(route)
        except CollectionBudgetExceeded as exc:
            event(
                {
                    "stage": "motion_interrupted",
                    "leg": leg,
                    "reason": str(exc),
                    "current_pose": navigation.pose.tolist(),
                }
            )
            raise
        event({"stage": "motion_finished", "leg": leg, "receipt": receipt})
        if not receipt["success"]:
            return {"status": "motion_failed", "completed_legs": leg, "reason": receipt["reason"]}
        navigation.update(sensor.capture())
        inspect(["turn_left"] * 18)
        event({"stage": "arrival_scan_complete", "leg": leg, "sequence": sensor.last["sequence"]})
    return {"status": "completed", "completed_legs": max_legs}
