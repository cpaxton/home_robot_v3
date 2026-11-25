"""Plan an in-range hold from a measured Innate Mars parked pose."""

import math

import numpy as np

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
ENCODER_STEP_RAD = math.tau / 4096


def shoulder_lower_limit(yaw_rad, lower_rad):
    """Innate 0.8's yaw-dependent shoulder clearance in published joint coordinates."""
    return float(np.interp(yaw_rad, (-1.35, -1.0, 1.0, 1.25), (lower_rad, -0.5, -0.5, lower_rad)))


def project_parked_hold(measured, published_limits, *, settling_reserve_rad=0.0):
    """Compute a hold candidate without changing feedback or moving the gripper.

    Keep a legal measured hold as-is. When starting outside the range, pass
    excess shoulder/elbow pitch down the chain as Innate does. Leave two encoder
    counts of room on a clamped joint for motor quantization and pose readback,
    plus the caller's reserve for measured motion while the hold settles.
    The caller must review the physical transition from the measured start.
    """
    try:
        start = np.asarray([measured[name] for name in JOINT_NAMES], dtype=float)
        limits = np.asarray([published_limits[name] for name in JOINT_NAMES], dtype=float)
    except (KeyError, TypeError) as exc:
        raise ValueError("complete measured arm and published limits required") from exc
    if (
        start.shape != (6,)
        or limits.shape != (6, 2)
        or not np.isfinite(start).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("finite arm and ordered published joint limits required")
    if np.any(start[4:] < limits[4:, 0]) or np.any(start[4:] > limits[4:, 1]):
        raise ValueError("wrist roll or gripper needs a separate recovery")

    if not math.isfinite(settling_reserve_rad) or settling_reserve_rad < 0:
        raise ValueError("settling reserve must be finite and nonnegative")
    margin = 2 * ENCODER_STEP_RAD + settling_reserve_rad

    def inside(value, lower, upper):
        if lower > upper:
            raise ValueError("shoulder clearance exceeds the published joint range")
        if lower <= value <= upper:
            return value
        if upper - lower <= 2 * margin:
            raise ValueError("projected hold has no room for the requested margin inside the published range")
        return float(np.clip(value, lower + margin, upper - margin))

    hold = start.copy()
    hold[0] = inside(start[0], *limits[0])
    excess = 0.0
    for index in range(1, 4):
        lower, upper = limits[index]
        if index == 1:
            lower = max(lower, shoulder_lower_limit(hold[0], lower))
        desired = start[index] + excess
        hold[index] = inside(desired, lower, upper)
        excess = desired - hold[index]
    return dict(zip(JOINT_NAMES, hold.tolist(), strict=True))


def plan_safe_start(measured, published_limits, *, settling_reserve_rad=math.radians(5)):
    """Return position waypoints for the supported folded-pose lift and hold."""
    hold = project_parked_hold(measured, published_limits)
    shoulder, elbow = hold["joint2"], hold["joint3"]
    lower, upper = published_limits["joint3"]
    if abs(shoulder + 1.2195) > math.radians(12) or abs(elbow - upper) > math.radians(12):
        raise ValueError("safe start requires a folded shoulder and elbow near the parked pose")
    if max(abs(hold[name] - measured[name]) for name in JOINT_NAMES) > math.radians(6):
        raise ValueError("parked correction exceeds the safe-start stroke envelope")

    elbow_floor = max(lower, 0.9)
    elbow_ceiling = upper - settling_reserve_rad - 2 * ENCODER_STEP_RAD
    if not math.isfinite(settling_reserve_rad) or settling_reserve_rad < 0 or elbow_ceiling < elbow_floor:
        raise ValueError("requested settling reserve exceeds the stock first lift")

    stages = []
    for fraction, stage in ((0.1, "first_lift"), (0.2, "settling_hold")):
        target = dict(hold)
        target["joint2"] = shoulder + fraction * (-0.9 - shoulder)
        target["joint3"] = elbow + fraction * (0.9 - elbow)
        if stage == "settling_hold":
            target["joint3"] = min(target["joint3"], elbow_ceiling)
        if project_parked_hold(target, published_limits) != target:
            raise ValueError("safe-start waypoint exceeds the published limits")
        stages.append({"stage": stage, "duration_s": 2.5, "target_rad": target})
    return {
        "requires_confirmed_powered_hold": True,
        "requires_auto_rest_disabled": True,
        "projected_start_rad": hold,
        "waypoints": stages,
    }


def run_safe_start(driver, published_limits, *, settling_reserve_rad=math.radians(5)):
    """Execute safe start with an exclusive driver that verifies its powered hold.

    The driver owns the watchdog and confirms fresh, stationary measured
    feedback from each move. Its hold preparation must preserve motor output
    during any profile or gain change and fail if automatic folding is active.
    """
    start = driver.prepare_safe_start_hold()
    plan = plan_safe_start(start, published_limits, settling_reserve_rad=settling_reserve_rad)
    try:
        for waypoint in plan["waypoints"]:
            target = waypoint["target_rad"]
            measured = driver.move_and_confirm_hold(target, duration_s=waypoint["duration_s"])
            for name in JOINT_NAMES:
                position = measured[name]
                lower, upper = published_limits[name]
                if not lower <= position <= upper or abs(position - target[name]) > 2 * ENCODER_STEP_RAD:
                    raise RuntimeError(f"safe-start hold did not track {name}")
    except Exception as exc:
        if not driver.stop_and_confirm():
            raise RuntimeError("safe-start stop unconfirmed") from exc
        raise
    return measured


def recover_supported_elbow(driver, published_limits):
    """Recover a latched servo 3 fault with an explicitly supported forearm.

    Driver methods must use exclusive ownership and fresh feedback. The
    targeted reset restores a powered hold, with idle folding disabled.
    """
    state = driver.read_recovery_state()
    lower, upper = published_limits["joint3"]
    reserve = math.radians(5)

    def supported_pose(sample):
        angle = sample["elbow_rad"]
        return (
            sample["forearm_supported"] is True
            and sample["auto_rest"] is False
            and math.isfinite(angle)
            and lower + reserve <= angle <= upper - reserve
        )

    if not supported_pose(state) or set(state["faulted_servos"]) != {3}:
        raise ValueError("elbow reset requires a supported in-range pose and an isolated servo 3 fault")
    try:
        driver.reset_servo_to_measured_hold(3)
        held = driver.confirm_recovery_hold()
        if (
            not supported_pose(held)
            or held["faulted_servos"]
            or held["torque_enabled"] is not True
            or abs(held["elbow_rad"] - state["elbow_rad"]) > 2 * ENCODER_STEP_RAD
        ):
            raise RuntimeError("supported elbow hold not confirmed")
        return held
    except Exception as exc:
        if not driver.stop_and_confirm():
            raise RuntimeError("elbow recovery stop unconfirmed") from exc
        raise
