"""Experimental joint/wheel witnesses for reachable whole-body pose references.

Wheel travel generates differential-drive motion at the axle, not independent
world X/Y splines. The actor still receives only masked SE(2)/SE(3) references.
Kinematic admission and teacher execution remain separate from these witnesses.
"""

from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from emet.rl.manipulation.arm_splines import (
    ANCHORS,
    ArmSplineTrackingEnv,
    SplineLevel,
    spline_joint_extrema,
)
from emet.rl.manipulation.coupled_tracking import spline_bounds
from emet.rl.manipulation.reference_timing import retime_derivative_bounds
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory
from emet.rl.manipulation.trajectory_metrics import SpreadRequirements, trajectory_spread

NAMESPACES = {"train": 570_000_000, "audit": 580_000_000}
LEVELS = tuple(
    SplineLevel(amplitude, speed, jitter, initial_hold_s=0.2, maximum_duration_s=30)
    for amplitude, speed, jitter in ((0.30, 0.25, 0.04), (0.45, 0.55, 0.10), (0.55, 0.85, 0.20))
)


def manifest(train_count=12, audit_count=8):
    if any(type(n) is not int or not 1 <= n <= 1000 for n in (train_count, audit_count)):
        raise ValueError("bounded positive whole-body populations required")
    postures = [*ANCHORS, "joint_space"]
    return {
        "format": "mars-whole-body-joint-splines-v1",
        "robot": "innate_mars",
        "status": "experimental; not in the running PPO mixture",
        "levels": [asdict(level) for level in LEVELS],
        "spread_requirements": SpreadRequirements().record(),
        "cases": [
            {
                "id": f"mars:whole-body-joint-splines-v1:{split}:l{level}:{i}",
                "seed": namespace + level * 10000 + i,
                "split": split,
                "level": level,
                "posture": postures[i % len(postures)],
            }
            for split, namespace, count in (
                ("train", NAMESPACES["train"], train_count),
                ("audit", NAMESPACES["audit"], audit_count),
            )
            for level in range(len(LEVELS))
            for i in range(count)
        ],
    }


class JointWheelPath:
    """Smooth joint/cumulative-wheel geometry with bounded common timing.

    Scalar forward speed and yaw rate are bounded in axle coordinates. These
    are kinematic bounds, not bounds on contact forces or world-frame jerk.
    """

    def __init__(self, arm, wheels, base, axle_offset, separation, speed, acceleration, jerk, *, initial_hold_s=0.2):
        arm, wheels, base, offset = (np.asarray(x, float) for x in (arm, wheels, base, axle_offset))
        if (
            arm.ndim != 2
            or not 3 <= len(arm) <= 6
            or arm.shape[1] < 1
            or wheels.shape != (len(arm), 2)
            or base.shape != (3,)
            or offset.shape != (2,)
            or any(not np.isfinite(x).all() for x in (arm, wheels, base, offset))
            or not np.array_equal(wheels[0], [0, 0])
            or isinstance(separation, (bool, np.bool_))
            or not np.isfinite(separation)
            or separation <= 0
            or isinstance(initial_hold_s, (bool, np.bool_))
            or not np.isfinite(initial_hold_s)
            or not 0 <= initial_hold_s <= 2
        ):
            raise ValueError("finite joint/wheel geometry and axle dimensions required")
        self.start_base, self.offset = base.copy(), offset.copy()
        self.initial_hold_s = float(initial_hold_s)
        self.arm_waypoints, self.wheel_waypoints = arm.copy(), wheels.copy()
        self.separation = float(separation)
        travel = wheels.mean(axis=1)
        yaw = (wheels[:, 1] - wheels[:, 0]) / separation
        self.geometry = CubicSpline(np.linspace(0, 1, len(arm)), np.column_stack([travel, yaw, arm]), bc_type="natural")
        self.phase, self.bounds = retime_derivative_bounds(
            *spline_bounds(self.geometry), speed, acceleration, jerk, mode="bounded_v2"
        )
        self.end = self.initial_hold_s + self.phase.end
        self.speed, self.acceleration, self.jerk = (np.asarray(x, float).copy() for x in (speed, acceleration, jerk))
        axle = base[:2] + self.rotation(base[2]) @ offset

        def derivative(phase, position):
            heading = base[2] + float(self.geometry(phase)[1])
            return self.geometry(phase, 1)[0] * np.array([np.cos(heading), np.sin(heading)])

        integral = solve_ivp(
            derivative, (0, 1), axle, method="DOP853", rtol=1e-10, atol=1e-12, max_step=0.01, dense_output=True
        )
        if not integral.success:
            raise ValueError("differential-drive path integration failed")
        self.axle_xy = integral.sol

    @staticmethod
    def rotation(yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def base_at(self, phase):
        yaw = self.start_base[2] + float(self.geometry(phase)[1])
        xy = self.axle_xy(phase) - self.rotation(yaw) @ self.offset
        return np.r_[xy, np.arctan2(np.sin(yaw), np.cos(yaw))]

    def coordinates(self, time):
        phase = float(self.phase.sample(float(np.clip(time - self.initial_hold_s, 0, self.phase.end)))[0][0])
        return self.base_at(phase), self.geometry(phase)[2:].copy()

    def derivatives(self, time):
        """Forward/yaw/joint derivatives; the hold has exactly zero motion."""
        if time < self.initial_hold_s or time > self.end:
            return np.zeros_like(self.bounds)
        s, v, a, j = (float(x[0]) for x in self.phase.sample(time - self.initial_hold_s))
        first, second, third = (self.geometry(s, order) for order in (1, 2, 3))
        return np.array([first * v, second * v**2 + first * a, third * v**3 + 3 * second * v * a + first * j])


class WholeBodySplineReference(TaskPoseTrajectory):
    def __init__(self, control, arm, wheels, level):
        if control.profile.robot != "innate_mars":
            raise ValueError("Mars whole-body joint/wheel reference required")
        p = control.profile
        if not np.allclose(np.asarray(arm)[0], control.positions(), atol=1e-10, rtol=0):
            raise ValueError("joint path must start at measured reset posture")
        speed = np.r_[p.base_speed, p.base_yaw_speed, p.speed] * level.speed_fraction
        acceleration = np.r_[p.base_acceleration, p.base_yaw_acceleration, p.acceleration] * 0.5
        self.path = JointWheelPath(
            arm,
            wheels,
            control.base_pose(),
            control.axle_offset[:2],
            p.wheel_separation,
            speed,
            acceleration,
            4 * acceleration,
            initial_hold_s=level.initial_hold_s,
        )
        if self.path.end > level.maximum_duration_s:
            raise ValueError("whole-body path exceeds duration limit")
        low, high = spline_joint_extrema(self.path.geometry)
        if np.any(low[2:] < control.limits[:, 0] + 0.01) or np.any(high[2:] > control.limits[:, 1] - 0.01):
            raise ValueError("between-knot whole-body joint margin")
        self.initial_hold_s, self.fixed_hand = level.initial_hold_s, None
        self.fk = TaskFrameFK(control.model, control.data, control)
        times = np.r_[np.arange(0, self.path.end, p.dt), self.path.end]
        coordinates = [self.coordinates(t) for t in times]
        bases = np.array([base for base, _ in coordinates])
        hands = []
        for base, q in coordinates:
            hand = self.fk(base, q)
            if np.linalg.norm(hand[:3] - self.fk.data.xpos[control.base]) > 0.7:
                raise ValueError("hand path outside nearby workspace")
            hands.append(hand)
        super().__init__(times, bases, hands, np.full(len(times), control.grip_reference), base_mask=[True] * 3)
        self.spread = trajectory_spread(times, self.hand, bases, joints=np.array([q for _, q in coordinates]))
        self.certificate = {
            "format": "joint-wheel-fk-witness-v1",
            "joint_waypoints": np.asarray(arm).tolist(),
            "wheel_travel_waypoints_m": np.asarray(wheels).tolist(),
            "wheel_separation_m": p.wheel_separation,
            "axle_offset_m": control.axle_offset[:2].tolist(),
            "start_base_pose": self.path.start_base.tolist(),
            "joint_limits_rad": control.limits.tolist(),
            "initial_hold_s": level.initial_hold_s,
            "duration_s": self.path.end,
            "level": asdict(level),
            "speed": speed.tolist(),
            "acceleration": acceleration.tolist(),
            "jerk": (4 * acceleration).tolist(),
            "derivative_bounds": self.path.bounds.tolist(),
            "base_pose_stationary": False,
            "base_physically_welded": False,
            "axle_integration": {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12, "max_phase_step": 0.01},
            "dynamic_feasibility_verified": False,
        }

    def coordinates(self, time):
        return self.path.coordinates(time)

    def sample(self, time):
        base, joints = self.coordinates(float(np.clip(time, 0, self.end)))
        return base, self.fk(base, joints), float(self.gripper[0])

    def record(self):
        return {**super().record(), "certificate": self.certificate, "format": "whole-body-joint-wheel-fk-v1"}


class WholeBodySplineTrackingEnv(ArmSplineTrackingEnv):
    """Separate experimental family; does not change any existing mixture."""

    def __init__(self, cases, *, levels=LEVELS, maximum_attempts=64):
        super().__init__(
            cases,
            levels=levels,
            maximum_attempts=maximum_attempts,
            timing="bounded_v2",
            finish_after_hold=True,
            spread_requirements=SpreadRequirements().record(),
        )

    def validate_case(self, case):
        if not isinstance(case, dict) or set(case) != {"id", "seed", "split", "level", "posture"}:
            raise ValueError("complete seeded whole-body case required")
        split, level = case["split"], case["level"]
        if (
            not isinstance(split, str)
            or split not in NAMESPACES
            or type(level) is not int
            or not 0 <= level < len(self.levels)
            or type(case["seed"]) is not int
            or not NAMESPACES[split] + level * 10000 <= case["seed"] < NAMESPACES[split] + (level + 1) * 10000
            or not isinstance(case["id"], str)
            or not case["id"].startswith(f"mars:whole-body-joint-splines-v1:{split}:l{level}:")
            or not isinstance(case["posture"], str)
            or case["posture"] not in {*ANCHORS, "joint_space"}
        ):
            raise ValueError("disjoint whole-body split/level identity and posture required")

    def sample_posture(self, rng, case, level):
        if case["posture"] != "joint_space":
            return super().sample_posture(rng, case, level)
        limits = self.unwrapped.control.limits
        return rng.uniform(limits[:, 0] + 0.03, limits[:, 1] - 0.03)

    def sample_reference(self, rng, case, level):
        c, p = self.unwrapped.control, self.unwrapped.profile
        initial = c.positions()
        low = np.maximum(initial - level.amplitude_rad, c.limits[:, 0] + 0.03)
        high = np.minimum(initial + level.amplitude_rad, c.limits[:, 1] - 0.03)
        arm = np.vstack([initial, rng.uniform(low, high, (level.knot_count - 1, p.arm_size))])
        increments = rng.uniform([-0.10, -0.30], [0.10, 0.30], (level.knot_count - 1, 2))
        travel, yaw = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)]).T
        wheels = np.column_stack([travel - 0.5 * p.wheel_separation * yaw, travel + 0.5 * p.wheel_separation * yaw])
        if rng.random() < 0.5:
            arm[-1], wheels[-1] = initial, 0
        return WholeBodySplineReference(c, arm, wheels, level)

    def settling_budget(self):
        return 6.0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.unwrapped.control.base_xy_tolerance = self.unwrapped.active_base_xy_tolerance
        return observation, {**info, "whole_body_joint_spline": True}
