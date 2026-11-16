"""Continuous nonholonomic references for unloaded, base-only endpoint tests.

These are rest-to-rest *reference* paths, not moving-start replans. An executor
must verify the initial motion boundary and admit geometry before execution.
No physics, controller targets, history, grip authority or faults are changed.
"""

import numpy as np

from emet.rl.manipulation.jerk_reference import compile_joint_path
from emet.rl.manipulation.tracking import TaskFrameFK, TaskPoseTrajectory


def _wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


class BaseGoalReference(TaskPoseTrajectory):
    """Turn–drive–turn about the actual drive axle, with fixed arm coordinates.

    Forward/reverse selection minimizes the initial turn; the final turn is
    angle-wrapped. Each segment has a C2 jerk-limited S-curve/trapezoidal velocity
    profile bounded by the robot's native speed/acceleration limits. Jerk uses
    the explicit experimental limit 4 * acceleration, not a hardware rating.
    Optional phase bounds reserve derivative capacity for a progress governor.

    Only the continuous sampler has these derivative bounds. The record has a
    distinct format: its display samples must not be silently decoded as the
    linearly interpolated task_pose_trajectory_v1 wire protocol. This compiler
    does not promise continuous moving-goal retargeting or dynamic feasibility.
    """

    def __init__(self, control, target, *, phase_acceleration=0.0, phase_jerk=0.0):
        p = control.profile
        target = np.asarray(target, dtype=float)
        start = np.asarray(control.base_pose(), dtype=float).copy()
        self.offset = np.asarray(control.axle_offset, dtype=float)[:2].copy()
        if (
            not p.wheel_bodies
            or target.shape != (3,)
            or start.shape != (3,)
            or self.offset.shape != (2,)
            or not all(np.isfinite(v).all() for v in (target, start, self.offset))
        ):
            raise ValueError("finite base x/y/yaw and a wheeled profile with an axle offset required")
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.joints = control.positions().copy()
        self.fixed_hand = None
        self._grip = float(control.grip_reference)
        axle0 = start[:2] + _rotation(start[2]) @ self.offset
        axle1 = target[:2] + _rotation(target[2]) @ self.offset
        delta = axle1 - axle0
        distance = float(np.linalg.norm(delta))
        bearing = np.arctan2(delta[1], delta[0]) if distance > 1e-10 else start[2]
        first = _wrap(bearing - start[2])
        reverse = _wrap(first + np.pi)
        if abs(reverse) < abs(first):
            first = reverse
        heading = start[2] + first
        direction = 1 if np.array([np.cos(heading), np.sin(heading)]) @ delta >= 0 else -1
        last = _wrap(target[2] - heading)
        self.parts, certificates, elapsed = [], [], 0.0
        for kind, magnitude, axle, yaw in (
            ("turn", first, axle0, start[2]),
            ("drive", direction * distance, axle0, heading),
            ("turn", last, axle1, heading),
        ):
            if abs(magnitude) < 1e-10:
                continue
            speed = p.base_speed if kind == "drive" else p.base_yaw_speed
            acceleration = p.base_acceleration if kind == "drive" else p.base_yaw_acceleration
            path, certificate = compile_joint_path(
                [[0.0], [magnitude]], [speed], [acceleration], [4 * acceleration],
                profile="s_curve", phase_acceleration=phase_acceleration, phase_jerk=phase_jerk,
            )
            self.parts.append((elapsed, kind, path, axle.copy(), float(yaw)))
            certificates.append(certificate)
            elapsed += path.end
        if not self.parts:
            path, certificate = compile_joint_path(
                [[0.0], [0.0]], [p.base_speed], [p.base_acceleration], [4 * p.base_acceleration],
                profile="s_curve", phase_acceleration=phase_acceleration, phase_jerk=phase_jerk,
            )
            self.parts = [(0.0, "drive", path, axle0, float(start[2]))]
            certificates = [certificate]
            elapsed = path.end
        self.duration = elapsed
        self.certificate = {
            "segments": certificates,
            "velocity_frame": "drive_axle_forward_and_world_yaw",
            "rest_to_rest_reference_only": True,
            "initial_motion_boundary_verified": False,
            "dynamic_feasibility_verified": False,
        }
        times = np.r_[np.arange(0, elapsed, p.dt), elapsed]
        samples = [self.coordinates(t) for t in times]
        super().__init__(
            times, [b for b, _ in samples], [self.fk(b, q) for b, q in samples],
            np.full(len(times), self._grip), base_mask=[True] * 3, hand_mask=False,
        )

    def coordinates(self, time):
        if not np.isfinite(time):
            raise ValueError("finite reference time required")
        time = float(np.clip(time, 0, self.duration))
        start, kind, path, axle, yaw = self.parts[0]
        for part in self.parts[1:]:
            if time >= part[0]:
                start, kind, path, axle, yaw = part
        position = float(path.sample(time - start)[0][0])
        axle = axle.copy()
        if kind == "turn":
            yaw += position
        else:
            axle += position * np.array([np.cos(yaw), np.sin(yaw)])
        return np.r_[axle - _rotation(yaw) @ self.offset, _wrap(yaw)], self.joints.copy()

    def sample(self, time):
        base, joints = self.coordinates(time)
        return base, self.fk(base, joints), self._grip

    def record(self):
        return {
            **super().record(),
            "format": "emet_base_goal_reference_v1",
            "segments": [
                {"start": t, "kind": k, "axle": a.tolist(), "yaw": y, "path": p.record()}
                for t, k, p, a, y in self.parts
            ],
            "certificate": self.certificate,
        }
