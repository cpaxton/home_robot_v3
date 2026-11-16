"""Continuous nonholonomic references for unloaded, base-only endpoint tests.

Initial paths are rest-to-rest; replans match the existing announced motion
and brake before changing direction. An executor must separately verify the
initial measured motion boundary and admit geometry before execution.
No physics, controller targets, history, grip authority or faults are changed.
"""

import numpy as np

from emet.rl.manipulation.base_braking import braking_path
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
    matches a previous BaseGoalReference's pose/velocity/acceleration on replan.
    Retimed/externally generated boundaries require a separate compiler.
    """

    def __init__(self, control, target, *, phase_acceleration=0.0, phase_jerk=0.0, previous=None, elapsed=None):
        p = control.profile
        self._owner = control
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
        self.target = target.copy()
        self.target[2] = _wrap(self.target[2])
        self.fk = TaskFrameFK(control.model, control.data, control)
        self.joints = control.positions().copy()
        self.fixed_hand = None
        self._grip = float(control.grip_reference)
        self.parts, certificates, elapsed_path = [], [], 0.0
        self.boundary = None
        self._past_parts, self._past_pose = [], start.copy()
        if previous is not None:
            if (
                not isinstance(previous, BaseGoalReference) or previous._owner is not control
                or elapsed is None or not np.isfinite(elapsed) or elapsed < 0
                or phase_acceleration != 0 or phase_jerk != 0
                or previous.phase_bounds != (0.0, 0.0)
                or not np.array_equal(self.offset, previous.offset)
            ):
                raise ValueError("replan requires the same unretimed base reference and a finite elapsed time")
            start, self.joints, velocity, acceleration, _ = previous.state(elapsed)
            # The observation includes a -50 ms reference sample and takes
            # central differences for announced twist. Clamping the new path
            # at zero would halve feedforward at every replan. Retain only the
            # needed primitive tail, without a chain of whole prior references.
            self._past_pose = previous.state(elapsed - 0.05)[0]
            past = [part for part in (*previous._past_parts, *previous.parts) if part[0] < elapsed]
            if past:
                first = max((i for i, part in enumerate(past) if part[0] <= elapsed - 0.05), default=0)
                self._past_parts = [(t - elapsed, k, path, axle, yaw) for t, k, path, axle, yaw in past[first:]]
            self.boundary = {
                "source_elapsed_s": float(elapsed), "pose": start.tolist(),
                "axle_velocity": velocity.tolist(), "axle_acceleration": acceleration.tolist(),
                "announced_motion_boundary_matched": True, "measured_motion_boundary_verified": False,
            }
            active = np.flatnonzero((abs(velocity) > 1e-12) | (abs(acceleration) > 1e-12))
            if len(active) > 1:
                raise ValueError("base reference braking requires one active drive/turn axis")
            if len(active):
                axis = int(active[0])
                speed = p.base_speed if axis == 0 else p.base_yaw_speed
                accel = p.base_acceleration if axis == 0 else p.base_yaw_acceleration
                path, certificate = braking_path(velocity[axis], acceleration[axis], speed, accel, 4 * accel)
                axle = start[:2] + _rotation(start[2]) @ self.offset
                kind = "drive" if axis == 0 else "turn"
                self.parts.append((0.0, kind, path, axle, float(start[2])))
                certificates.append(certificate)
                elapsed_path = path.end
                self.duration = elapsed_path
                start, _ = self.coordinates(elapsed_path)
            self.boundary["braking_duration_s"] = elapsed_path
        elif elapsed is not None:
            raise ValueError("elapsed time requires a previous reference")
        self.phase_bounds = (float(phase_acceleration), float(phase_jerk))
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
        elapsed = elapsed_path
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
        if self.boundary is not None:
            self.certificate.update(rest_to_rest_reference_only=False, announced_motion_boundary_matched=True)
        times = np.r_[np.arange(0, elapsed, p.dt), elapsed]
        samples = [self.coordinates(t) for t in times]
        super().__init__(
            times, [b for b, _ in samples], [self.fk(b, q) for b, q in samples],
            np.full(len(times), self._grip), base_mask=[True] * 3, hand_mask=False,
        )

    def coordinates(self, time):
        base, joints, _, _, _ = self.state(time)
        return base, joints

    def state(self, time):
        """Pose/joints and axle-frame velocity, acceleration, jerk (forward/yaw)."""
        if not np.isfinite(time):
            raise ValueError("finite reference time required")
        if time < 0:
            time = max(float(time), -0.05)
            if not self._past_parts or time < self._past_parts[0][0]:
                return self._past_pose.copy(), self.joints.copy(), np.zeros(2), np.zeros(2), np.zeros(2)
            parts = self._past_parts
        else:
            time, parts = float(min(time, self.duration)), self.parts
        start, kind, path, axle, yaw = parts[0]
        for part in parts[1:]:
            if time >= part[0]:
                start, kind, path, axle, yaw = part
        position, velocity, acceleration, jerk = (float(x[0]) for x in path.sample(time - start))
        axle = axle.copy()
        if kind == "turn":
            yaw += position
        else:
            axle += position * np.array([np.cos(yaw), np.sin(yaw)])
        axis = np.array([0.0, 1.0]) if kind == "turn" else np.array([1.0, 0.0])
        return (
            np.r_[axle - _rotation(yaw) @ self.offset, _wrap(yaw)], self.joints.copy(),
            axis * velocity, axis * acceleration, axis * jerk,
        )

    def sample(self, time):
        base, joints = self.coordinates(time)
        return base, self.fk(base, joints), self._grip

    def record(self):
        return {
            **super().record(),
            "format": "emet_base_goal_reference_v1" if self.boundary is None else "emet_base_goal_reference_v2",
            **({"boundary": self.boundary} if self.boundary is not None else {}),
            **({"past_window_s": 0.05, "past_segments": [
                {"start": t, "kind": k, "axle": a.tolist(), "yaw": y, "path": p.record()}
                for t, k, p, a, y in self._past_parts
            ]} if self.boundary is not None else {}),
            "segments": [
                {"start": t, "kind": k, "axle": a.tolist(), "yaw": y, "path": p.record()}
                for t, k, p, a, y in self.parts
            ],
            "certificate": self.certificate,
        }


def install_base_goal(env, target):
    """Admit an unloaded base-only goal/replan, then install it atomically.

    Moving replans preserve the old announced boundary, not a new measurement
    with reference velocity reset to zero. Other reference types and explicit
    phase retiming need their own moving-boundary compiler. Failures preserve
    the previous schedule, targets, history, grip state and fault latch.
    """
    from emet.rl.manipulation.coupled_tracking import admit_reference

    c = env.control
    if c.stopped or c.motion_fault or getattr(env, "_done", False):
        raise ValueError("base goal requires an active, nonfaulted episode")
    if c.hand_mask or c.retain_object or not c.base_mask.all() or env.scenario != "reach":
        raise ValueError("base reference goals require an unloaded, fully constrained base-only reach")
    previous = c.task_trajectory
    if isinstance(previous, BaseGoalReference):
        if c.task_progress is not None or c.task_progress_rate != 1:
            raise ValueError("moving base replan does not support an externally retimed reference")
        path = BaseGoalReference(c, target, previous=previous, elapsed=c.trajectory_elapsed())
    else:
        # Whole free-base norms include vertical/roll/pitch motion. A quiet
        # forward/yaw projection alone cannot establish a rest boundary.
        velocity = np.asarray(env.data.qvel[c.base_dof : c.base_dof + 6]).copy()
        arm = np.array([np.sum(env.data.qvel[group]) for group in c.v])
        if (
            c.base_dof < 0 or velocity.shape != (6,)
            or not np.isfinite(np.r_[velocity, arm]).all()
            or np.linalg.norm(velocity[:3]) > 0.005 or np.linalg.norm(velocity[3:]) > 0.02
            or np.any(abs(arm) > 0.03)
        ):
            raise ValueError("initial base reference requires measured rest; stop or finish the existing motion first")
        if previous is not None:
            from emet.rl.manipulation.whole_body_observation import reference_velocity

            if np.any(abs(reference_velocity(c)) > 1e-6):
                raise ValueError("cannot replace an unrelated moving reference with a rest-start path")
        path = BaseGoalReference(c, target)
    admission = admit_reference(env, path)
    if not admission["accepted"]:
        raise ValueError(f"base reference rejected: {admission['reason']}")
    env.set_task_trajectory(path)
    return path, admission
