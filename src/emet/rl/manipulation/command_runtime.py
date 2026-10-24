"""Simulator-only task-pose execution through the shared command lifecycle.

Owns an existing manipulation environment, never resets physics or clears a
failed contract. One caller owns advance(); this is not a hardware transport.
Gripper schedules remain preview inputs; only the task executor can release.
"""

import time

import mujoco
import numpy as np

from emet.core.command_client import send_command
from emet.core.command_runtime import CommandRuntime
from emet.rl.manipulation.tracking import TaskPoseTrajectory


def task_pose_teacher_action(control):
    """Bounded resolved-rate teacher; no iterative configuration IK at runtime.

    The baseline joint teacher cannot consume hand-only task schedules. Use a
    damped measured Jacobian here, with measured base motion compensation.
    The learned actor uses its normal observation path and never sees this solve.
    """
    from scipy.spatial.transform import Rotation

    c, p = control, control.profile
    c.update_task_reference()
    jp, jr = np.zeros((3, c.model.nv)), np.zeros((3, c.model.nv))
    mujoco.mj_jacBody(c.model, c.data, jp, jr, c.ee)
    jacobian = np.vstack([jp, jr])
    arm = np.column_stack([jacobian[:, group].mean(axis=1) for group in c.v])
    current = c.ee_pose()
    columns = c.hand_goal[3:].reshape(3, 2)
    goal_rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
    rotation = c.data.xmat[c.ee].reshape(3, 3).copy()
    error = np.r_[c.hand_goal[:3] - current[:3], Rotation.from_matrix(goal_rotation @ rotation.T).as_rotvec()]
    # Remove measured non-arm motion from the requested world-frame twist.
    external_twist = jacobian @ c.data.qvel.copy() - arm @ c.velocities()
    feedforward = np.zeros(6)
    base_velocity = None
    if c.task_trajectory is not None:
        path = c.task_trajectory
        elapsed = c.trajectory_elapsed()
        before, after = max(0.0, elapsed - p.dt / 2), min(path.end, elapsed + p.dt / 2)
        if after > before:
            a, b = path.sample(before)[1], path.sample(after)[1]

            def matrix(pose):
                columns = pose[3:].reshape(3, 2)
                return np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])

            feedforward[:3] = (b[:3] - a[:3]) / (after - before)
            feedforward[3:] = Rotation.from_matrix(matrix(b) @ matrix(a).T).as_rotvec() / (after - before)
            feedforward *= c.task_progress_rate
            base_a, base_b = path.sample(before)[0], path.sample(after)[0]

            def axle(pose):
                c0, s0 = np.cos(pose[2]), np.sin(pose[2])
                return pose[:2] + np.array([[c0, -s0], [s0, c0]]) @ c.axle_offset[:2]

            planar = (axle(base_b) - axle(base_a)) / (after - before)
            yaw_rate = np.arctan2(np.sin(base_b[2] - base_a[2]), np.cos(base_b[2] - base_a[2])) / (after - before)
            nominal_motion = np.linalg.norm(planar) > 1e-6 or abs(yaw_rate) > 1e-6
            planar *= c.task_progress_rate
            yaw_rate *= c.task_progress_rate
            if p.wheel_bodies and nominal_motion:
                # Track a timed nonholonomic path, rather than repeatedly
                # treating each sample as a new point-navigation goal. During
                # a turn a small lateral caster error cannot be corrected by
                # translation and must not redirect the heading servo.
                pose = c.base_pose()
                yaw = pose[2]
                heading = np.array([np.cos(yaw), np.sin(yaw)])
                lateral = np.array([-heading[1], heading[0]])
                current_axle = c.data.xpos[c.base] + c.data.xmat[c.base].reshape(3, 3) @ c.axle_offset
                delta = axle(c.base_goal) - current_axle[:2]
                turn_error = np.arctan2(np.sin(c.base_goal[2] - yaw), np.cos(c.base_goal[2] - yaw))
                forward = planar @ heading
                base_velocity = np.array(
                    [
                        forward + 2 * (delta @ heading),
                        yaw_rate + 2.5 * turn_error + 8 * forward * (delta @ lateral),
                    ]
                )
    twist = feedforward + 4 * error - external_twist
    velocity = arm.T @ np.linalg.solve(arm @ arm.T + 1e-4 * np.eye(6), twist)
    velocity *= float(c.hand_mask)
    base = (
        (c.base_feedback() if base_velocity is None else base_velocity) / [p.base_speed, p.base_yaw_speed]
        if p.wheel_bodies
        else []
    )
    return np.clip(np.r_[base, velocity / p.speed, 0.0], -1, 1)


def parse_trajectory_goal(goal):
    if not isinstance(goal, dict) or set(goal) != {"trajectory"}:
        raise ValueError("task-pose command requires one trajectory")
    record = goal["trajectory"]
    if (
        not isinstance(record, dict)
        or set(record) != {"format", "times", "base", "hand", "gripper", "base_mask", "hand_mask"}
        or record["format"] != "task_pose_trajectory_v1"
    ):
        raise ValueError("unsupported task-pose trajectory record")
    if (
        type(record["hand_mask"]) is not bool
        or not isinstance(record["base_mask"], list)
        or any(type(v) is not bool for v in record["base_mask"])
    ):
        raise ValueError("trajectory masks must be booleans")
    return TaskPoseTrajectory(**{k: v for k, v in record.items() if k != "format"})


def send_task_pose_trajectory(client, trajectory, *, motion_timeout_s=15.0):
    """Send a validated schedule only to an explicitly advertised pose executor."""
    if not isinstance(trajectory, TaskPoseTrajectory):
        raise ValueError("expected a task-pose trajectory")
    if not getattr(client, "_state", {}).get("task_pose_control"):
        raise RuntimeError("peer does not advertise task-pose control")
    return send_command(
        client,
        {
            "reactive_goal": {"trajectory": trajectory.record()},
            "reactive_timeout_s": motion_timeout_s,
        },
    )


class TaskPoseCommandRuntime(CommandRuntime):
    def __init__(self, env, *, policy=None):
        if env.config.action_mode != "direct_velocity" or not env.control.preview_enabled:
            raise ValueError("task-pose runtime requires direct velocity and preview")
        if policy is not None and (
            policy.robot != env.profile.robot
            or policy.manifest["fingerprint"] != env.fingerprint
            or policy.action_mode != "direct_velocity"
            or "preview" not in policy.normalization
        ):
            raise ValueError("task-pose policy/model mismatch")
        self.env, self.policy = env, policy
        self.initialize_commands()
        self._last_step = -1
        self.outcome = None
        self.last_advance = time.monotonic()
        self.last_info = None

    def supports_reactive_control(self):
        return True

    def command_message(self, message):
        if message is None:
            return None
        return super().command_message({**message, "task_pose_control": True})

    def validate_reactive_goal(self, goal):
        path = parse_trajectory_goal(goal)
        limits = self.env.control.grip_limits
        if np.any(path.gripper < limits[0]) or np.any(path.gripper > limits[1]):
            raise ValueError("gripper trajectory exceeds physical limits")

    def start_reactive_command(self, goal):
        if self.env._done or self.env.control.stopped or self.env.initialization_failed:
            raise RuntimeError("executor must prepare an active verified stage first")
        self.env.set_task_trajectory(parse_trajectory_goal(goal))
        self.outcome = None
        self.last_advance = time.monotonic()
        return {
            "frame": "world",
            "motion_mode": "task_pose",
            "controller": "teacher" if self.policy is None else "learned",
        }

    def update_reactive_command(self, goal):
        if self.outcome is not None or self.env._done or self.env.control.stopped:
            raise RuntimeError("cannot update a completed or failed contract")
        self.env.set_task_trajectory(parse_trajectory_goal(goal))
        return {"frame": "world", "motion_mode": "task_pose"}

    def advance(self):
        """Advance one control interval; caller is the sole physics writer."""
        with self._command_lock:
            if self._reactive_command is not None and not self._reactive_fault and self.outcome is None:
                # Expiry is checked before inference or another actuator command.
                self.poll_reactive_command()
                if self._reactive_command is not None and not self._reactive_fault:
                    try:
                        started = time.monotonic()
                        obs = self.env.control.observe()
                        action = task_pose_teacher_action(self.env.control) if self.policy is None else self.policy(obs)
                        if time.monotonic() - started > 2 * self.env.profile.dt:
                            raise RuntimeError("policy inference deadline exceeded")
                        _, _, done, truncated, info = self.env.step(action)
                        self.last_info = info
                        if time.monotonic() - started > 2 * self.env.profile.dt:
                            raise RuntimeError("control step deadline exceeded")
                        if done or truncated:
                            self.outcome = (
                                "succeeded" if info["is_success"] else "failed",
                                {
                                    "code": info["code"],
                                    "measurements": info,
                                },
                            )
                    except Exception as exc:
                        self.outcome = ("failed", {"failure": str(exc)})
                    self.last_advance = time.monotonic()
                    self.poll_reactive_command()
            self._last_step += 1

    def reactive_command_result(self, context):
        if time.monotonic() - self.last_advance > 0.25:
            return "failed", {"failure": "stale physics state"}
        return self.outcome

    def cancel_reactive_command(self):
        env = self.env
        env.control.stop()
        deadline = time.monotonic() + 0.75
        while True:
            control = env.control
            velocity = env.data.qvel[control.base_dof : control.base_dof + 6] if control.base_dof >= 0 else np.zeros(6)
            if (
                not env._physics_invalid
                and np.isfinite(velocity).all()
                and np.max(np.abs(control.velocities())) < 0.03
                and np.linalg.norm(velocity[:2]) < 0.005
                and abs(velocity[5]) < 0.02
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            env._tick(np.zeros(env.action_space.shape), score=False)
