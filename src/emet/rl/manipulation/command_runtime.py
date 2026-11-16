"""Simulator-only task-pose execution through the shared command lifecycle.

Owns an existing manipulation environment, never resets physics or clears a
failed contract. One caller owns advance(); this is not a hardware transport.
Gripper schedules remain preview inputs; only the task executor can release.
"""

import time
from dataclasses import asdict, dataclass
from numbers import Real

import mujoco
import numpy as np

from emet.core.command_client import send_command
from emet.core.command_runtime import CommandRuntime
from emet.rl.manipulation.safety import whole_body_stopped
from emet.rl.manipulation.tracking import TaskPoseTrajectory


@dataclass(frozen=True)
class TaskPoseTeacherConfig:
    """Explicit spatial feedback units and least-squares task metric.

    Gains map m to m/s and rad to rad/s, respectively. The characteristic
    length converts angular rows to linear units in the damped solve. Defaults
    preserve the existing teacher (including its 1 m/rad heuristic); tuning is
    separate from this runtime repair and needs paired trajectory validation.
    """

    position_gain_per_s: float = 4.0
    orientation_gain_per_s: float = 4.0
    rotation_length_m_per_rad: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError("teacher gains and rotation length must be positive and finite")
            object.__setattr__(self, name, float(value))

    def feedback_twist(self, position_error_m, orientation_error_rad):
        return np.r_[
            self.position_gain_per_s * np.asarray(position_error_m),
            self.orientation_gain_per_s * np.asarray(orientation_error_rad),
        ]


@dataclass(frozen=True)
class TaskPoseRuntimeTiming:
    """Explicit simulator wall-time limits, never learned from contention.

    None budgets resolve to twice the profile control interval. The cycle
    budget includes observation, inference and stepping. Command expiry and
    measured-state freshness remain independent checks, including after inference.
    These settings do not configure a hardware watchdog or odometry freshness.
    """

    inference_budget_s: float | None = None
    cycle_budget_s: float | None = None
    max_state_age_s: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is None and name != "max_state_age_s":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError("runtime timing limits must be positive and finite")
            object.__setattr__(self, name, float(value))

    def resolve(self, dt):
        limits = {
            "inference_budget_s": 2 * dt if self.inference_budget_s is None else self.inference_budget_s,
            "cycle_budget_s": 2 * dt if self.cycle_budget_s is None else self.cycle_budget_s,
            "max_state_age_s": self.max_state_age_s,
        }
        if not np.isfinite(dt) or dt <= 0 or limits["cycle_budget_s"] < limits["inference_budget_s"]:
            raise ValueError("cycle budget must cover inference, with a positive control interval")
        return limits


def task_pose_teacher_action(control, *, config=None):
    """Bounded resolved-rate teacher; no iterative configuration IK at runtime.

    The baseline joint teacher cannot consume hand-only task schedules. Use a
    damped measured Jacobian here, with measured base motion compensation.
    The learned actor uses its normal observation path and never sees this solve.
    """
    from scipy.spatial.transform import Rotation

    c, p = control, control.profile
    config = TaskPoseTeacherConfig() if config is None else config
    if not isinstance(config, TaskPoseTeacherConfig):
        raise TypeError("expected TaskPoseTeacherConfig")
    c.update_task_reference()
    jp, jr = np.zeros((3, c.model.nv)), np.zeros((3, c.model.nv))
    mujoco.mj_jacBody(c.model, c.data, jp, jr, c.ee)
    jacobian = np.vstack([jp, jr])
    arm = np.column_stack([jacobian[:, group].mean(axis=1) for group in c.v])
    current = c.ee_pose()
    columns = c.hand_goal[3:].reshape(3, 2)
    goal_rotation = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
    rotation = c.data.xmat[c.ee].reshape(3, 3).copy()
    feedback = config.feedback_twist(
        c.hand_goal[:3] - current[:3], Rotation.from_matrix(goal_rotation @ rotation.T).as_rotvec()
    )
    metric = np.r_[np.ones(3), np.full(3, config.rotation_length_m_per_rad)]
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
    twist = feedforward + feedback - external_twist
    if p.wheel_bodies and not c.base_mask.any() and c.hand_mask:
        # With an unspecified base, a five-DOF arm cannot independently track
        # every hand pose. Include feasible differential-drive forward/yaw
        # motion in the same instantaneous solve. No inactive base target is
        # consulted, and this remains a teacher, never an actor constraint.
        yaw = c.base_pose()[2]
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        pivot = c.data.xpos[c.base] + c.data.xmat[c.base].reshape(3, 3) @ c.axle_offset
        yaw_axis = np.array([0.0, 0.0, 1.0])
        base_jacobian = np.column_stack(
            [
                np.r_[heading, np.zeros(3)],
                np.r_[np.cross(yaw_axis, current[:3] - pivot), yaw_axis],
            ]
        )
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(c.model, c.data, mujoco.mjtObj.mjOBJ_BODY, c.base, spatial, 0)
        pivot_velocity = spatial[3:] + np.cross(spatial[:3], pivot - c.data.xpos[c.base])
        measured = np.array([pivot_velocity @ heading, spatial[2]])
        twist += base_jacobian @ measured
        # Prefer arm motion when it suffices; use the base to supply missing
        # task directions. Shared speed/slew/stop guards still bound commands.
        scales = np.r_[0.5 * p.base_speed, 0.5 * p.base_yaw_speed, p.speed]
        whole_body = np.column_stack([base_jacobian, arm]) * scales * metric[:, None]
        velocity = scales * (
            whole_body.T @ np.linalg.solve(whole_body @ whole_body.T + 1e-4 * np.eye(6), twist * metric)
        )
        return np.clip(np.r_[velocity / np.r_[p.base_speed, p.base_yaw_speed, p.speed], 0.0], -1, 1)
    weighted_arm = arm * metric[:, None]
    velocity = weighted_arm.T @ np.linalg.solve(weighted_arm @ weighted_arm.T + 1e-4 * np.eye(6), twist * metric)
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
    record = trajectory.record()
    # Internal continuous-FK references must never be silently sent as a v1
    # linear pose schedule. A future wire extension needs explicit negotiation.
    parse_trajectory_goal({"trajectory": record})
    if not getattr(client, "_state", {}).get("task_pose_control"):
        raise RuntimeError("peer does not advertise task-pose control")
    return send_command(
        client,
        {
            "reactive_goal": {"trajectory": record},
            "reactive_timeout_s": motion_timeout_s,
        },
    )


class TaskPoseCommandRuntime(CommandRuntime):
    def __init__(self, env, *, policy=None, timing=None, teacher_config=None):
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
        timing = TaskPoseRuntimeTiming() if timing is None else timing
        if not isinstance(timing, TaskPoseRuntimeTiming):
            raise TypeError("expected TaskPoseRuntimeTiming")
        self.timing_limits = timing.resolve(env.profile.dt)
        self.teacher_config = TaskPoseTeacherConfig() if teacher_config is None else teacher_config
        if not isinstance(self.teacher_config, TaskPoseTeacherConfig):
            raise TypeError("expected TaskPoseTeacherConfig")
        self.last_timing = {}
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
        return super().command_message(
            {**message, "task_pose_control": True, "task_pose_runtime_timing": dict(self.timing_limits)}
        )

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
            "runtime_timing": dict(self.timing_limits),
            "teacher_parameters": asdict(self.teacher_config) if self.policy is None else None,
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
                        self.last_timing = {}
                        obs = self.env.control.observe()
                        action = (
                            task_pose_teacher_action(self.env.control, config=self.teacher_config)
                            if self.policy is None
                            else self.policy(obs)
                        )
                        self.last_timing["observe_and_infer_s"] = time.monotonic() - started
                        if self.last_timing["observe_and_infer_s"] > self.timing_limits["inference_budget_s"]:
                            raise RuntimeError("policy inference deadline exceeded")
                        # Inference can consume a command's remaining lifetime
                        # even when it meets its own budget. Check expiry and
                        # state freshness again before issuing an actuator step.
                        self.poll_reactive_command()
                        if self._reactive_command is not None and not self._reactive_fault:
                            _, _, done, truncated, info = self.env.step(action)
                            self.last_info = info
                            self.last_timing["cycle_s"] = time.monotonic() - started
                            if self.last_timing["cycle_s"] > self.timing_limits["cycle_budget_s"]:
                                raise RuntimeError("control step deadline exceeded")
                            if done or truncated:
                                self.outcome = (
                                    "succeeded" if info["is_success"] else "failed",
                                    {
                                        "code": info["code"],
                                        "measurements": info,
                                        "timing": dict(self.last_timing),
                                    },
                                )
                    except Exception as exc:
                        self.outcome = ("failed", {"failure": str(exc), "timing": dict(self.last_timing)})
                    self.last_advance = time.monotonic()
                    self.poll_reactive_command()
            self._last_step += 1

    def reactive_command_result(self, context):
        age = time.monotonic() - self.last_advance
        if not 0 <= age <= self.timing_limits["max_state_age_s"]:
            return "failed", {
                "failure": "stale physics state",
                "state_age_s": age if np.isfinite(age) else None,
                "max_state_age_s": self.timing_limits["max_state_age_s"],
            }
        return self.outcome

    def cancel_reactive_command(self):
        env = self.env
        env.control.stop()
        deadline = time.monotonic() + 0.75
        simulation_deadline = float(env.data.time) + 0.75
        while True:
            if not env._physics_invalid and whole_body_stopped(env.control):
                return True
            if time.monotonic() >= deadline or float(env.data.time) >= simulation_deadline:
                return False
            env._tick(np.zeros(env.action_space.shape), score=False)
