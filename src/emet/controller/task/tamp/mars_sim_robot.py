"""Independently stepped, actuator-only Mars facade for public-tool testing.

No network, hardware or object relocation during execution. Geometry and object
poses are available only to privileged planning/scoring; actor inputs are normal
proprioceptive history and declared task targets.
"""

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from emet.controller.manipulation.native_scene import model_signature
from emet.rl.manipulation.command_runtime import TaskPoseRuntimeTiming, task_pose_teacher_action
from emet.rl.manipulation.safety import relative_pose, retention_error, whole_body_stopped
from emet.rl.manipulation.tracking import TaskFrameFK

_CLOCK_LOCK = threading.Lock()


class MarsSimRobot:
    native_mars_task_control = True

    def __init__(
        self, env, *, policy=None, timing=None, record_observations=False, arm_execution="joint", arm_reference="legacy"
    ):
        if env.profile.robot != "innate_mars" or env.initialization_failed:
            raise ValueError("requires a prepared Mars simulation fixture")
        if arm_execution not in {"joint", "cartesian"} or (arm_execution == "cartesian" and policy is None):
            raise ValueError("Cartesian arm execution requires an explicit simulator policy")
        self.arm_execution = arm_execution
        if arm_reference not in {"legacy", "jerk_limited"} or (
            arm_reference != "legacy" and arm_execution != "cartesian"
        ):
            raise ValueError("jerk-limited references require Cartesian simulation execution")
        self.arm_reference = arm_reference
        self.reference_phase = None
        self.reference_phase_schedule = None
        self.env, self.model, self.data, self.servo = env, env.model, env.data, env.control
        self.signature = model_signature(self.model)
        self._state_lock = threading.RLock()
        self.boot = str(id(self))
        self.phase = "pregrasp"
        self.failure = None
        self.stop_confirmed = None
        self.payload = env.payload
        self.retention_reference = None
        self.last_bilateral_contact = False
        self.placement_support = "table"
        self.commands = []
        self.trace = []
        self.max_translation = self.max_rotation = 0.0
        self.policy = policy
        self.policy_calls = 0
        self.policy_error = None
        self.record_observations = bool(record_observations)
        self._pose_command_received = False
        self.timing = (timing or TaskPoseRuntimeTiming()).resolve(env.profile.dt)
        self.servo.action_mode = "direct_velocity"
        self.servo.preview_enabled = self.servo.pose_only_targets = True
        self.servo.set_physical_wheel_slew(True)
        self.servo.set_reference(self.servo.positions(), phase="pregrasp")
        self.servo.set_pose_targets(self.servo.base_pose(), np.ones(3, dtype=bool), self.servo.ee_pose(), True)
        self.refresh()

    @classmethod
    def from_checkpoint(
        cls,
        env,
        directory,
        *,
        evidence,
        timing=None,
        record_observations=False,
        arm_execution="joint",
        arm_reference="legacy",
    ):
        from emet.rl.manipulation.tamp_policy import CheckpointTampPolicy

        return cls(
            env,
            policy=CheckpointTampPolicy(directory, env, evidence=evidence),
            timing=timing,
            record_observations=record_observations,
            arm_execution=arm_execution,
            arm_reference=arm_reference,
        )

    def refresh(self):
        self.signature = model_signature(self.model)
        self._state_received_monotonic = time.monotonic()
        self._state = {
            "command_protocol": {"server_boot_id": self.boot},
            "native_tamp_robot": "innate_mars",
            "native_tamp_snapshot": {
                "source": "simulator_ground_truth",
                "model_signature": self.signature,
                "qpos": self.data.qpos.tolist(),
                "qvel": self.data.qvel.tolist(),
                "sim_time": float(self.data.time),
                "captured_at": time.time(),
            },
        }

    def get_emet_session(self):
        return {
            "is_simulation": True,
            "robot_name": "innate_mars",
            "sim_object_placements": {
                "object1": {"cat": "blue cube"},
                "object2": {"cat": "red cylinder"},
                "table": {"cat": "source support"},
                "destination": {"cat": "destination support"},
            },
        }

    def command_arm_reference(self, q, *, blocking):
        if self.failure or blocking:
            return False
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.isfinite(q).all():
            raise ValueError("expected five finite Mars arm coordinates")
        if np.any(q < self.servo.limits[:, 0]) or np.any(q > self.servo.limits[:, 1]):
            raise ValueError("arm reference outside robot joint limits")
        self.servo.clear_task_trajectory()
        # Preserve the reference's measured-frame semantics for the baseline.
        # A learned execution goal stays anchored at the commanded base pose:
        # measured base drift must not drag its world hand target along with it.
        base_frame = self.servo.base_goal if self.policy is not None else self.servo.base_pose()
        hand = TaskFrameFK(self.model, self.data, self.servo)(base_frame, q)
        self.commands.append(
            {
                "sim_time": float(self.data.time),
                "phase": self.phase,
                "arm": q.tolist(),
                "hand_pose": hand.tolist(),
                "base_pose": self.servo.base_goal.tolist(),
                "base_mask": [True, True, True],
                "hand_mask": True,
                "target_source": "certified_joint_witness_fk",
                "controller": self.controller_name,
            }
        )
        self.servo.set_reference(q, phase=self.phase, release=self.phase == "release" and self.servo.release_authorized)
        self.servo.set_pose_targets(self.servo.base_goal, np.ones(3, dtype=bool), hand, True)
        self._pose_command_received = True
        return True

    def support_load(self, support=None):
        from emet.rl.contact import solved_contact_force

        table = self.model.body(support or self.placement_support).id
        load = 0.0
        for i, c in enumerate(self.data.contact[: self.data.ncon]):
            if set(self.model.geom_bodyid[c.geom]) == {self.payload, table} and abs(c.frame[2]) > 0.8:
                load += solved_contact_force(self.model, self.data, i)[0]
        return load

    def step(self, duration):
        for _ in range(max(1, int(np.ceil(duration / self.env.profile.dt)))):
            cycle_started = time.monotonic()
            action = np.zeros(self.env.action_space.shape)
            inference_s = None
            observation = None
            physics_before = None
            action_source = "fault_stop"
            if self.phase == "place" and self.support_load() >= max(
                0.01, 0.05 * self.model.body_mass[self.payload] * 9.81
            ):
                self.servo.set_base_bias_compensation(False)
            if not self.failure:
                observation = self.servo.observe()
                if self.record_observations:
                    physics_before = {
                        "time": float(self.data.time),
                        "qpos": self.data.qpos.copy().tolist(),
                        "qvel": self.data.qvel.copy().tolist(),
                        "ctrl": self.data.ctrl.copy().tolist(),
                    }
                if self.policy is not None and (not self._pose_command_received or self.servo.holding):
                    # Planning/settling waits are not permission for an actor to
                    # wander. Keep the bounded actuator hold and collect history;
                    # only explicit arm/trajectory targets activate inference.
                    action_source = "executor_terminal_hold" if self.servo.holding else "executor_idle_hold"
                elif self.policy is not None:
                    action_source = "policy"
                    started = time.monotonic()
                    try:
                        self.policy_calls += 1
                        action = np.asarray(self.policy(observation), dtype=float)
                        if action.shape != self.env.action_space.shape or not np.isfinite(action).all():
                            raise ValueError("invalid policy action")
                        if np.any(abs(action) > 1 + 1e-6):
                            raise ValueError("policy action outside normalized limits")
                    except Exception as exc:
                        self.policy_error = f"{type(exc).__name__}: {exc}"
                        self.failure = "policy_inference_failed"
                    inference_s = time.monotonic() - started
                    if inference_s > self.timing["inference_budget_s"]:
                        self.failure = self.failure or "policy_inference_deadline"
                    if self.failure:
                        self.servo.stop()
                        action = np.zeros(self.env.action_space.shape)
                else:
                    action_source = "reference_teacher"
                    action = (
                        task_pose_teacher_action(self.servo)
                        if self.servo.task_trajectory is not None
                        else self._joint_teacher_action()
                    )
                evidence = self.env._tick(action, score=not self.failure)
            else:
                evidence = self.env._tick(np.zeros(self.env.action_space.shape), score=False)
            if self.policy is not None and time.monotonic() - cycle_started > self.timing["cycle_budget_s"]:
                self.failure = self.failure or "policy_cycle_deadline"
            self.last_bilateral_contact = bool(evidence.get("bilateral_contact", False))
            self.data, self.servo = self.env.data, self.env.control
            if self.env._physics_invalid:
                self.failure = self.failure or "invalid_physics"
            elif self.env.monitor.wheel_lift:
                self.failure = self.failure or "wheel_lift"
            elif self.env.monitor.peak_force > self.env.profile.unsafe_force:
                self.failure = self.failure or "unsafe_contact"
            if self.retention_reference is not None:
                translation, rotation = retention_error(
                    self.retention_reference, relative_pose(self.data, self.servo.ee, self.payload)
                )
                self.max_translation = max(self.max_translation, translation)
                self.max_rotation = max(self.max_rotation, rotation)
                if translation > 0.02 or rotation > 0.1:
                    self.failure = self.failure or "retention_lost"
            if self.failure:
                self.servo.stop()
            self.trace.append(
                {
                    "time": float(self.data.time),
                    "phase": self.phase,
                    "failure": self.failure,
                    "controller": self.controller_name,
                    "action_source": action_source,
                    "policy_calls": self.policy_calls,
                    "policy_inference_s": inference_s,
                    "normalized_action": action.tolist(),
                    "base": self.servo.base_pose().tolist(),
                    "base_goal": self.servo.base_goal.tolist(),
                    "hand": self.servo.ee_pose().tolist(),
                    "force": self.env.monitor.peak_force,
                    "joint_velocity": self.servo.velocities().tolist(),
                    "commanded_joint_velocity": self.servo.velocity.tolist(),
                    "commanded_base_velocity": self.servo.base_command.tolist(),
                    "applied_base_velocity": self.servo.applied_base_velocity.tolist(),
                    "joint_position": self.servo.positions().tolist(),
                    "hand_goal": self.servo.hand_goal.tolist(),
                    "payload_rotation": self.data.xmat[self.payload].copy().tolist(),
                    "gripper_position": float(self.data.qpos[self.servo.grip_q]),
                    "grip_target": self.servo.grip_target,
                    "grip_reference": self.servo.grip_reference,
                    "grip_torque_Nm": float(self.data.actuator_force[self.servo.grip_act]),
                    "bilateral_contact": self.last_bilateral_contact,
                    "payload_height": float(self.data.xpos[self.payload, 2]),
                    "trajectory_progress_s": self.servo.trajectory_elapsed() if self.servo.task_trajectory else None,
                    "trajectory_rate": self.servo.task_progress_rate if self.servo.task_trajectory else None,
                    "holding": self.servo.holding,
                }
            )
            if (
                self.record_observations
                and self.reference_phase is not None
                and self.servo.task_trajectory is self.reference_phase_schedule
            ):
                phase = self.reference_phase
                self.trace[-1]["reference_motion"] = self.servo.task_trajectory.sample_motion(
                    phase.progress,
                    rate=phase.rate,
                    phase_acceleration=phase.acceleration,
                    phase_jerk=phase.jerk,
                )
                self.trace[-1]["phase_acceleration"] = phase.acceleration
                self.trace[-1]["phase_jerk"] = phase.jerk
            if self.record_observations and observation is not None:
                self.trace[-1]["actor_observation"] = {k: v.tolist() for k, v in observation.items()}
                self.trace[-1]["physics_before"] = physics_before
                self.trace[-1]["physics_after"] = {
                    "time": float(self.data.time),
                    "qpos": self.data.qpos.copy().tolist(),
                    "qvel": self.data.qvel.copy().tolist(),
                    "ctrl": self.data.ctrl.copy().tolist(),
                }
        self.refresh()

    @property
    def controller_name(self):
        return (
            getattr(self.policy, "controller_name", "checkpoint_policy")
            if self.policy is not None
            else "reference_teacher"
        )

    def _joint_teacher_action(self):
        """Track the joint command certified by the physical arm executor.

        Its joint-space certificate must not be silently retargeted by the
        pose teacher's shoulder counter-rotation. Timed whole-body transport
        uses the pose teacher and its measured base-motion compensation.
        Both use the same actuator speed, acceleration and effort limits.
        """
        p = self.env.profile
        base = self.servo.base_feedback() / [p.base_speed, p.base_yaw_speed]
        arm = 4 * (self.servo.reference - self.servo.target) / p.speed
        return np.clip(np.r_[base, arm, 0.0], -1, 1)

    def follow_trajectory(self, path, *, loaded=True):
        from emet.controller.manipulation.physical_pick_place import PhysicalMotionResult

        if self.failure or self.servo.stopped:
            return PhysicalMotionResult(False, self.failure or "controller_stopped", "transport")
        self.phase = "carry" if loaded else "pregrasp"
        self.servo.set_reference(self.servo.reference, phase=self.phase)
        self.servo.set_base_bias_compensation(True)
        self.servo.set_task_trajectory(path)
        self._pose_command_received = True
        progress, rate = 0.0, 0.0
        self.servo.set_trajectory_progress(0.0, rate=0.0)
        # Reserve the required quiet dwell separately from the motion budget.
        # Convergence on the final motion tick must still permit verification.
        deadline = float(self.data.time) + 3 * path.end + 5
        # Geometric task contract with explicit, bounded retiming. Do not run
        # the hand ahead of an underactuated base and then demand an unreachable
        # instantaneous arm correction. Report the executed clock in the trace.
        for _ in range(int(np.ceil((3 * path.end + 4) / self.env.profile.dt))):
            delta = self.servo.base_goal - self.servo.base_pose()
            yaw = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            hand_error = np.linalg.norm(self.servo.ee_pose()[:3] - self.servo.hand_goal[:3])
            wanted = float(
                np.clip(
                    min(
                        1 - (np.linalg.norm(delta[:2]) / 0.005) ** 2,
                        1 - (yaw / 0.025) ** 2,
                        1 - (hand_error / 0.0075) ** 2,
                    ),
                    0,
                    1,
                )
            )
            rate += float(np.clip(wanted - rate, -2 * self.env.profile.dt, 2 * self.env.profile.dt))
            progress = min(path.end, progress + rate * self.env.profile.dt)
            self.servo.set_trajectory_progress(progress, rate=rate if progress < path.end else 0.0)
            self.step(self.env.profile.dt)
            if self.failure:
                return PhysicalMotionResult(False, self.failure, "transport")
            hand = self.servo.ee_pose()
            columns = self.servo.hand_goal[3:].reshape(3, 2)
            target = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
            current = self.data.xmat[self.servo.ee].reshape(3, 3)
            angle = np.arccos(np.clip((np.trace(target.T @ current) - 1) / 2, -1, 1))
            delta = self.servo.base_goal - self.servo.base_pose()
            yaw = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            base_v = self.data.qvel[self.servo.base_dof : self.servo.base_dof + 6].copy()
            reached = (
                progress >= path.end
                and np.linalg.norm(hand[:3] - self.servo.hand_goal[:3]) < 0.005
                and angle < 0.05
                and np.linalg.norm(delta[:2]) < 0.005
                and abs(yaw) < 0.04
                and np.max(abs(self.servo.velocities())) < 0.06
                and np.linalg.norm(base_v[:2]) < 0.01
                and abs(base_v[5]) < 0.04
            )
            if reached:
                if self.arm_execution == "cartesian":
                    # Clear transport drag compensation while the pose loop
                    # still controls the arm. Entering position hold with a
                    # moving chassis/arm can turn a small target rebase into a
                    # wheel-unloading transient. The quiet loop is viable with
                    # the separately validated low-gain wheel configuration.
                    self.servo.set_base_bias_compensation(False)
                    if not whole_body_stopped(self.servo):
                        continue
                # Enter hold once the pose is reached, then VERIFY settling.
                # Requiring quiet motion before holding can make tiny pose
                # corrections prevent the very settling we are waiting for.
                self.servo.clear_task_trajectory()
                self.servo.set_base_bias_compensation(False)
                self.servo.hold_position()
                remaining = min(3.0, max(0.0, deadline - float(self.data.time)))
                if self._verify_quiet_hold(1.0, remaining):
                    now = self.servo.ee_pose()
                    columns = now[3:].reshape(3, 2)
                    actual = np.column_stack([columns, np.cross(columns[:, 0], columns[:, 1])])
                    delta = self.servo.base_goal - self.servo.base_pose()
                    if (
                        np.linalg.norm(now[:3] - self.servo.hand_goal[:3]) < 0.005
                        and np.arccos(np.clip((np.trace(target.T @ actual) - 1) / 2, -1, 1)) < 0.05
                        and np.linalg.norm(delta[:2]) < 0.005
                        and abs(np.arctan2(np.sin(delta[2]), np.cos(delta[2]))) < 0.04
                    ):
                        return PhysicalMotionResult(True, "measured_pose_and_retention", "transport")
                break
        # Quiet-hold verification can discover a wheel/force/retention fault.
        # Preserve that first cause instead of disguising it as slow tracking.
        self.failure = self.failure or "transport_tracking_timeout"
        self.servo.stop()
        return PhysicalMotionResult(False, self.failure, "transport")

    def stop_and_confirm(self, reason):
        self.failure = self.failure or reason
        self.servo.stop()
        deadline = time.monotonic() + 0.75
        simulation_deadline = float(self.data.time) + 0.75
        while not self.env._physics_invalid:
            if whole_body_stopped(self.servo):
                self.stop_confirmed = True
                return True
            if time.monotonic() >= deadline or float(self.data.time) >= simulation_deadline:
                break
            self.step(self.env.profile.dt)
        self.stop_confirmed = False
        return False

    def confirm_support_contact(self):
        """Executor-only 6 mm / 3 s vertical probe, never a release command.

        Stop descending at measured half-weight support and hold the arm before
        confirming 200 ms of load. Existing retention/force/wheel guards remain
        active; excessive support load latches a fault. No configuration IK,
        geometry adjustment or object relocation occurs here.
        """
        from emet.rl.manipulation.tracking import TaskPoseTrajectory

        if self.failure or self.retention_reference is None or self.phase != "place":
            return False
        weight = self.model.body_mass[self.payload] * 9.81
        start = self.servo.ee_pose().copy()
        end = start.copy()
        end[2] -= 0.006
        base = self.servo.base_goal.copy()
        path = TaskPoseTrajectory(
            [0, 3], [base, base], [start, end], [self.servo.grip_reference] * 2, base_mask=[True] * 3
        )
        self.servo.set_task_trajectory(path)
        held = 0.0
        touched = False
        for _ in range(int(np.ceil(3.0 / self.env.profile.dt)) + 1):
            load = self.support_load()
            if self.failure or load > weight + 1.0 or self.servo.ee_pose()[2] < start[2] - 0.007:
                self.failure = self.failure or "support_probe_limit"
                self.servo.stop()
                return False
            if load >= 0.5 * weight:
                if not touched:
                    self.servo.clear_task_trajectory()
                    self.servo.set_base_bias_compensation(False)
                    self.servo.set_pose_targets(self.servo.base_goal, self.servo.base_mask, self.servo.ee_pose(), True)
                    self.servo.hold_arm_position()
                    touched = True
                held += self.env.profile.dt
                if held >= 0.2:
                    self.commands.append(
                        {
                            "sim_time": float(self.data.time),
                            "support_confirmed_N": load,
                            "probe_travel_m": float(start[2] - self.servo.ee_pose()[2]),
                        }
                    )
                    return True
            else:
                held = 0.0
            self.step(self.env.profile.dt)
        self.servo.clear_task_trajectory()
        self.servo.hold_arm_position()
        return False

    def finish_and_hold(self):
        """Complete only after two continuous seconds of measured quiet hold."""
        if self.failure or self.retention_reference is not None or not self.servo.gripper_is_open():
            return False
        self.servo.clear_task_trajectory()
        self.servo.set_base_bias_compensation(False)
        self.servo.set_pose_targets(self.servo.base_goal, self.servo.base_mask, self.servo.ee_pose(), True)
        self.servo.hold_position()
        if self._verify_quiet_hold(2.0, 3.5):
            return True
        self.failure = self.failure or "terminal_motion_unsettled"
        self.servo.stop()
        return False

    def _verify_quiet_hold(self, required_s, timeout_s):
        held = 0.0
        for _ in range(int(np.ceil(timeout_s / self.env.profile.dt))):
            self.step(self.env.profile.dt)
            quiet = not self.failure and whole_body_stopped(self.servo)
            held = held + self.env.profile.dt if quiet else 0.0
            if held >= required_s:
                return True
            if self.failure:
                return False
        return False

    def get_gripper_position(self):
        return float(self.data.qpos[self.servo.grip_q])

    def close_gripper(self, *, blocking):
        if self.failure:
            return False
        self.commands.append({"sim_time": float(self.data.time), "gripper": "close"})
        self.servo.set_reference(self.servo.reference, phase="close", grip=self.env.profile.gripper_closed)
        self.phase = "close"
        if blocking:
            self.step(1.0)
        return not self.failure

    def open_gripper(self, *, blocking):
        if self.failure:
            return False
        if self.retention_reference is not None:
            # Opening a loaded grasp is executor-only and requires physical
            # support. Never declare a dropped/unsupported object as placed.
            weight = self.model.body_mass[self.payload] * 9.81
            if self.support_load() < 0.5 * weight:
                self.failure = "release_support_unverified"
                self.servo.stop()
                return False
            self.servo.set_base_bias_compensation(False)
        self.retention_reference = None
        self.commands.append({"sim_time": float(self.data.time), "gripper": "open"})
        self.servo.set_reference(self.servo.reference, phase="release", release=True)
        self.phase = "release"
        if blocking:
            budget = float(np.ptp(self.servo.grip_limits)) / self.env.profile.gripper_speed + 0.5
            for _ in range(int(np.ceil(budget / self.env.profile.dt))):
                self.step(self.env.profile.dt)
                if self.failure:
                    return False
                if self.servo.gripper_is_open():
                    return True
            return False
        return True

    @contextmanager
    def stepped_clock(self):
        """Exclusive in-process harness; never patch the shared time module."""
        from emet.controller.manipulation import physical_pick_place
        from emet.controller.task.tamp import native_adapter

        if not _CLOCK_LOCK.acquire(blocking=False):
            raise RuntimeError("simulator tool clocks require isolated processes")
        try:
            old_executor, old_adapter = physical_pick_place.time, native_adapter.time
            clock = SimpleNamespace(sleep=self.step, monotonic=lambda: float(self.data.time))
            physical_pick_place.time = native_adapter.time = clock
            try:
                yield
            finally:
                physical_pick_place.time, native_adapter.time = old_executor, old_adapter
        finally:
            _CLOCK_LOCK.release()
