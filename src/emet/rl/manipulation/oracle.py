"""Actuator-only evaluation seam for the privileged native TAMP harness.

The oracle may see ground truth; the residual policy receives only the normal
controller history. This adapter is for an independently stepped simulator,
not a hardware client or permission to relax the executor's acceptance checks.
"""

import mujoco
import numpy as np

from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.profiles import STRETCH, require_finite
from emet.rl.manipulation.safety import ContactMonitor, relative_pose, retention_error


def stretch_arm_reference(command):
    q = require_finite(command, (6,))
    if q[0] != 0:
        raise ValueError("stationary arm adapter cannot execute base motion")
    # Public arm_to order: base, lift, extension, roll, pitch, yaw.
    return q[[1, 2, 5, 4, 3]]


def oracle_robot_class(base_class, *, controller="feedback", policy=None):
    if controller not in {"direct", "feedback", "policy"} or (controller == "policy") != (policy is not None):
        raise ValueError("invalid oracle controller selection")

    class OracleRobot(base_class):
        latest = None

        def __init__(self, *args, **kwargs):
            self.servo = None
            self.monitor = None
            self.phase = "pregrasp"
            self.failure = None
            self.payload = None
            self.retention_reference = None
            self.max_translation = self.max_rotation = 0.0
            self.paths = []
            self.trace = []
            super().__init__(*args, **kwargs)
            self.servo = ManipulationControl(self.model, self.data, STRETCH)
            self.servo.action_mode = policy.action_mode if policy is not None else "residual"
            self.servo.pose_only_targets = policy is not None and policy.manifest.get("pose_only_targets", False)
            self.servo.preview_enabled = policy is not None and "preview" in policy.normalization
            if self.servo.preview_enabled or self.servo.pose_only_targets:
                from emet.rl.manipulation.tracking import TaskFrameFK

                self.reference_fk = TaskFrameFK(self.model, self.data, self.servo)
            # Preserve the teacher's warm-start actuator targets, including head.
            if controller == "direct":
                self.data.ctrl[:] = self.initial_ctrl
            self.monitor = ContactMonitor(self.model, STRETCH)
            self.next_control = self.next_trace = float(self.data.time)
            OracleRobot.latest = self

        def step(self, duration):
            if self.servo is None:
                super().step(duration)
                self.initial_ctrl = self.data.ctrl.copy()
                return
            for _ in range(max(1, round(duration / self.model.opt.timestep))):
                if self.data.time + 1e-9 >= self.next_control:
                    if controller != "direct":
                        obs = self.servo.observe()
                        action = np.zeros(self.servo.profile.action_size) if policy is None else policy(obs)
                        self.servo.accept(action)
                    self.next_control += STRETCH.dt
                if controller != "direct" or self.failure:
                    self.servo.apply()
                mujoco.mj_step(self.model, self.data)
                metrics = self.monitor.update(self.data, self.payload)
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    self.failure = self.failure or "invalid_physics"
                elif self.monitor.wheel_lift:
                    self.failure = self.failure or "wheel_lift"
                elif metrics["force"] > STRETCH.unsafe_force:
                    self.failure = self.failure or "unsafe_contact"
                if self.retention_reference is not None:
                    translation, rotation = retention_error(
                        self.retention_reference, relative_pose(self.data, self.servo.ee, self.payload)
                    )
                    self.max_translation = max(self.max_translation, translation)
                    self.max_rotation = max(self.max_rotation, rotation)
                    if translation > 0.02 or rotation > 0.1:
                        self.failure = self.failure or "retention_lost"
                if self.failure and not self.servo.stopped:
                    self.servo.stop()
                if self.data.time + 1e-9 >= self.next_trace:
                    self.trace.append(
                        {
                            "time": float(self.data.time),
                            "phase": self.phase,
                            "q": self.servo.positions().tolist(),
                            "reference": self.servo.reference.tolist(),
                            "velocity": self.servo.velocities().tolist(),
                            "grip_target": self.servo.grip_target,
                            "force": metrics["force"],
                            "wheel_gap": metrics["wheel_gap"].tolist(),
                            "failure": self.failure,
                        }
                    )
                    self.next_trace += STRETCH.dt

        def arm_to(self, q, *, head, blocking):
            target = stretch_arm_reference(q)
            if self.failure:
                return False
            if controller == "direct":
                result = super().arm_to(q, head=head, blocking=blocking)
                self.servo.reference = target
                return result
            if blocking:
                raise ValueError("TAMP owns measured convergence")
            self.commands.append({"sim_time": float(self.data.time), "arm": list(q), "head": list(head)})
            self.servo.set_reference(target, phase=self.phase)
            if self.servo.preview_enabled or self.servo.pose_only_targets:
                self.servo.set_pose_targets(
                    self.servo.base_pose(),
                    np.zeros(3, dtype=bool),
                    self.reference_fk(self.servo.base_pose(), target),
                    True,
                )
            for name, value in zip(("head_pan", "head_tilt"), require_finite(head, (2,)), strict=True):
                self.data.ctrl[self.model.actuator(name).id] = value
            return True

        def close_gripper(self, *, blocking):
            if self.failure:
                return False
            if controller == "direct":
                result = super().close_gripper(blocking=blocking)
                self.servo.grip_target = float(self.data.ctrl[self.servo.grip_act])
                return result
            self.commands.append({"sim_time": float(self.data.time), "gripper": "close"})
            self.servo.set_reference(self.servo.reference, phase="close", grip=STRETCH.gripper_closed)
            return True

        def open_gripper(self, *, blocking):
            if self.failure:
                return False
            self.retention_reference = None  # Only the executor can request release.
            if controller == "direct":
                self.servo.grip_target = STRETCH.gripper_open
                return super().open_gripper(blocking=blocking)
            self.commands.append({"sim_time": float(self.data.time), "gripper": "open"})
            self.servo.set_reference(self.servo.reference, phase="release", grip=STRETCH.gripper_open, release=True)
            if not blocking:
                return True
            # The bounded-speed servo needs a travel-time budget, not the
            # direct position actuator's fixed one-second sleep.
            timeout = float(np.ptp(self.servo.grip_limits)) / STRETCH.gripper_speed + 0.5
            for _ in range(int(np.ceil(timeout / STRETCH.dt))):
                self.step(STRETCH.dt)
                if self.failure:
                    return False
                if abs(self.data.qpos[self.servo.grip_q] - STRETCH.gripper_open) < 0.005:
                    return True
            return False

    return OracleRobot


def instrument_adapter(adapter_class):
    """Record actual oracle paths and phase; leave planning/scoring code untouched."""

    class InstrumentedAdapter(adapter_class):
        def _executor(self, obj):
            executor = super()._executor(obj)
            original = executor._execute_path

            def execute(phase, path):
                self.robot.payload = self.model.body(obj).id
                if phase == "lift":
                    self.robot.retention_reference = relative_pose(
                        self.robot.data, self.robot.servo.ee, self.robot.payload
                    )
                self.robot.phase = {
                    "pregrasp": "pregrasp",
                    "grasp": "pregrasp",
                    "lift": "carry",
                    "preplace": "carry",
                    "place": "place",
                    "retreat": "reach",
                }.get(phase, "reach")
                self.robot.paths.append(
                    {
                        "phase": phase,
                        "object": obj,
                        "joint_names": list(executor.joint_names),
                        "waypoints": np.asarray(path).tolist(),
                        "start_qpos": self.robot.data.qpos.tolist(),
                        "start_qvel": self.robot.data.qvel.tolist(),
                        "start_ctrl": self.robot.data.ctrl.tolist(),
                        "start_time": float(self.robot.data.time),
                    }
                )
                return original(phase, path)

            executor._execute_path = execute
            return executor

    return InstrumentedAdapter
