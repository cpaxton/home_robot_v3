# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""MuJoCo backend for the whole-body torque contract.

The first backend is adapter-matched Mars: the contract still emits torques,
but the simulated arm uses the same position-servo adapter intended for the
current XL430/XC430 hardware. This is deliberately not a true-torque plant.

The base is the planar ``MarsControl`` plant (``base_x``/``base_y``/``base_yaw``
velocity actuators), not the weak physical-wheel contact model. That keeps the
contract's base-velocity action realizable while the arm still uses the
retained position-servo plant.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import ServoCommand
from emet.controller.manipulation.wbc_contract import (
    ArmObservation,
    EmbodimentProfile,
    GripperObservation,
    WbcObservation,
)
from emet.controller.manipulation.wbc_profiles import mars_sim_profile
from emet.rl.control import MarsControl, invalid_physics
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.safety import ContactMonitor
from emet.rl.model import build_model, model_fingerprint
from emet.rl.profile import MARS


def _pose7(pose9: np.ndarray) -> tuple[float, ...]:
    """Convert the controller's position + two rotation columns to xyzw."""
    columns = np.asarray(pose9[3:], dtype=np.float64).reshape(3, 2)
    third = np.cross(columns[:, 0], columns[:, 1])
    rotation = np.column_stack([columns, third])
    quat = Rotation.from_matrix(rotation).as_quat()
    return tuple(float(v) for v in np.r_[pose9[:3], quat])


class MuJoCoWbcBackend:
    """Adapter-matched Mars plant with procedurally sampled whole-body goals."""

    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        *,
        seed: int = 0,
        control_dt: float = 0.05,
        target_range: float = 0.35,
        arm_goal_probability: float = 1.0,
        base_goal_probability: float = 0.5,
        base_goal_mode: str = "forward",
        scene: str = "procedural",
        gripper_mode: str = "hold_open",
    ) -> None:
        self.profile = profile or mars_sim_profile()
        if self.profile.meta("robot") != "innate_mars":
            raise ValueError("MuJoCoWbcBackend currently supports the innate_mars profile only")
        if not np.isfinite([control_dt, target_range, arm_goal_probability, base_goal_probability]).all():
            raise ValueError("backend timing/range values must be finite")
        if (
            control_dt <= 0
            or target_range < 0
            or not 0 <= arm_goal_probability <= 1
            or not 0 <= base_goal_probability <= 1
        ):
            raise ValueError("invalid backend timing/range values")
        if self.profile.controlled_joint_names != tuple(MARS.arm_joints):
            raise ValueError("contract joint order does not match the Mars control model")
        if self.profile.arms[0].gripper is None or self.profile.arms[0].gripper.joint_names != ("joint6",):
            raise ValueError("contract gripper does not match the Mars control model")
        if any(arm.target_frame != "world" for arm in self.profile.arms):
            raise ValueError("MuJoCoWbcBackend emits world-frame EE poses; use target_frame='world'")
        if scene not in {"procedural", "table"}:
            raise ValueError("WBC MuJoCo scene must be procedural or table")
        if gripper_mode not in {"hold_open", "policy"}:
            raise ValueError("gripper_mode must be hold_open or policy")
        if base_goal_mode not in {"forward", "any"}:
            raise ValueError("base_goal_mode must be forward or any")
        self.gripper_mode = gripper_mode
        self.base_goal_mode = base_goal_mode
        self.scene = scene
        self.control_dt = float(control_dt)
        self.target_range = float(target_range)
        self.arm_goal_probability = float(arm_goal_probability)
        self.base_goal_probability = float(base_goal_probability)
        self.rng = np.random.default_rng(seed)
        self.control_profile = PROFILES["innate_mars"]
        self.model = build_model(scene)
        # The upstream Mars XML carries 360 unused sensors. The actor uses
        # contract observations only, so skip sensor evaluation at every
        # physics step (same choice as MarsReachEnv).
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        self.model_fingerprint = model_fingerprint()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = MarsControl(self.model, self.data)
        # The upstream planar base has no physical wheels; treat the base body's
        # floor contact as support rather than collision force.
        monitor_profile = replace(self.control_profile, wheel_bodies=("base_link",))
        self.monitor = ContactMonitor(self.model, monitor_profile)
        self.substeps = max(1, round(self.control_dt / self.model.opt.timestep))
        self._dofs = np.asarray(self.control.arm_v, dtype=int)
        self._grip_q = int(self.model.jnt_qposadr[self.model.joint("joint6").id])
        self._grip_v = int(self.model.jnt_dofadr[self.model.joint("joint6").id])
        self._grip_held_index = tuple(MARS.held_joints).index("joint6")
        self._target_ee: tuple[float, ...] = (0.0,) * 7
        self._target_ee9: np.ndarray | None = None
        self._target_q: np.ndarray | None = None
        self._target_base: tuple[float, float, float] | None = None
        self._arm_command = np.zeros(len(MARS.arm_joints))
        self._base_command = np.zeros(2)
        self._grip_command = 0.0
        self._teacher_phase = "base"
        self._elapsed = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.control.base_q] = (0.0, 0.0, 0.0)
        q0 = np.clip(
            self.rng.uniform(-0.15, 0.15, size=len(self.control.arm_q)),
            self.control.limits[:, 0],
            self.control.limits[:, 1],
        )
        self.data.qpos[self.control.arm_q] = q0
        mujoco.mj_forward(self.model, self.data)
        self.control.reset()
        self._arm_command = self.control.arm_positions().copy()
        self._base_command = np.zeros(2)
        self._grip_command = self.control_profile.gripper_open
        self.control.held_values[self._grip_held_index] = self._grip_command
        self.control.apply()
        if self.rng.random() < self.base_goal_probability:
            base = self.control.base_pose()
            if self.base_goal_mode == "forward":
                # A differential base needs most of a short horizon to turn
                # around; start with reachable forward-hemisphere goals.
                distance = float(self.rng.uniform(0.05, 0.12))
                bearing = float(self.rng.uniform(-0.25, 0.25))
                heading = base[2] + bearing
                target_x = base[0] + distance * np.cos(heading)
                target_y = base[1] + distance * np.sin(heading)
                target_yaw = base[2] + self.rng.uniform(-0.1, 0.1)
            else:
                target_x = base[0] + self.rng.uniform(-0.15, 0.15)
                target_y = base[1] + self.rng.uniform(-0.15, 0.15)
                target_yaw = base[2] + self.rng.uniform(-0.3, 0.3)
            self._target_base = (
                float(target_x),
                float(target_y),
                float(target_yaw),
            )
        else:
            self._target_base = None
        if self.rng.random() < self.arm_goal_probability:
            # Sample the arm target at the base goal so the world EE pose
            # remains reachable after the base arrives.
            self._target_ee = self._sample_ee_target(self._target_base)
        else:
            self._target_ee = None
            self._target_q = self.control.arm_positions().copy()
            self._target_ee9 = None
        self._teacher_phase = "base" if self._target_base is not None else "arm"
        self._elapsed = 0.0
        self.monitor.reset()
        return self._observation()

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")
        if not command.joint_positions:
            raise ValueError("MuJoCoWbcBackend uses the position-servo adapter and requires joint_positions")
        target = np.asarray(
            [command.joint_positions[name] for name in self.profile.controlled_joint_names],
            dtype=np.float64,
        )
        if target.shape != self.control.limits[:, 0].shape or not np.isfinite(target).all():
            raise ValueError("invalid contract joint position target")
        # Executor-side slew: the policy may jump, but the actuator target and
        # base/gripper commands move at the retained controller's physical
        # rates. This prevents qacc spikes from being reported as invalid
        # physics instead of a bounded command.
        joint_step = MARS.max_joint_speed * self.control_dt
        self._arm_command += np.clip(target - self._arm_command, -joint_step, joint_step)
        self._arm_command = np.clip(self._arm_command, self.control.limits[:, 0], self.control.limits[:, 1])
        self.control.targets = self._arm_command.copy()
        # This backend directly owns bounded position targets; the residual
        # controller's stall latch is for velocity-residual actions and would
        # otherwise freeze a joint during the first high-torque samples.
        self.control.clear_obstruction()
        self.control.last_action[:] = 0.0
        base = np.asarray(command.base_velocity, dtype=np.float64)
        if base.shape != (2,) or not np.isfinite(base).all():
            raise ValueError("MuJoCoWbcBackend requires a planar two-value base velocity")
        desired_base = np.clip(base, [-MARS.max_v, -MARS.max_w], [MARS.max_v, MARS.max_w])
        base_step = np.array([MARS.max_linear_acceleration, MARS.max_angular_acceleration]) * self.control_dt
        self._base_command += np.clip(desired_base - self._base_command, -base_step, base_step)
        self.control.last_action[0] = float(self._base_command[0] / MARS.max_v)
        self.control.last_action[1] = float(self._base_command[1] / MARS.max_w)
        if self.gripper_mode == "policy" and "primary" in command.grippers:
            values = command.grippers["primary"].values
            if len(values) != 1 or not np.isfinite(values[0]):
                raise ValueError("Mars gripper command must contain one finite value")
            grip_limit = self.control_profile.gripper_open
            desired_grip = float(np.clip(values[0], -grip_limit, grip_limit))
            grip_step = self.control_profile.gripper_speed * self.control_dt
            self._grip_command += float(np.clip(desired_grip - self._grip_command, -grip_step, grip_step))
            self.control.held_values[self._grip_held_index] = self._grip_command

        terminated = False
        invalid = False
        invalid_state: dict[str, float] = {}
        evidence: dict[str, Any] = {"force": 0.0, "wheel_lift": False}
        safe_qpos = self.data.qpos.copy()
        safe_qvel = self.data.qvel.copy()
        safe_time = float(self.data.time)
        for _ in range(self.substeps):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            if invalid_physics(self.data):
                invalid_state = {
                    "ncon": float(self.data.ncon),
                    "nefc": float(self.data.nefc),
                    "maxuse_arena": float(self.data.maxuse_arena),
                    "narena": float(self.data.narena),
                    "qpos_max": float(np.max(np.abs(self.data.qpos), initial=0.0)),
                    "qvel_max": float(np.max(np.abs(self.data.qvel), initial=0.0)),
                    "qacc_max": float(np.max(np.abs(self.data.qacc), initial=0.0)),
                }
                self.data.qpos[:] = safe_qpos
                self.data.qvel[:] = safe_qvel
                self.data.time = safe_time
                mujoco.mj_forward(self.model, self.data)
                self.control.stop()
                terminated = invalid = True
                break
            evidence = self.monitor.update(self.data)
            if evidence.get("invalid_contact"):
                terminated = True
                break
            safe_qpos = self.data.qpos.copy()
            safe_qvel = self.data.qvel.copy()
            safe_time = float(self.data.time)
        if not terminated:
            mujoco.mj_forward(self.model, self.data)
            evidence = self.monitor.update(self.data)
            self._elapsed += self.control_dt
            contact_force = float(evidence.get("force", 0.0))
            if evidence.get("wheel_lift") or contact_force > self.control_profile.unsafe_force:
                terminated = True
        info = {
            "contact_force": float(evidence.get("force", 0.0)),
            "collision_measured": True,
            "wheel_lift": bool(evidence.get("wheel_lift", False)),
            "invalid_physics": invalid,
            "invalid_state": invalid_state,
            "elapsed_s": self._elapsed,
            "model_fingerprint": self.model_fingerprint,
        }
        return self._observation(), terminated, info

    def _base_tracking_error(self) -> tuple[float, float]:
        if self._target_base is None:
            return 0.0, 0.0
        base = self.control.base_pose()
        dx = self._target_base[0] - base[0]
        dy = self._target_base[1] - base[1]
        yaw_error = float(
            np.arctan2(
                np.sin(self._target_base[2] - base[2]),
                np.cos(self._target_base[2] - base[2]),
            )
        )
        return float(np.hypot(dx, dy)), abs(yaw_error)

    def teacher_action(self) -> np.ndarray:
        """Privileged PD teacher for behavior-cloning warm start.

        Base goals are staged before the arm: the base drives to its target,
        then holds while an IK teacher solves the arm against the settled base.
        The deployed actor never receives the sampled joint target or phase
        flag; it sees only the contract observation.
        """
        if self._target_q is None:
            raise ValueError("teacher action requires a sampled target")
        joints = self.profile.joint_group(self.profile.arms[0].joint_group).joints
        q = self.control.arm_positions().copy()
        qd = np.asarray(self.data.qvel[self.control.arm_v], dtype=np.float64)
        kp = np.asarray([joint.kp_eff for joint in joints], dtype=np.float64)
        kd = np.asarray([joint.kd_eff or 0.0 for joint in joints], dtype=np.float64)
        limits = np.asarray([joint.torque_limit for joint in joints], dtype=np.float64)
        action = np.zeros(self.profile.action_dim, dtype=np.float32)

        base_distance, base_heading = self._base_tracking_error()
        base_velocity = self.control.base_velocity()
        if self._target_base is not None and self._teacher_phase == "base":
            quiet = abs(float(base_velocity[0])) < 0.01 and abs(float(base_velocity[2])) < 0.05
            if base_distance <= 0.03 and base_heading <= 0.03 and quiet:
                self._teacher_phase = "arm"
            else:
                base = self.control.base_pose()
                dx = self._target_base[0] - base[0]
                dy = self._target_base[1] - base[1]
                target_yaw_error = float(
                    np.arctan2(
                        np.sin(self._target_base[2] - base[2]),
                        np.cos(self._target_base[2] - base[2]),
                    )
                )
                if base_distance > 0.004:
                    bearing = float(np.arctan2(dy, dx))
                    bearing_error = float(np.arctan2(np.sin(bearing - base[2]), np.cos(bearing - base[2])))
                    forward = float(min(MARS.max_v, max(0.01, 0.8 * base_distance)) * max(0.0, np.cos(bearing_error)))
                    # In the forward curriculum, hold the target heading while
                    # driving; chasing the instantaneous bearing causes orbit.
                    heading_error = target_yaw_error if self.base_goal_mode == "forward" else bearing_error
                else:
                    heading_error = target_yaw_error
                    forward = 0.0
                yaw_rate = float(np.clip(1.0 * heading_error - 0.2 * float(base_velocity[2]), -MARS.max_w, MARS.max_w))
                action[0] = forward / MARS.max_v
                action[1] = yaw_rate / MARS.max_w

        if self._teacher_phase == "arm" and self._target_ee is not None:
            q_des = self._target_q
            if self._target_base is not None and self._target_ee9 is not None:
                try:
                    _, q_des = self.control.solve_ee_pose(
                        self._target_ee9,
                        base=self.control.base_pose(),
                        base_locked=True,
                    )
                except ValueError:
                    q_des = self._target_q
            torque = np.clip(kp * (q_des - q) - kd * qd, -limits, limits)
        else:
            # Hold the arm while the base settles, or for a base-only task.
            torque = np.clip(-kd * qd, -limits, limits)
        start = self.profile.base.action_dim
        action[start : start + len(joints)] = torque / limits
        if self.profile.torso is not None:
            raise ValueError("Mars teacher does not define a torso action")
        return action

    def _set_arm_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.control.arm_q] = q

    def _sample_ee_target(self, base_target: tuple[float, float, float] | None = None) -> tuple[float, ...]:
        base_q = self.data.qpos[self.control.base_q].copy()
        if base_target is not None:
            self.data.qpos[self.control.base_q] = base_target
            mujoco.mj_forward(self.model, self.data)
        q0 = self.control.arm_positions().copy()
        ee0 = self.data.xpos[self.control.ee_body].copy()
        try:
            for _ in range(64):
                q = np.clip(
                    q0 + self.rng.uniform(-self.target_range, self.target_range, size=q0.size),
                    self.control.limits[:, 0],
                    self.control.limits[:, 1],
                )
                self._set_arm_q(q)
                mujoco.mj_forward(self.model, self.data)
                ee = self.data.xpos[self.control.ee_body]
                if np.linalg.norm(ee - ee0) >= 0.04:
                    self._target_q = q.copy()
                    self._target_ee9 = np.r_[
                        ee, self.data.xmat[self.control.ee_body].reshape(3, 3)[:, :2].ravel()
                    ].copy()
                    return _pose7(self._target_ee9)
            self._target_q = q0.copy()
            self._target_ee9 = np.r_[
                self.data.xpos[self.control.ee_body],
                self.data.xmat[self.control.ee_body].reshape(3, 3)[:, :2].ravel(),
            ].copy()
            return _pose7(self._target_ee9)
        finally:
            self._set_arm_q(q0)
            self.data.qpos[self.control.base_q] = base_q
            mujoco.mj_forward(self.model, self.data)
            self.control.reset()
            self.control.held_values[self._grip_held_index] = self.control_profile.gripper_open
            self.control.apply()

    def _observation(self) -> WbcObservation:
        q = self.control.arm_positions().copy()
        qd = np.asarray(self.data.qvel[self.control.arm_v], dtype=np.float64)
        torque = np.asarray(self.data.qfrc_actuator, dtype=np.float64)[self._dofs]
        base_twist = self.control.base_velocity()[[0, 2]]
        gripper = self.profile.arms[0].gripper
        assert gripper is not None
        ee = np.r_[
            self.data.xpos[self.control.ee_body],
            self.data.xmat[self.control.ee_body].reshape(3, 3)[:, :2].ravel(),
        ]
        arm = ArmObservation(
            name=self.profile.arms[0].name,
            joint_position=tuple(float(v) for v in q),
            joint_velocity=tuple(float(v) for v in qd),
            joint_torque=tuple(float(v) for v in torque),
            ee_pose=_pose7(ee),
            ee_target=self._target_ee,
            target_mask=(True, True),
            gripper=GripperObservation(
                position=(float(self.data.qpos[self._grip_q]),),
                velocity=(float(self.data.qvel[self._grip_v]),),
                target=(float(self._grip_command),),
            ),
        )
        return WbcObservation(
            base_pose=tuple(float(v) for v in self.control.base_pose()),
            base_twist=tuple(float(v) for v in base_twist),
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )

    def close(self) -> None:
        return None


__all__ = ["MuJoCoWbcBackend"]
