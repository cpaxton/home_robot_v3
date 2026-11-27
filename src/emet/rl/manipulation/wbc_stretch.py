# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo WBC backend for the Stretch manipulation plant.

The contract profile exposes the same five motion axes the retained
``ManipulationControl`` drives: ``lift``, the virtual telescope axis ``arm``
(the sum of the four coupled telescope joints), and the three wrist joints.
The gripper is a separate position channel. ``ManipulationControl`` owns the
position-servo targets, the differential-wheel base command and the grip
command; this backend only translates :class:`ServoCommand` into those
targets, advances physics, and reads a contract observation back out.

The base is the physical wheel/freejoint plant rather than an ideal planar
base. Wheel slip means the measured base velocity trails the command, so the
contract observation reports measured velocity. The legacy Mars pose/velocity
feedback loop is disabled; a Stretch base command is a wheel-velocity
setpoint.

Deliberate limitations:

- Collision evidence is measured from solved MuJoCo contacts. Wheel-floor and
  caster-floor contacts are support; chassis/arm contacts with the table or
  scene and non-gripper self-collisions are reported as collisions. A hand
  contact is treated as intentional reach/grasp contact, not a collision.
- The arm target is sampled by forward kinematics at the reset base pose. If a
  base goal is also sampled the world-frame EE target does not move with the
  base, so a both-goals episode is a reach-after-drive task.
- The teacher is a privileged PD/P controller over the sampled targets, not an
  IK solver. It is intended for behavior-cloning warm starts.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import ServoCommand
from emet.controller.manipulation.wbc_contract import (
    ArmAction,
    ArmObservation,
    EmbodimentProfile,
    GripperCommand,
    GripperObservation,
    WbcAction,
    WbcObservation,
    normalized_action_vector,
)
from emet.controller.manipulation.wbc_profiles import stretch_sim_profile
from emet.rl.manipulation.control import ManipulationControl
from emet.rl.manipulation.models import build_manipulation_model
from emet.rl.manipulation.profiles import PROFILES
from emet.rl.manipulation.wbc_contact import MuJoCoContactSensor

# The packaged scene's zero posture folds the wrist into the chassis, so the
# plant resets to a raised ready pose instead. Lift and the virtual telescope
# axis are meters; the wrists are radians.
_READY_Q = np.array([0.50, 0.15, 0.0, 0.20, 0.0])

# Sampling half-widths for the five-axis arm target. Lift and the virtual
# telescope axis move in meters; the wrists move in radians.
_TARGET_SCALE = np.array([0.12, 0.12, 0.35, 0.35, 0.50])
_TARGET_MIN_EE_DISPLACEMENT_M = 0.04
_TARGET_SAMPLE_ATTEMPTS = 64

# Bodies that may touch the world/environment without rejecting a sampled
# posture. The arm must not start a reach against its own chassis or the table.
_NON_ARM_ROBOT_BODIES = frozenset({"base_link", "link_left_wheel", "link_right_wheel"})
_ENVIRONMENT_PREFIXES = ("table", "object")

# Forward-hemisphere base goal envelope, mirroring the Mars backend.
_BASE_GOAL_DISTANCE_M = (0.05, 0.12)
_BASE_GOAL_BEARING_RAD = 0.25
_BASE_GOAL_YAW_RAD = 0.10

# Teacher staging and gains.
_BASE_POSITION_GAIN = 4.0
_BASE_YAW_GAIN = 3.0
_BASE_LATERAL_GAIN = 3.0
_BASE_SETTLE_DISTANCE_M = 0.01
_BASE_SETTLE_HEADING_RAD = 0.02
_BASE_SETTLE_SPEED_M_S = 0.01
_BASE_SETTLE_YAW_RATE_RAD_S = 0.02
_BASE_STAGE_TIMEOUT_S = 2.5


def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _pose7(pose9: np.ndarray) -> tuple[float, ...]:
    """Convert position + two rotation columns to an xyzw pose tuple."""
    columns = np.asarray(pose9[3:], dtype=np.float64).reshape(3, 2)
    third = np.cross(columns[:, 0], columns[:, 1])
    rotation = np.column_stack([columns, third])
    quat = Rotation.from_matrix(rotation).as_quat()
    return tuple(float(v) for v in np.r_[pose9[:3], quat])


class StretchWbcBackend:
    """MuJoCo position-servo plant behind the WBC ``WbcBackend`` protocol."""

    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        seed: int = 0,
        control_dt: float = 0.05,
        target_range: float = 0.15,
        arm_goal_probability: float = 1.0,
        base_goal_probability: float = 0.5,
        base_goal_mode: str = "forward",
    ) -> None:
        self.profile = profile or stretch_sim_profile()
        if self.profile.meta("robot") != "stretch":
            raise ValueError("StretchWbcBackend requires a stretch profile")
        if self.profile.base.kind != "planar" or self.profile.base.action_dim != 2:
            raise ValueError("StretchWbcBackend requires a planar two-axis base")
        if any(arm.target_frame != "world" for arm in self.profile.arms):
            raise ValueError("StretchWbcBackend emits world-frame EE poses; use target_frame='world'")
        if len(self.profile.arms) != 1:
            raise ValueError("StretchWbcBackend supports exactly one arm")
        if not np.isfinite([control_dt, arm_goal_probability, base_goal_probability]).all():
            raise ValueError("backend timing/probability values must be finite")
        if control_dt <= 0 or not 0 <= arm_goal_probability <= 1 or not 0 <= base_goal_probability <= 1:
            raise ValueError("invalid backend timing/probability values")
        if arm_goal_probability > 0 and target_range <= 0:
            raise ValueError("target_range must be positive when arm goals are enabled")
        if base_goal_mode not in {"forward", "any"}:
            raise ValueError("base_goal_mode must be forward or any")

        self.control_dt = float(control_dt)
        self.arm_goal_probability = float(arm_goal_probability)
        self.base_goal_probability = float(base_goal_probability)
        self.base_goal_mode = base_goal_mode
        self.target_range = float(target_range)
        self._target_scale = _TARGET_SCALE * (self.target_range / 0.15 if self.target_range > 0 else 1.0)
        self.rng = np.random.default_rng(seed)

        self.control_profile = PROFILES["stretch"]
        self.model, self.model_fingerprint = build_manipulation_model("stretch")
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.control = ManipulationControl(self.model, self.data, self.control_profile)
        # A Stretch base command is the wheel-velocity setpoint. The legacy
        # Mars pose/bias feedback must never reinterpret it.
        self.control.base_feedback_at_rest = False
        self.control.base_velocity_feedback_gain = 0.0
        self.control.base_bias_compensation_enabled = False

        arm_spec = self.profile.arms[0]
        base_body = int(self.model.body("base_link").id)
        support_geoms = tuple(
            index
            for index in range(self.model.ngeom)
            if int(self.model.geom_bodyid[index]) == base_body
            and int(self.model.geom_type[index]) == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        )
        self._contact_sensor = MuJoCoContactSensor(
            self.model,
            base_body="base_link",
            ee_body=arm_spec.ee_link,
            hand_bodies=("link_SG3_gripper_body",),
            finger_bodies=("link_gripper_finger_left", "link_gripper_finger_right"),
            support_bodies=("base_link", "link_left_wheel", "link_right_wheel", "link_docking_station"),
            support_geoms=support_geoms,
        )
        self.collision_measured = True
        self._joints = self.profile.joint_group(arm_spec.joint_group).joints
        expected_names = tuple(joint.name for joint in self._joints)
        if expected_names != ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll"):
            raise ValueError("contract joint order does not match the Stretch control model")
        if tuple(self.profile.controlled_joint_names) != expected_names:
            raise ValueError("contract controlled joints must be the five Stretch motion axes")
        if arm_spec.gripper is None or arm_spec.gripper.joint_names != (self.control_profile.gripper_joint,):
            raise ValueError("contract gripper does not match the Stretch control model")
        if self.control_profile.gripper_joint != "joint_gripper_slide":
            raise ValueError("unexpected Stretch gripper joint")
        if arm_spec.gripper.command_limits is None:
            raise ValueError("Stretch gripper requires command limits")

        steps = round(self.control_dt / self.model.opt.timestep)
        self.substeps = max(1, int(steps))
        if abs(self.substeps * self.model.opt.timestep - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of the model timestep")

        self._arm_low = np.asarray([joint.position_limit[0] for joint in self._joints], dtype=np.float64)
        self._arm_high = np.asarray([joint.position_limit[1] for joint in self._joints], dtype=np.float64)
        self._torque_limits = np.asarray([joint.torque_limit for joint in self._joints], dtype=np.float64)
        self._kp = np.asarray([float(joint.kp_eff or 0.0) for joint in self._joints], dtype=np.float64)
        self._kd = np.asarray([float(joint.kd_eff or 0.0) for joint in self._joints], dtype=np.float64)
        self._joint_step = (
            np.asarray([joint.velocity_limit for joint in self._joints], dtype=np.float64) * self.control_dt
        )
        self._base_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        self._base_step = (
            np.asarray(
                [self.control_profile.base_acceleration, self.control_profile.base_yaw_acceleration],
                dtype=np.float64,
            )
            * self.control_dt
        )
        self._grip_limits = tuple(float(v) for v in arm_spec.gripper.command_limits[0])
        self._grip_open = float(self.control_profile.gripper_open)
        self._grip_speed = float(self.control_profile.gripper_speed)
        self._grip_q = int(self.control.grip_q)
        self._grip_v = int(self.control.grip_v)

        self._joint_command = np.zeros(len(self._joints))
        self._base_command = np.zeros(2)
        self._grip_command = 0.0
        self._target_q: np.ndarray | None = None
        self._target_ee: tuple[float, ...] | None = None
        self._target_ee_local: tuple[np.ndarray, Rotation] | None = None
        self._target_base: tuple[float, float, float] | None = None
        self._teacher_phase = "arm"
        self._elapsed = 0.0

    # ------------------------------------------------------------------
    # WbcBackend protocol
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_arm_q(_READY_Q)
        self.data.qpos[self._grip_q] = self._grip_open
        mujoco.mj_forward(self.model, self.data)
        self.control.reset()

        self._joint_command = self.control.positions().copy()
        self._base_command = np.zeros(2)
        self._grip_command = float(self.data.qpos[self._grip_q])

        self._target_base = self._sample_base_target()
        self._target_q = self.control.positions().copy()
        self._target_ee: tuple[float, ...] | None = None
        self._target_ee_local: tuple[np.ndarray, Rotation] | None = None
        if self.rng.random() < self.arm_goal_probability:
            self._target_q, target_ee = self._sample_arm_target()
            self._target_ee_local = self._pose_in_base_frame(target_ee, self.control.base_pose())
            anchor = self._target_base if self._target_base is not None else self.control.base_pose()
            self._target_ee = self._pose_from_base_frame(self._target_ee_local, anchor)
        self._teacher_phase = "base" if self._target_base is not None else "arm"
        self._elapsed = 0.0
        # Refresh actuator forces so the first observation matches the reset
        # controller command rather than the pre-reset zero ctrl.
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if not np.isfinite(float(dt)) or abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")
        names = self.profile.controlled_joint_names
        unknown = set(command.joint_positions) - set(names)
        if unknown:
            raise ValueError(f"joint position command references unknown joints: {sorted(unknown)}")
        target = self._joint_command.copy()
        for index, name in enumerate(names):
            if name not in command.joint_positions:
                continue
            value = float(command.joint_positions[name])
            if not np.isfinite(value):
                raise ValueError(f"{name} position target is not finite")
            target[index] = value
        target = np.clip(target, self._arm_low, self._arm_high)
        # Executor-side slew: a policy jump becomes a bounded actuator move.
        self._joint_command += np.clip(target - self._joint_command, -self._joint_step, self._joint_step)
        self._joint_command = np.clip(self._joint_command, self._arm_low, self._arm_high)
        self.control.target = self._joint_command.copy()

        base = np.asarray(command.base_velocity, dtype=np.float64).reshape(-1)
        if base.shape != (2,) or not np.isfinite(base).all():
            raise ValueError("StretchWbcBackend requires a planar two-value base velocity")
        desired_base = np.clip(base, -self._base_limits, self._base_limits)
        self._base_command += np.clip(desired_base - self._base_command, -self._base_step, self._base_step)
        self.control.base_command = self._base_command.copy()

        arm_name = self.profile.arms[0].name
        unknown_grippers = set(command.grippers) - {arm_name}
        if unknown_grippers:
            raise ValueError(f"gripper command references unknown arm: {sorted(unknown_grippers)}")
        grip_command = command.grippers.get(arm_name)
        if grip_command is not None:
            if grip_command.mode != "position":
                raise ValueError("StretchWbcBackend only accepts position gripper commands")
            values = np.asarray(grip_command.values, dtype=np.float64).reshape(-1)
            if values.shape != (1,) or not np.isfinite(values).all():
                raise ValueError("Stretch gripper command must contain one finite value")
            desired_grip = float(np.clip(values[0], *self._grip_limits))
            grip_step = self._grip_speed * self.control_dt
            self._grip_command += float(np.clip(desired_grip - self._grip_command, -grip_step, grip_step))
        self._grip_command = float(np.clip(self._grip_command, *self._grip_limits))
        self.control.grip_target = self._grip_command

        terminated = False
        invalid = False
        for _ in range(self.substeps):
            self.control.apply()
            mujoco.mj_step(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                invalid = terminated = True
                break
        if not invalid:
            mujoco.mj_forward(self.model, self.data)
        contact = self._contact_sensor.read(self.data)
        self._elapsed += self.control_dt
        info = {
            "contact_force": contact.collision_force,
            "collision": contact.self_collision,
            "collision_measured": True,
            "hand_contact_force": contact.hand_force,
            "invalid_physics": invalid,
            "elapsed_s": self._elapsed,
            "sim_time": float(self.data.time),
            "ncon": int(self.data.ncon),
            "model_fingerprint": self.model_fingerprint,
        }
        return self._observation(), terminated, info

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Privileged teacher
    # ------------------------------------------------------------------
    def teacher_action(self) -> np.ndarray:
        """Normalized BC action: staged base P servo, then arm PD tracking.

        The base stage moves a sampled forward goal while the arm is damped;
        once the base settles, the arm drives to the sampled five-axis target.
        The gripper is always commanded open. The function is deterministic and
        does not advance physics or consume the RNG.
        """
        if self._target_q is None:
            raise ValueError("teacher action requires a reset with a sampled target")
        arm_spec = self.profile.arms[0]
        assert arm_spec.gripper is not None
        base_stage = self._target_base is not None and self._teacher_phase == "base"
        if base_stage and (self._base_goal_reached() or self._elapsed >= _BASE_STAGE_TIMEOUT_S):
            self._teacher_phase = "arm"
            base_stage = False
            if self._target_ee_local is not None:
                # Re-anchor the reach goal to the measured chassis pose. The
                # base goal is world-fixed, but millimeter-level wheel slip
                # would otherwise consume the arm's reach tolerance.
                self._target_ee = self._pose_from_base_frame(self._target_ee_local, self.control.base_pose())
        base_velocity = self._base_teacher_velocity() if base_stage else np.zeros(2)
        if not base_stage and self._target_ee is not None:
            q = self.control.positions()
            qd = self.control.velocities()
            torque = np.clip(self._kp * (self._target_q - q) - self._kd * qd, -self._torque_limits, self._torque_limits)
        else:
            torque = np.clip(-self._kd * self.control.velocities(), -self._torque_limits, self._torque_limits)
        action = WbcAction(
            base_velocity=(float(base_velocity[0]), float(base_velocity[1])),
            arms=(
                ArmAction(
                    name=arm_spec.name,
                    joint_torques=tuple(float(v) for v in torque),
                    gripper=GripperCommand(values=(self._grip_open,), mode=arm_spec.gripper.command_mode),
                ),
            ),
        )
        return np.clip(normalized_action_vector(action, self.profile), -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _set_arm_q(self, q: np.ndarray) -> None:
        for group, value in zip(self.control.q, q, strict=True):
            self.data.qpos[group] = float(value) / len(group)

    def _sample_arm_target(self) -> tuple[np.ndarray, tuple[float, ...]]:
        q0 = self.control.positions().copy()
        ee0 = self.data.xpos[self.control.ee].copy()
        target_q = q0.copy()
        target_ee = _pose7(self.control.ee_pose())
        try:
            for _ in range(_TARGET_SAMPLE_ATTEMPTS):
                q = np.clip(
                    q0 + self.rng.uniform(-self._target_scale, self._target_scale),
                    self._arm_low,
                    self._arm_high,
                )
                self._set_arm_q(q)
                mujoco.mj_forward(self.model, self.data)
                if (
                    np.linalg.norm(self.data.xpos[self.control.ee] - ee0) >= _TARGET_MIN_EE_DISPLACEMENT_M
                    and not self._posture_blocked()
                ):
                    target_q = q
                    target_ee = _pose7(self.control.ee_pose())
                    break
        finally:
            self._set_arm_q(q0)
            mujoco.mj_forward(self.model, self.data)
            self.control.reset()
        self._joint_command = q0.copy()
        self._grip_command = float(self.data.qpos[self._grip_q])
        return target_q, target_ee

    def _posture_blocked(self) -> bool:
        """True when a sampled arm posture contacts the chassis or scene.

        The packaged scene keeps table/object contacts at rest. A target reach
        that starts against the chassis or table cannot be tracked by the PD
        teacher, so it is resampled instead of producing an impossible label.
        """
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.body(int(self.model.geom_bodyid[contact.geom1])).name
            second = self.model.body(int(self.model.geom_bodyid[contact.geom2])).name
            if first == "world" or second == "world":
                continue
            if first in _NON_ARM_ROBOT_BODIES and second in _NON_ARM_ROBOT_BODIES:
                continue
            if first.startswith(_ENVIRONMENT_PREFIXES) and second.startswith(_ENVIRONMENT_PREFIXES):
                continue
            return True
        return False

    @staticmethod
    def _pose_in_base_frame(pose: tuple[float, ...], base: tuple[float, float, float]) -> tuple[np.ndarray, Rotation]:
        """Convert a world pose to planar chassis-relative position/orientation."""
        rotation = Rotation.from_euler("z", -base[2])
        position = rotation.apply(np.asarray(pose[:3], dtype=np.float64) - np.array([base[0], base[1], 0.0]))
        return position, rotation * Rotation.from_quat(pose[3:])

    @staticmethod
    def _pose_from_base_frame(
        local: tuple[np.ndarray, Rotation], base: tuple[float, float, float]
    ) -> tuple[float, ...]:
        """Convert a chassis-relative pose back to world coordinates."""
        position, orientation = local
        rotation = Rotation.from_euler("z", base[2])
        world_position = rotation.apply(position) + np.array([base[0], base[1], 0.0])
        return tuple(float(value) for value in np.r_[world_position, (rotation * orientation).as_quat()])

    def _sample_base_target(self) -> tuple[float, float, float] | None:
        if self.rng.random() >= self.base_goal_probability:
            return None
        base = self.control.base_pose()
        if self.base_goal_mode == "forward":
            distance = float(self.rng.uniform(*_BASE_GOAL_DISTANCE_M))
            bearing = float(self.rng.uniform(-_BASE_GOAL_BEARING_RAD, _BASE_GOAL_BEARING_RAD))
            heading = base[2] + bearing
            target_x = base[0] + distance * np.cos(heading)
            target_y = base[1] + distance * np.sin(heading)
            target_yaw = base[2] + self.rng.uniform(-_BASE_GOAL_YAW_RAD, _BASE_GOAL_YAW_RAD)
        else:
            target_x = base[0] + self.rng.uniform(-0.15, 0.15)
            target_y = base[1] + self.rng.uniform(-0.15, 0.15)
            target_yaw = base[2] + self.rng.uniform(-0.3, 0.3)
        return (float(target_x), float(target_y), float(target_yaw))

    def _base_velocity(self) -> np.ndarray:
        """Measured forward speed and world yaw rate of the base body."""
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self.control.base,
            spatial,
            0,
        )
        forward = self.data.xmat[self.control.base].reshape(3, 3)[:, 0]
        return np.array([float(spatial[3:] @ forward), float(spatial[2])])

    def _base_goal_reached(self) -> bool:
        if self._target_base is None:
            return True
        base = self.control.base_pose()
        distance = float(np.hypot(self._target_base[0] - base[0], self._target_base[1] - base[1]))
        heading = abs(_wrap_angle(self._target_base[2] - base[2]))
        velocity = self._base_velocity()
        return bool(
            distance <= _BASE_SETTLE_DISTANCE_M
            and heading <= _BASE_SETTLE_HEADING_RAD
            and abs(float(velocity[0])) <= _BASE_SETTLE_SPEED_M_S
            and abs(float(velocity[1])) <= _BASE_SETTLE_YAW_RATE_RAD_S
        )

    def _base_teacher_velocity(self) -> np.ndarray:
        if self._target_base is None:
            return np.zeros(2)
        base = self.control.base_pose()
        dx = self._target_base[0] - base[0]
        dy = self._target_base[1] - base[1]
        cos_yaw, sin_yaw = np.cos(base[2]), np.sin(base[2])
        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw
        heading_error = _wrap_angle(self._target_base[2] - base[2])
        vx = float(np.clip(_BASE_POSITION_GAIN * forward, -self._base_limits[0], self._base_limits[0]))
        wz = float(
            np.clip(
                _BASE_YAW_GAIN * heading_error + _BASE_LATERAL_GAIN * lateral,
                -self._base_limits[1],
                self._base_limits[1],
            )
        )
        return np.array([vx, wz])

    def _observation(self) -> WbcObservation:
        arm_spec = self.profile.arms[0]
        q = self.control.positions().copy()
        qd = self.control.velocities().copy()
        torque = np.asarray(
            [float(np.sum(self.data.qfrc_actuator[group])) for group in self.control.v],
            dtype=np.float64,
        )
        arm = ArmObservation(
            name=arm_spec.name,
            joint_position=tuple(float(v) for v in q),
            joint_velocity=tuple(float(v) for v in qd),
            joint_torque=tuple(float(v) for v in torque),
            ee_pose=_pose7(self.control.ee_pose()),
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
            base_twist=tuple(float(v) for v in self._base_velocity()),
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )


__all__ = ["StretchWbcBackend"]
