# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo PandaOmron plant composed from Robosuite assets.

RoboCasa composes Omron LD-60 + one 7-DOF Panda arm + a two-finger gripper.
The arm and torso are torque motors in the composed model, so this backend uses
the contract's true-torque adapter; the base is the Omron holonomic joint set
and the gripper is position-controlled.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import ServoCommand, TorqueAdapter
from emet.controller.manipulation.wbc_contract import (
    ArmAction,
    ArmObservation,
    EmbodimentProfile,
    GripperCommand,
    GripperObservation,
    JointGroupObservation,
    WbcAction,
    WbcObservation,
    normalized_action_vector,
)
from emet.controller.manipulation.wbc_profiles import (
    PANDAOMRON_EE_LINK_ALIASES,
    PANDAOMRON_GRIPPER_JOINT_ALIASES,
    pandaomron_sim_profile,
    resolve_body_name,
    resolve_joint_names,
)
from emet.rl.manipulation.wbc_contact import ContactReading, MuJoCoContactSensor
from emet.rl.manipulation.wbc_env import WbcPerturbation

_BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
)
_BASE_ACTUATORS = (
    "mobilebase0_actuator_mobile_forward",
    "mobilebase0_actuator_mobile_side",
    "mobilebase0_actuator_mobile_yaw",
)
_TORSO_JOINT = "mobilebase0_joint_torso_height"
_TORSO_ACTUATOR = "mobilebase0_actuator_torso_height"
_ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
_ARM_ACTUATORS = tuple(f"robot0_torq_j{i}" for i in range(1, 8))
_GRIPPER_JOINTS = ("gripper0_finger_joint1", "gripper0_finger_joint2")
_GRIPPER_ACTUATORS = ("gripper0_gripper_finger_joint1", "gripper0_gripper_finger_joint2")
_BASE_BODY = "mobilebase0_base"
_EE_BODY = "robot0_right_hand"
# The composed Panda's zero pose folds the wrist into link5. Robosuite's
# Panda initial posture is collision-free and keeps the EE in front of the base.
_READY_Q = np.array([0.0, -0.00365046, 0.0, -2.61799388, 0.0, 2.74159265, 0.78539816])
_READY_TORSO = 0.2
_TARGET_MIN_EE_DISPLACEMENT_M = 0.03


def _wrap(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _pose7(position: np.ndarray, rotation: np.ndarray) -> tuple[float, ...]:
    quat = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat()
    return tuple(float(v) for v in np.r_[position, quat])


class PandaOmronWbcBackend:
    """Direct-MuJoCo PandaOmron plant for the contract."""

    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        seed: int = 0,
        control_dt: float = 0.05,
        target_range: float = 0.15,
        arm_goal_probability: float = 1.0,
        base_goal_probability: float = 0.5,
        base_goal_mode: str = "forward",
        torso_range: tuple[float, float] | None = None,
    ) -> None:
        self.profile = profile or pandaomron_sim_profile()
        if self.profile.meta("robot") != "pandaomron":
            raise ValueError("PandaOmronWbcBackend requires a pandaomron profile")
        if self.profile.base.kind != "holonomic" or self.profile.base.action_dim != 3:
            raise ValueError("PandaOmronWbcBackend requires a holonomic three-axis base")
        if tuple(arm.name for arm in self.profile.arms) != ("right",):
            raise ValueError("PandaOmronWbcBackend requires one right arm")
        if self.profile.torso is None:
            raise ValueError("PandaOmronWbcBackend requires a torso group")
        if any(arm.target_frame != "world" for arm in self.profile.arms):
            raise ValueError("PandaOmronWbcBackend emits world-frame EE poses")
        if not np.isfinite([control_dt, target_range, arm_goal_probability, base_goal_probability]).all():
            raise ValueError("backend timing/probability values must be finite")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if arm_goal_probability > 0 and target_range <= 0:
            raise ValueError("target_range must be positive when arm goals are enabled")
        if not 0 <= arm_goal_probability <= 1 or not 0 <= base_goal_probability <= 1:
            raise ValueError("goal probabilities must be in [0, 1]")
        if base_goal_mode not in {"forward", "any"}:
            raise ValueError("base_goal_mode must be forward or any")

        self.control_dt = float(control_dt)
        self.target_range = float(target_range)
        self.arm_goal_probability = float(arm_goal_probability)
        self.base_goal_probability = float(base_goal_probability)
        self.base_goal_mode = base_goal_mode
        self.torso_range = None if torso_range is None else (float(torso_range[0]), float(torso_range[1]))
        if self.torso_range is not None:
            low, high = self.profile.torso.joints[0].position_limit
            if not np.isfinite(self.torso_range).all() or not low <= self.torso_range[0] <= self.torso_range[1] <= high:
                raise ValueError(f"torso_range must be finite and inside [{low}, {high}]")
        self.rng = np.random.default_rng(seed)

        self.model, self.model_fingerprint = _build_model()
        self.data = mujoco.MjData(self.model)
        gripper = self.profile.arms[0].gripper
        assert gripper is not None
        if resolve_body_name(self.model, self.profile.arms[0].ee_link, PANDAOMRON_EE_LINK_ALIASES) != _EE_BODY:
            raise ValueError("direct Panda EE aliases do not match the composed model")
        if resolve_joint_names(self.model, gripper.joint_names, PANDAOMRON_GRIPPER_JOINT_ALIASES) != _GRIPPER_JOINTS:
            raise ValueError("direct Panda gripper aliases do not match the composed model")
        self._contact_sensor = MuJoCoContactSensor(
            self.model,
            base_body="robot0_base",
            ee_body=_EE_BODY,
            finger_bodies=("gripper0_leftfinger", "gripper0_rightfinger"),
            force_sensor="gripper0_force_ee",
            torque_sensor="gripper0_torque_ee",
        )
        self.substeps = int(round(self.control_dt / self.model.opt.timestep))
        if self.substeps < 1 or abs(self.substeps * self.model.opt.timestep - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer number of model timesteps")
        self.preferred_adapter = TorqueAdapter(self.profile)

        self._base_q = np.asarray([self.model.joint(name).qposadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_v = np.asarray([self.model.joint(name).dofadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_act = np.asarray([self.model.actuator(name).id for name in _BASE_ACTUATORS], dtype=int)
        self._torso_q = int(self.model.joint(_TORSO_JOINT).qposadr[0])
        self._torso_v = int(self.model.joint(_TORSO_JOINT).dofadr[0])
        self._torso_act = int(self.model.actuator(_TORSO_ACTUATOR).id)
        self._arm_q = np.asarray([self.model.joint(name).qposadr[0] for name in _ARM_JOINTS], dtype=int)
        self._arm_v = np.asarray([self.model.joint(name).dofadr[0] for name in _ARM_JOINTS], dtype=int)
        self._arm_act = np.asarray([self.model.actuator(name).id for name in _ARM_ACTUATORS], dtype=int)
        self._grip_q = np.asarray([self.model.joint(name).qposadr[0] for name in _GRIPPER_JOINTS], dtype=int)
        self._grip_v = np.asarray([self.model.joint(name).dofadr[0] for name in _GRIPPER_JOINTS], dtype=int)
        self._grip_act = np.asarray([self.model.actuator(name).id for name in _GRIPPER_ACTUATORS], dtype=int)
        self._base_body = self.model.body(_BASE_BODY).id
        self._ee_body = self.model.body(_EE_BODY).id
        self._payload_body = int(self.model.body("wbc_payload").id)
        self._perturbation: WbcPerturbation | None = None
        self._teacher_gripper = (0.04, -0.04)
        self._arm_low = np.asarray(
            [self.profile.joint_group("panda_arm").joints[i].position_limit[0] for i in range(7)]
        )
        self._arm_high = np.asarray(
            [self.profile.joint_group("panda_arm").joints[i].position_limit[1] for i in range(7)]
        )
        self._torque_limits = np.asarray([joint.torque_limit for joint in self.profile.joint_group("panda_arm").joints])
        self._kp = np.asarray([joint.kp_eff or 0.0 for joint in self.profile.joint_group("panda_arm").joints])
        self._kd = np.asarray([joint.kd_eff or 0.0 for joint in self.profile.joint_group("panda_arm").joints])
        self._base_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        self._base_step = np.array([0.4, 0.4, 0.8]) * self.control_dt
        self._arm_step = (
            np.asarray([j.velocity_limit for j in self.profile.joint_group("panda_arm").joints]) * self.control_dt
        )
        self._grip_limits = self.profile.arms[0].gripper.command_limits
        self._grip_open = (0.04, -0.04)
        self._joint_command = np.zeros(7)
        self._base_command = np.zeros(3)
        self._grip_command = np.asarray((0.04, -0.04))
        self._target_q: np.ndarray | None = None
        self._target_torso = 0.0
        self._target_ee: tuple[float, ...] | None = None
        self._target_ee_local: tuple[np.ndarray, Rotation] | None = None
        self._target_base: tuple[float, float, float] | None = None
        self._teacher_phase = "base"
        self._elapsed = 0.0

    def set_perturbation(self, perturbation: WbcPerturbation | None) -> None:
        self._perturbation = perturbation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        payload_mass = 0.0 if self._perturbation is None else float(self._perturbation.payload_mass)
        self.model.body_mass[self._payload_body] = max(payload_mass, 1e-9)
        payload_radius = 0.03
        self.model.body_inertia[self._payload_body] = 0.4 * payload_mass * payload_radius**2
        self._teacher_gripper = (
            (0.0, 0.0) if self._perturbation is not None and self._perturbation.gripper_close else self._grip_open
        )
        if self.torso_range is not None:
            self._target_torso = float(self.rng.uniform(*self.torso_range))
        else:
            self._target_torso = _READY_TORSO
        self.data.qpos[self._base_q] = 0.0
        self.data.qpos[self._torso_q] = self._target_torso
        self.data.qpos[self._arm_q] = _READY_Q
        self.data.qpos[self._grip_q] = self._grip_open
        mujoco.mj_forward(self.model, self.data)
        q0 = _READY_Q.copy()
        self._joint_command = q0.copy()
        self._base_command[:] = 0.0
        self._grip_command = np.asarray(self._grip_open)
        self._target_torso = float(self.data.qpos[self._torso_q])
        self._target_base = self._sample_base_target()
        self._target_q = q0.copy()
        self._target_ee = None
        self._target_ee_local = None
        if self.rng.random() < self.arm_goal_probability:
            self._target_q, target_ee = self._sample_arm_target(self._target_base)
            anchor = self._target_base if self._target_base is not None else self._base_pose()
            self._target_ee_local = self._pose_in_base_frame(target_ee, anchor)
            self._target_ee = target_ee
        self._teacher_phase = "base" if self._target_base is not None else "arm"
        self._elapsed = 0.0
        return self._observation()

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if not np.isfinite(float(dt)) or abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")
        if command.joint_positions:
            raise ValueError("PandaOmronWbcBackend uses the torque adapter; joint_positions must be empty")
        torques = command.commanded_torques
        for index, name in enumerate(("joint_torso_height", *[f"joint{i}" for i in range(1, 8)])):
            value = float(torques.get(name, 0.0))
            limit = 2000.0 if name == "joint_torso_height" else float(self._torque_limits[index - 1])
            if not np.isfinite(value):
                raise ValueError(f"{name} torque is not finite")
            if name == "joint_torso_height":
                self.data.ctrl[self._torso_act] = float(np.clip(value, -limit, limit))
            else:
                self.data.ctrl[self._arm_act[index - 1]] = float(np.clip(value, -limit, limit))
        desired_base = np.clip(
            np.asarray(command.base_velocity, dtype=np.float64), -self._base_limits, self._base_limits
        )
        self._base_command += np.clip(desired_base - self._base_command, -self._base_step, self._base_step)
        self.data.ctrl[self._base_act] = self._base_command
        if "right" in command.grippers:
            values = np.asarray(command.grippers["right"].values, dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("PandaOmron gripper command must contain two finite values")
            desired = np.asarray([np.clip(values[0], *self._grip_limits[0]), np.clip(values[1], *self._grip_limits[1])])
            self._grip_command += np.clip(desired - self._grip_command, -0.05, 0.05)
            self.data.ctrl[self._grip_act] = self._grip_command
        terminated = False
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                terminated = True
                break
        mujoco.mj_forward(self.model, self.data)
        self._elapsed += self.control_dt
        contact = self._contact_sensor.read(self.data)
        return (
            self._observation(contact=contact),
            terminated,
            {
                **contact.to_info(),
                "collision_measured": True,
                "contact_state_available": True,
                "wheel_lift": False,
                "invalid_physics": terminated,
                "elapsed_s": self._elapsed,
                "model_fingerprint": self.model_fingerprint,
            },
        )

    def teacher_action(self) -> np.ndarray:
        if self._target_q is None:
            raise ValueError("teacher action requires a sampled target")
        q = self.data.qpos[self._arm_q]
        qd = self.data.qvel[self._arm_v]
        base = self._base_pose()
        base_distance = (
            0.0
            if self._target_base is None
            else float(np.hypot(self._target_base[0] - base[0], self._target_base[1] - base[1]))
        )
        base_heading = 0.0 if self._target_base is None else abs(_wrap(self._target_base[2] - base[2]))
        base_command = np.zeros(3)
        if self._target_base is not None and self._teacher_phase == "base":
            if base_distance <= 0.03 and base_heading <= 0.05:
                self._teacher_phase = "arm"
                if self._target_ee_local is not None:
                    # Re-anchor the reach target to the measured base pose so
                    # the base's settle residual does not consume the arm's
                    # 2 cm reach tolerance.
                    self._target_ee = self._pose_from_base_frame(self._target_ee_local, base)
            else:
                dx = self._target_base[0] - base[0]
                dy = self._target_base[1] - base[1]
                yaw_error = _wrap(self._target_base[2] - base[2])
                forward = dx * np.cos(base[2]) + dy * np.sin(base[2])
                lateral = -dx * np.sin(base[2]) + dy * np.cos(base[2])

                def drive(error: float, limit: float) -> float:
                    if abs(error) <= 0.02:
                        return 0.0
                    return float(np.sign(error) * min(limit, max(0.3, 4.0 * abs(error))))

                vx = drive(float(forward), self._base_limits[0])
                vy = drive(float(lateral), self._base_limits[1])
                if abs(yaw_error) <= 0.03:
                    wz = 0.0
                else:
                    wz = float(np.sign(yaw_error) * min(self._base_limits[2], max(0.3, 2.0 * abs(yaw_error))))
                base_command = np.array([vx, vy, wz])
        else:
            base_command = np.zeros(3)
        bias = np.asarray(self.data.qfrc_bias[self._arm_v], dtype=np.float64)
        if self._teacher_phase == "arm" and self._target_ee is not None:
            torque = np.clip(
                self._kp * (self._target_q - q) - self._kd * qd + bias,
                -self._torque_limits,
                self._torque_limits,
            )
        else:
            torque = np.clip(-self._kd * qd + bias, -self._torque_limits, self._torque_limits)
        torso = float(self.data.qpos[self._torso_q])
        torso_qd = float(self.data.qvel[self._torso_v])
        torso_bias = float(self.data.qfrc_bias[self._torso_v])
        torso_torque = float(
            np.clip(1000.0 * (self._target_torso - torso) - 200.0 * torso_qd + torso_bias, -2000.0, 2000.0)
        )
        action = WbcAction(
            base_velocity=tuple(float(v) for v in base_command),
            torso_torques=(torso_torque,),
            arms=(
                ArmAction(
                    name="right",
                    joint_torques=tuple(float(v) for v in torque),
                    gripper=GripperCommand(values=self._teacher_gripper),
                ),
            ),
        )
        return np.clip(normalized_action_vector(action, self.profile), -1.0, 1.0).astype(np.float32)

    def load_hard_case(self, case) -> WbcObservation | None:
        """Reproduce a recorded Panda failure posture and target posture."""
        robot = str(getattr(case, "robot", "") or "")
        if robot not in ("", "panda_omron", "pandaomron"):
            return None
        initial = np.asarray(getattr(case, "initial_q", ()), dtype=np.float64).reshape(-1)
        target = np.asarray(getattr(case, "target_q", ()), dtype=np.float64).reshape(-1)
        if initial.shape != (7,) or target.shape != (7,):
            return None
        if not (np.isfinite(initial).all() and np.isfinite(target).all()):
            return None
        initial = np.clip(initial, self._arm_low, self._arm_high)
        target = np.clip(target, self._arm_low, self._arm_high)
        self.data.qpos[self._arm_q] = initial
        self.data.qvel[self._arm_v] = 0.0
        self._joint_command = initial.copy()
        self._target_q = target.copy()
        self._target_base = None
        self._target_ee_local = None
        self._teacher_phase = "arm"
        self._target_torso = float(self.data.qpos[self._torso_q])
        self.data.qpos[self._arm_q] = target
        mujoco.mj_forward(self.model, self.data)
        self._target_ee = _pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body])
        self.data.qpos[self._arm_q] = initial
        self.data.qvel[self._arm_v] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._elapsed = 0.0
        return self._observation()

    def close(self) -> None:
        return None

    def _base_pose(self) -> tuple[float, float, float]:
        rotation = self.data.xmat[self._base_body].reshape(3, 3)
        return (
            float(self.data.xpos[self._base_body, 0]),
            float(self.data.xpos[self._base_body, 1]),
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        )

    @staticmethod
    def _pose_in_base_frame(pose: tuple[float, ...], base: tuple[float, float, float]) -> tuple[np.ndarray, Rotation]:
        rotation = Rotation.from_euler("z", -base[2])
        position = rotation.apply(np.asarray(pose[:3], dtype=np.float64) - np.array([base[0], base[1], 0.0]))
        return position, rotation * Rotation.from_quat(pose[3:])

    @staticmethod
    def _pose_from_base_frame(
        local: tuple[np.ndarray, Rotation], base: tuple[float, float, float]
    ) -> tuple[float, ...]:
        position, orientation = local
        rotation = Rotation.from_euler("z", base[2])
        world_position = rotation.apply(position) + np.array([base[0], base[1], 0.0])
        return tuple(float(value) for value in np.r_[world_position, (rotation * orientation).as_quat()])

    def _posture_blocked(self) -> bool:
        """True when a sampled arm posture self-collides outside the gripper."""
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            first = self.model.body(int(self.model.geom_bodyid[contact.geom1])).name
            second = self.model.body(int(self.model.geom_bodyid[contact.geom2])).name
            if first == "world" or second == "world":
                continue
            if first.startswith("gripper0_") and second.startswith("gripper0_"):
                continue
            return True
        return False

    def _sample_arm_target(
        self, base_target: tuple[float, float, float] | None = None
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        saved_base = self.data.qpos[self._base_q].copy()
        if base_target is not None:
            self.data.qpos[self._base_q] = base_target
            mujoco.mj_forward(self.model, self.data)
        q0 = self.data.qpos[self._arm_q].copy()
        ee0 = self.data.xpos[self._ee_body].copy()
        try:
            for _ in range(64):
                q = np.clip(
                    q0 + self.rng.uniform(-self.target_range, self.target_range, 7),
                    self._arm_low,
                    self._arm_high,
                )
                self.data.qpos[self._arm_q] = q
                mujoco.mj_forward(self.model, self.data)
                if (
                    np.linalg.norm(self.data.xpos[self._ee_body] - ee0) >= _TARGET_MIN_EE_DISPLACEMENT_M
                    and not self._posture_blocked()
                ):
                    pose = _pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body])
                    return q.copy(), pose
            self.data.qpos[self._arm_q] = q0
            mujoco.mj_forward(self.model, self.data)
            return q0.copy(), _pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body])
        finally:
            self.data.qpos[self._base_q] = saved_base
            self.data.qpos[self._arm_q] = q0
            mujoco.mj_forward(self.model, self.data)

    def _sample_base_target(self) -> tuple[float, float, float] | None:
        if self.rng.random() >= self.base_goal_probability:
            return None
        x, y, yaw = self._base_pose()
        if self.base_goal_mode == "forward":
            distance = float(self.rng.uniform(0.05, 0.15))
            bearing = float(self.rng.uniform(-0.25, 0.25))
            target_x = x + distance * np.cos(yaw + bearing)
            target_y = y + distance * np.sin(yaw + bearing)
            target_yaw = yaw + self.rng.uniform(-0.1, 0.1)
        else:
            target_x = x + self.rng.uniform(-0.15, 0.15)
            target_y = y + self.rng.uniform(-0.15, 0.15)
            target_yaw = yaw + self.rng.uniform(-0.3, 0.3)
        return (float(target_x), float(target_y), float(target_yaw))

    def _observation(self, *, contact: ContactReading | None = None) -> WbcObservation:
        if contact is None:
            contact = self._contact_sensor.read(self.data)
        base_pose = self._base_pose()
        base_twist = tuple(float(v) for v in self.data.qvel[self._base_v])
        torso = JointGroupObservation(
            name="torso",
            position=(float(self.data.qpos[self._torso_q]),),
            velocity=(float(self.data.qvel[self._torso_v]),),
            torque=(float(self.data.qfrc_actuator[self._torso_v]),),
            target=(float(self._target_torso),),
        )
        arm = ArmObservation(
            name="right",
            joint_position=tuple(float(v) for v in self.data.qpos[self._arm_q]),
            joint_velocity=tuple(float(v) for v in self.data.qvel[self._arm_v]),
            joint_torque=tuple(float(v) for v in self.data.qfrc_actuator[self._arm_v]),
            ee_pose=_pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body]),
            ee_target=self._target_ee,
            target_mask=(True, True),
            gripper=GripperObservation(
                position=tuple(float(v) for v in self.data.qpos[self._grip_q]),
                velocity=tuple(float(v) for v in self.data.qvel[self._grip_v]),
                force=contact.finger_force,
                target=tuple(float(v) for v in self._grip_command),
            ),
            contact=contact.contact,
        )
        return WbcObservation(
            base_pose=base_pose,
            base_twist=base_twist,
            torso=torso,
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )


def _build_model() -> tuple[mujoco.MjModel, str]:
    from robosuite.models.bases import OmronMobileBase
    from robosuite.models.grippers import PandaGripper
    from robosuite.models.robots import PandaOmron

    robot = PandaOmron()
    robot.add_base(OmronMobileBase())
    robot.add_gripper(PandaGripper())
    root = ET.fromstring(robot.get_xml())
    world = root.find("worldbody")
    if world is None:
        raise ValueError("PandaOmron XML has no worldbody")
    if not any(child.tag == "geom" and child.get("type") == "plane" for child in world):
        # Visual only: the Omron base has no wheels in the composed model, so
        # a colliding floor would pin the planar base joints in place.
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="2 2 0.1",
            rgba="0.8 0.8 0.8 1",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(world, "light", pos="0 0 3", dir="0 0 -1")
    hand = root.find(".//body[@name='gripper0_right_gripper']")
    if hand is None:
        raise ValueError("PandaOmron XML has no gripper0_right_gripper body")
    payload = ET.SubElement(hand, "body", name="wbc_payload", pos="0 0 -0.06")
    ET.SubElement(
        payload,
        "geom",
        name="wbc_payload_geom",
        type="sphere",
        size="0.03",
        mass="0.000000001",
        contype="0",
        conaffinity="0",
        rgba="0.2 0.8 0.2 1",
    )
    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml), hashlib.sha256(xml.encode()).hexdigest()


__all__ = ["PandaOmronWbcBackend"]
