# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Run the PandaOmron WBC checkpoint inside a RoboCasa/Robosuite kitchen env.

RoboCasa builds the same Omron + Panda model as the direct plant, but inside a
kitchen scene and with Robosuite's composite controller. This backend bypasses
that controller and writes the contract torques/velocities directly to the
sim actuators, so the direct-MuJoCo policy can be evaluated in the benchmark
scene. Task targets are supplied by a higher-level skill/TAMP layer; the default
reset target is a small free-space offset from the current end effector.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from emet.controller.manipulation.torque_adapter import ServoCommand, TorqueAdapter
from emet.controller.manipulation.wbc_contract import (
    ArmObservation,
    GripperObservation,
    JointGroupObservation,
    WbcObservation,
)
from emet.controller.manipulation.wbc_profiles import pandaomron_sim_profile
from emet.rl.manipulation.wbc_pandaomron import _READY_Q, _READY_TORSO

_ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
_ARM_ACTUATORS = tuple(f"robot0_torq_j{i}" for i in range(1, 8))
_TORSO_JOINT = "mobilebase0_joint_torso_height"
_TORSO_ACTUATOR = "mobilebase0_actuator_torso_height"
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
_GRIPPER_JOINTS = ("gripper0_right_finger_joint1", "gripper0_right_finger_joint2")
_GRIPPER_ACTUATORS = (
    "gripper0_right_gripper_finger_joint1",
    "gripper0_right_gripper_finger_joint2",
)
_BASE_BODY = "mobilebase0_base"
_EE_BODY = "robot0_right_hand"


def _wrap(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _pose7(position: np.ndarray, rotation: np.ndarray) -> tuple[float, ...]:
    quat = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat()
    return tuple(float(value) for value in np.r_[position, quat])


class RoboCasaPandaBackend:
    """Contract backend for a RoboCasa env using the composed PandaOmron model."""

    def __init__(
        self,
        env: Any,
        *,
        profile=None,
        seed: int = 0,
        control_dt: float = 0.05,
        target_offset: tuple[float, float, float] = (0.04, 0.04, 0.06),
        base_backoff: float = 0.25,
    ) -> None:
        self.env = env
        self.profile = profile or pandaomron_sim_profile()
        if self.profile.meta("robot") != "pandaomron":
            raise ValueError("RoboCasaPandaBackend requires a pandaomron profile")
        if self.profile.base.kind != "holonomic" or self.profile.base.action_dim != 3:
            raise ValueError("RoboCasaPandaBackend requires a holonomic three-axis base")
        self.control_dt = float(control_dt)
        self.target_offset = tuple(float(v) for v in target_offset)
        self.base_backoff = float(base_backoff)
        if self.base_backoff < 0:
            raise ValueError("base_backoff must be nonnegative")
        self.rng = np.random.default_rng(seed)
        self.model = None
        self.data = None
        self.substeps = 0
        self.model_fingerprint = ""
        self.preferred_adapter = TorqueAdapter(self.profile)
        self._target_ee: tuple[float, ...] | None = None
        self._target_ee_override: tuple[float, ...] | None = None
        self._target_base: tuple[float, float, float] | None = None
        self._base_command = np.zeros(3)
        self._grip_command = np.asarray((0.04, -0.04))
        self._elapsed = 0.0
        self._origin_xy = np.zeros(2)
        self._origin_yaw = 0.0

    def _initialize_sim(self) -> None:
        self.model = self.env.sim.model
        self.data = self.env.sim.data
        self.substeps = int(round(self.control_dt / float(self.model.opt.timestep)))
        if self.substeps < 1 or abs(self.substeps * float(self.model.opt.timestep) - self.control_dt) > 1e-9:
            raise ValueError("RoboCasa control_dt must be an integer number of simulation timesteps")
        self.model_fingerprint = hashlib.sha256(self.env.sim.model.get_xml().encode()).hexdigest()[:16]
        self._bind()

    def _bind(self) -> None:
        self._arm_q = np.asarray([self.model.joint(name).qposadr[0] for name in _ARM_JOINTS], dtype=int)
        self._arm_v = np.asarray([self.model.joint(name).dofadr[0] for name in _ARM_JOINTS], dtype=int)
        self._arm_act = np.asarray([self.model.actuator(name).id for name in _ARM_ACTUATORS], dtype=int)
        self._torso_q = int(self.model.joint(_TORSO_JOINT).qposadr[0])
        self._torso_v = int(self.model.joint(_TORSO_JOINT).dofadr[0])
        self._torso_act = int(self.model.actuator(_TORSO_ACTUATOR).id)
        self._base_q = np.asarray([self.model.joint(name).qposadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_v = np.asarray([self.model.joint(name).dofadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_act = np.asarray([self.model.actuator(name).id for name in _BASE_ACTUATORS], dtype=int)
        self._grip_q = np.asarray([self.model.joint(name).qposadr[0] for name in _GRIPPER_JOINTS], dtype=int)
        self._grip_v = np.asarray([self.model.joint(name).dofadr[0] for name in _GRIPPER_JOINTS], dtype=int)
        self._grip_act = np.asarray([self.model.actuator(name).id for name in _GRIPPER_ACTUATORS], dtype=int)
        self._base_body = int(self.model.body(_BASE_BODY).id)
        self._ee_body = int(self.model.body(_EE_BODY).id)
        self._torque_acts = {f"joint{i}": int(self.model.actuator(name).id) for i, name in enumerate(_ARM_ACTUATORS, 1)}
        self._torque_acts["joint_torso_height"] = self._torso_act

    def set_ee_target(self, pose: tuple[float, ...] | None) -> None:
        """Set a world-frame target (7 values) or restore the reset offset."""
        if pose is not None and len(pose) != 7:
            raise ValueError("EE target must be a 7-value position/quaternion pose")
        self._target_ee_override = None if pose is None else tuple(float(v) for v in pose)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
            random.seed(seed)
            if hasattr(self.env, "rng"):
                self.env.rng = np.random.default_rng(seed)
        self.env.reset()
        self._initialize_sim()
        # Back the chassis away from the counter before raising the arm. The
        # task spawn can put the ready posture inside a counter fixture.
        if self.base_backoff > 0:
            # The Omron planar slides are in the chassis frame; move backward
            # along the forward slide, not along the kitchen's world axes.
            self.data.qpos[self._base_q[0]] -= self.base_backoff
        # Match the direct plant's collision-free ready state. RoboCasa's task
        # reset leaves the arm at a different posture and the torso at zero,
        # which is outside the reach policy's training distribution.
        self.data.qpos[self._arm_q] = _READY_Q
        self.data.qvel[self._arm_v] = 0.0
        self.data.qpos[self._torso_q] = _READY_TORSO
        self.data.qvel[self._torso_v] = 0.0
        self.data.qpos[self._grip_q] = (0.04, -0.04)
        self.env.sim.forward()
        self._base_command[:] = 0.0
        self._grip_command = np.asarray((0.04, -0.04))
        self._elapsed = 0.0
        world_base = self._world_base_pose()
        self._origin_xy = np.asarray(world_base[:2], dtype=np.float64)
        self._origin_yaw = float(world_base[2])
        ee = self._local_pose(_pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body]))
        if self._target_ee_override is not None:
            self._target_ee = self._target_ee_override
        else:
            offset = np.asarray(self.target_offset, dtype=np.float64)
            self._target_ee = tuple(float(v) for v in np.r_[np.asarray(ee[:3]) + offset, ee[3:]])
        return self._observation()

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if not np.isfinite(float(dt)) or abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")
        for name, value in command.commanded_torques.items():
            actuator = self._torque_acts.get(name)
            if actuator is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} torque is not finite")
            low, high = self.model.actuator_ctrlrange[actuator]
            self.data.ctrl[actuator] = float(np.clip(value, low, high))
        # Hold the torso at the direct plant's ready height. The arm-only
        # policy's torso channel is not a reliable height servo in the kitchen
        # scene, and the reach contract only needs a stable shoulder frame.
        torso = float(self.data.qpos[self._torso_q])
        torso_qd = float(self.data.qvel[self._torso_v])
        torso_bias = float(self.data.qfrc_bias[self._torso_v])
        torso_torque = float(np.clip(1000.0 * (_READY_TORSO - torso) - 200.0 * torso_qd + torso_bias, -2000.0, 2000.0))
        self.data.ctrl[self._torso_act] = torso_torque
        base = np.asarray(command.base_velocity, dtype=np.float64).reshape(-1)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("RoboCasaPandaBackend requires a finite three-axis base velocity")
        limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        self._base_command = np.clip(base, -limits, limits)
        self.data.ctrl[self._base_act] = self._base_command
        grip = command.grippers.get("right")
        if grip is not None:
            values = np.asarray(grip.values, dtype=np.float64).reshape(-1)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("Panda gripper command must contain two finite values")
            self._grip_command = values.copy()
            for actuator, value in zip(self._grip_act, values, strict=True):
                low, high = self.model.actuator_ctrlrange[actuator]
                self.data.ctrl[actuator] = float(np.clip(value, low, high))
        invalid = False
        for _ in range(self.substeps):
            self.env.sim.step()
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                invalid = True
                break
        self.env.sim.forward()
        self._elapsed += self.control_dt
        return (
            self._observation(),
            invalid,
            {
                "contact_force": 0.0,
                "collision": False,
                "invalid_physics": invalid,
                "elapsed_s": self._elapsed,
                "ncon": int(self.data.ncon),
                "model_fingerprint": self.model_fingerprint,
            },
        )

    def close(self) -> None:
        self.env.close()

    def _world_base_pose(self) -> tuple[float, float, float]:
        rotation = self.data.xmat[self._base_body].reshape(3, 3)
        return (
            float(self.data.xpos[self._base_body, 0]),
            float(self.data.xpos[self._base_body, 1]),
            float(np.arctan2(rotation[1, 0], rotation[0, 0])),
        )

    def _local_pose(self, pose: tuple[float, ...]) -> tuple[float, ...]:
        """Express a world pose in the reset-anchored robot frame."""
        rotation = Rotation.from_euler("z", -self._origin_yaw)
        position = rotation.apply(
            np.asarray(pose[:3], dtype=np.float64) - np.array([self._origin_xy[0], self._origin_xy[1], 0.0])
        )
        orientation = rotation * Rotation.from_quat(pose[3:])
        return tuple(float(value) for value in np.r_[position, orientation.as_quat()])

    def _base_pose(self) -> tuple[float, float, float]:
        world = self._world_base_pose()
        return (
            float(world[0] - self._origin_xy[0]),
            float(world[1] - self._origin_xy[1]),
            float(_wrap(world[2] - self._origin_yaw)),
        )

    def _observation(self) -> WbcObservation:
        torso = JointGroupObservation(
            name="torso",
            position=(float(self.data.qpos[self._torso_q]),),
            velocity=(float(self.data.qvel[self._torso_v]),),
            torque=(float(self.data.qfrc_actuator[self._torso_v]),),
        )
        arm = ArmObservation(
            name="right",
            joint_position=tuple(float(v) for v in self.data.qpos[self._arm_q]),
            joint_velocity=tuple(float(v) for v in self.data.qvel[self._arm_v]),
            joint_torque=tuple(float(v) for v in self.data.qfrc_actuator[self._arm_v]),
            ee_pose=self._local_pose(_pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body])),
            ee_target=self._target_ee,
            target_mask=(True, True),
            gripper=GripperObservation(
                position=tuple(float(v) for v in self.data.qpos[self._grip_q]),
                velocity=tuple(float(v) for v in self.data.qvel[self._grip_v]),
            ),
        )
        base_yaw = self._world_base_pose()[2]
        cos_yaw, sin_yaw = np.cos(base_yaw), np.sin(base_yaw)
        qvel = self.data.qvel[self._base_v]
        base_twist = (
            float(cos_yaw * qvel[0] + sin_yaw * qvel[1]),
            float(-sin_yaw * qvel[0] + cos_yaw * qvel[1]),
            float(qvel[2]),
        )
        return WbcObservation(
            base_pose=self._base_pose(),
            base_twist=base_twist,
            torso=torso,
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )


__all__ = ["RoboCasaPandaBackend"]
