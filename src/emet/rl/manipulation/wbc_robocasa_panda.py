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
    ArmAction,
    ArmObservation,
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
from emet.rl.manipulation.wbc_contact import MuJoCoContactSensor
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
        target_mode: str = "offset",
        target_range: float = 0.15,
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
        self.target_mode = str(target_mode)
        self.target_range = float(target_range)
        if self.target_mode not in {"offset", "joint"}:
            raise ValueError("target_mode must be 'offset' or 'joint'")
        if self.base_backoff < 0:
            raise ValueError("base_backoff must be nonnegative")
        if self.target_range <= 0:
            raise ValueError("target_range must be positive")
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
        self._gripper_target = np.asarray((0.04, -0.04), dtype=np.float64)
        self._force_gripper = False
        self._teacher_phase = "arm"
        self._elapsed = 0.0
        self._origin_xy = np.zeros(2)
        self._origin_yaw = 0.0
        arm_joints = self.profile.joint_group("panda_arm").joints
        self._arm_joints = arm_joints
        self._kp = np.asarray([float(joint.kp_eff or 0.0) for joint in arm_joints])
        self._kd = np.asarray([float(joint.kd_eff or 0.0) for joint in arm_joints])
        self._torque_limits = np.asarray([float(joint.torque_limit) for joint in arm_joints])
        self._arm_low = np.asarray([joint.position_limit[0] for joint in arm_joints])
        self._arm_high = np.asarray([joint.position_limit[1] for joint in arm_joints])
        self._target_q: np.ndarray | None = None

    def _initialize_sim(self) -> None:
        self.model = self.env.sim.model
        self.data = self.env.sim.data
        self.substeps = int(round(self.control_dt / float(self.model.opt.timestep)))
        if self.substeps < 1 or abs(self.substeps * float(self.model.opt.timestep) - self.control_dt) > 1e-9:
            raise ValueError("RoboCasa control_dt must be an integer number of simulation timesteps")
        self.model_fingerprint = hashlib.sha256(self.env.sim.model.get_xml().encode()).hexdigest()[:16]
        gripper = self.profile.arms[0].gripper
        assert gripper is not None
        if resolve_body_name(self.model, self.profile.arms[0].ee_link, PANDAOMRON_EE_LINK_ALIASES) != _EE_BODY:
            raise ValueError("RoboCasa Panda EE aliases do not match the composed model")
        if resolve_joint_names(self.model, gripper.joint_names, PANDAOMRON_GRIPPER_JOINT_ALIASES) != _GRIPPER_JOINTS:
            raise ValueError("RoboCasa Panda gripper aliases do not match the composed model")
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
        self._contact_sensor = MuJoCoContactSensor(
            self.model,
            base_body=_BASE_BODY,
            ee_body=_EE_BODY,
            finger_bodies=("gripper0_right_leftfinger", "gripper0_right_rightfinger"),
            support_bodies=("mobilebase0_fixed_support", "mobilebase0_support"),
        )
        self.collision_measured = True

    def set_ee_target(self, pose: tuple[float, ...] | None) -> None:
        """Set a world-frame target (7 values) or restore the reset offset.

        The actor observation stores targets in the reset-anchored robot frame;
        a task/skill layer naturally supplies world poses, so convert here.
        """
        if pose is not None and len(pose) != 7:
            raise ValueError("EE target must be a 7-value position/quaternion pose")
        if pose is not None and self.model is None:
            raise ValueError("set_ee_target requires an initialized reset")
        self._target_ee_override = None if pose is None else self._local_pose(tuple(float(v) for v in pose))
        self._target_q = None
        if pose is not None:
            self._target_ee = self._target_ee_override

    def set_base_target(self, pose: tuple[float, float, float] | None, *, world: bool = False) -> None:
        """Set a base XYT target in the reset-anchored frame (or world frame)."""
        if pose is None:
            self._target_base = None
            self._teacher_phase = "arm"
            return
        if len(pose) != 3 or not np.isfinite(np.asarray(pose, dtype=np.float64)).all():
            raise ValueError("base target must contain three finite values")
        if world:
            rotation = Rotation.from_euler("z", -self._origin_yaw)
            local_xy = rotation.apply(
                np.r_[np.asarray(pose[:2], dtype=np.float64) - self._origin_xy, 0.0]
            )[:2]
            pose = (float(local_xy[0]), float(local_xy[1]), float(_wrap(float(pose[2]) - self._origin_yaw)))
        self._target_base = (float(pose[0]), float(pose[1]), float(pose[2]))
        self._teacher_phase = "base"

    def set_world_base_target(self, pose: tuple[float, float, float] | None) -> None:
        self.set_base_target(pose, world=True)

    def set_gripper(self, open_: bool, *, force: bool = True) -> None:
        """Set a scripted gripper target for task execution."""
        self._gripper_target = np.asarray((0.04, -0.04) if open_ else (0.0, 0.0), dtype=np.float64)
        self._grip_command = self._gripper_target.copy()
        self._force_gripper = bool(force)

    def teleport_base_target(self) -> bool:
        """Privileged teacher convenience: snap the planar base to its target.

        The RoboCasa merged scene can carry a non-zero world offset; solve the
        planar joints through FK instead of writing the reset-frame XYT into qpos.
        """
        if self._target_base is None or self.data is None:
            return False
        from emet.simulation.spawn_planar import write_planar_base_xyt

        rotation = Rotation.from_euler("z", self._origin_yaw)
        world_xy = rotation.apply(
            np.r_[np.asarray(self._target_base[:2], dtype=np.float64), 0.0]
        )[:2] + self._origin_xy
        world_yaw = _wrap(float(self._target_base[2]) + self._origin_yaw)
        if not write_planar_base_xyt(
            self.model,
            self.data,
            joint_names=_BASE_JOINTS,
            world_x=float(world_xy[0]),
            world_y=float(world_xy[1]),
            world_yaw=world_yaw,
            base_body_name=_BASE_BODY,
        ):
            return False
        self.data.qvel[self._base_v] = 0.0
        self.env.sim.forward()
        self._base_command[:] = 0.0
        return True

    def set_local_ee_target(self, pose: tuple[float, ...] | None) -> None:
        """Set an end-effector target in the reset-anchored robot frame."""
        if pose is not None and len(pose) != 7:
            raise ValueError("EE target must be a 7-value position/quaternion pose")
        self._target_ee_override = None if pose is None else tuple(float(v) for v in pose)
        self._target_q = None
        if pose is not None:
            self._target_ee = self._target_ee_override

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
            self._target_q = None
        elif self.target_mode == "joint":
            self._target_q, self._target_ee = self._sample_joint_target()
        else:
            self._target_q = None
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
        if self._force_gripper:
            values = np.asarray(self._gripper_target, dtype=np.float64).reshape(-1)
        elif grip is not None:
            values = np.asarray(grip.values, dtype=np.float64).reshape(-1)
        else:
            values = None
        if values is not None:
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
        contact = self._contact_sensor.read(self.data)
        self._elapsed += self.control_dt
        return (
            self._observation(),
            invalid,
            {
                "contact_force": contact.collision_force,
                "collision": contact.self_collision,
                "collision_measured": True,
                "hand_contact_force": contact.hand_force,
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

    def _world_pose(self, local: tuple[float, ...]) -> tuple[np.ndarray, Rotation]:
        rotation = Rotation.from_euler("z", self._origin_yaw)
        position = rotation.apply(np.asarray(local[:3], dtype=np.float64)) + np.array(
            [self._origin_xy[0], self._origin_xy[1], 0.0]
        )
        return position, rotation * Rotation.from_quat(local[3:])

    def _sample_joint_target(self) -> tuple[np.ndarray, tuple[float, ...]]:
        q0 = self.data.qpos[self._arm_q].copy()
        q = np.clip(
            q0 + self.rng.uniform(-self.target_range, self.target_range, len(q0)),
            self._arm_low,
            self._arm_high,
        )
        self.data.qpos[self._arm_q] = q
        self.env.sim.forward()
        target = self._local_pose(_pose7(self.data.xpos[self._ee_body], self.data.xmat[self._ee_body]))
        self.data.qpos[self._arm_q] = q0
        self.env.sim.forward()
        return q.copy(), target

    def _numerical_jacobian(self, q: np.ndarray) -> np.ndarray:
        """6x7 world-frame EE Jacobian by finite differences."""
        eps = 1e-3
        position = self.data.xpos[self._ee_body].copy()
        rotation = Rotation.from_matrix(self.data.xmat[self._ee_body].reshape(3, 3))
        jacobian = np.zeros((6, len(q)), dtype=np.float64)
        for index in range(len(q)):
            perturbed = q.copy()
            perturbed[index] = float(np.clip(perturbed[index] + eps, self._arm_low[index], self._arm_high[index]))
            if perturbed[index] == q[index]:
                continue
            self.data.qpos[self._arm_q] = perturbed
            self.env.sim.forward()
            next_position = self.data.xpos[self._ee_body]
            next_rotation = Rotation.from_matrix(self.data.xmat[self._ee_body].reshape(3, 3))
            jacobian[:3, index] = (next_position - position) / eps
            jacobian[3:, index] = (next_rotation * rotation.inv()).as_rotvec() / eps
        self.data.qpos[self._arm_q] = q
        self.env.sim.forward()
        return jacobian

    def _cartesian_torque(self, q: np.ndarray, qd: np.ndarray, bias: np.ndarray) -> np.ndarray:
        target_position, target_rotation = self._world_pose(self._target_ee)  # type: ignore[arg-type]
        position = self.data.xpos[self._ee_body].copy()
        rotation = Rotation.from_matrix(self.data.xmat[self._ee_body].reshape(3, 3))
        position_error = target_position - position
        orientation_error = (target_rotation * rotation.inv()).as_rotvec()
        jacobian = self._numerical_jacobian(q)
        linear_velocity = jacobian[:3] @ qd
        angular_velocity = jacobian[3:] @ qd
        wrench = np.r_[
            500.0 * position_error - 50.0 * linear_velocity,
            200.0 * orientation_error - 20.0 * angular_velocity,
        ]
        return jacobian.T @ wrench + bias

    def teacher_action(self) -> np.ndarray:
        """Privileged joint-PD (or Cartesian) teacher for RoboCasa data generation."""
        if self.data is None:
            raise RuntimeError("reset required before teacher_action")
        q = np.asarray(self.data.qpos[self._arm_q], dtype=np.float64)
        qd = np.asarray(self.data.qvel[self._arm_v], dtype=np.float64)
        bias = np.asarray(self.data.qfrc_bias[self._arm_v], dtype=np.float64)
        if self._target_q is not None:
            torque = self._kp * (self._target_q - q) - self._kd * qd + bias
        elif self._target_ee is not None:
            torque = self._cartesian_torque(q, qd, bias)
        else:
            torque = -self._kd * qd + bias
        torque = np.clip(torque, -self._torque_limits, self._torque_limits)
        torso = float(self.data.qpos[self._torso_q])
        torso_qd = float(self.data.qvel[self._torso_v])
        torso_bias = float(self.data.qfrc_bias[self._torso_v])
        torso_torque = float(np.clip(1000.0 * (_READY_TORSO - torso) - 200.0 * torso_qd + torso_bias, -2000.0, 2000.0))
        base_velocity = (0.0, 0.0, 0.0)
        if self._target_base is not None:
            base = np.asarray(self._base_pose(), dtype=np.float64)
            target = np.asarray(self._target_base, dtype=np.float64)
            dx, dy = target[0] - base[0], target[1] - base[1]
            yaw = _wrap(float(target[2] - base[2]))
            if float(np.hypot(dx, dy)) > 0.03 or abs(yaw) > 0.05:
                self._teacher_phase = "base"
                forward = dx * np.cos(base[2]) + dy * np.sin(base[2])
                lateral = -dx * np.sin(base[2]) + dy * np.cos(base[2])
                limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
                base_velocity = (
                    float(np.sign(forward) * min(limits[0], max(0.2, 2.0 * abs(forward))))
                    if abs(forward) > 0.02
                    else 0.0,
                    float(np.sign(lateral) * min(limits[1], max(0.2, 2.0 * abs(lateral))))
                    if abs(lateral) > 0.02
                    else 0.0,
                    float(np.sign(yaw) * min(limits[2], max(0.2, 2.0 * abs(yaw))))
                    if abs(yaw) > 0.03
                    else 0.0,
                )
            else:
                self._teacher_phase = "arm"
        action = WbcAction(
            base_velocity=base_velocity,
            torso_torques=(torso_torque,),
            arms=(
                ArmAction(
                    name="right",
                    joint_torques=tuple(float(v) for v in torque),
                    gripper=GripperCommand(values=tuple(float(v) for v in self._gripper_target)),
                ),
            ),
        )
        return normalized_action_vector(action, self.profile).astype(np.float32)

    def _base_pose(self) -> tuple[float, float, float]:
        world = self._world_base_pose()
        rotation = Rotation.from_euler("z", -self._origin_yaw)
        position = rotation.apply(
            np.asarray(
                (world[0] - self._origin_xy[0], world[1] - self._origin_xy[1], 0.0),
                dtype=np.float64,
            )
        )
        return (float(position[0]), float(position[1]), float(_wrap(world[2] - self._origin_yaw)))

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
        # The Omron planar slide joints are already expressed in the chassis
        # frame, matching the direct plant's base twist.
        qvel = self.data.qvel[self._base_v]
        base_twist = (float(qvel[0]), float(qvel[1]), float(qvel[2]))
        return WbcObservation(
            base_pose=self._base_pose(),
            base_twist=base_twist,
            torso=torso,
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )


__all__ = ["RoboCasaPandaBackend"]
