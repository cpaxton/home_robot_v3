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
        self._base_step = np.asarray((0.05, 0.05, 0.10), dtype=np.float64)
        self._base_jnt = np.zeros(3, dtype=int)
        self._base_stiction_ctrl = np.zeros(3, dtype=np.float64)
        self._teacher_base_speed_scale = 1.0
        self._target_torso = float(_READY_TORSO)
        self._ee_in_base_T: np.ndarray | None = None
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
        self._base_jnt = np.asarray([self.model.joint(name).id for name in _BASE_JOINTS], dtype=int)
        self._base_q = np.asarray([self.model.joint(name).qposadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_v = np.asarray([self.model.joint(name).dofadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_act = np.asarray([self.model.actuator(name).id for name in _BASE_ACTUATORS], dtype=int)
        kv = np.abs(np.asarray(self.model.actuator_gainprm[self._base_act, 0], dtype=np.float64))
        friction = np.asarray(self.model.dof_frictionloss[self._base_v], dtype=np.float64)
        self._base_stiction_ctrl = np.divide(friction, kv, out=np.zeros(3, dtype=np.float64), where=kv > 1e-6)
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
        self._ee_in_base_T = None
        self._target_q = None
        if pose is not None:
            self._target_ee = self._target_ee_override

    def hold_ee_in_base_frame(self, enabled: bool = True) -> None:
        """Keep the active EE target fixed in the moving base frame during transport."""
        if not enabled:
            self._ee_in_base_T = None
            return
        if self._target_ee is None or self.data is None:
            raise ValueError("hold_ee_in_base_frame requires an initialized EE target")
        target_pos, target_rot = self._world_pose(self._target_ee)
        ee_world = np.eye(4, dtype=np.float64)
        ee_world[:3, :3] = target_rot.as_matrix()
        ee_world[:3, 3] = target_pos
        base_world = np.eye(4, dtype=np.float64)
        base_world[:3, :3] = np.asarray(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        base_world[:3, 3] = np.asarray(self.data.xpos[self._base_body], dtype=np.float64)
        self._ee_in_base_T = np.linalg.inv(base_world) @ ee_world

    def set_base_speed_scale(self, scale: float) -> None:
        """Scale the teacher's planar base speed limits (e.g. gentler loaded transport)."""
        value = float(scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("base speed scale must be positive and finite")
        self._teacher_base_speed_scale = value

    def set_torso_target(self, height: float) -> None:
        """Set the torso lift target in meters within ``[0.0, 0.34]``."""
        value = float(height)
        if not np.isfinite(value):
            raise ValueError("torso target must be finite")
        self._target_torso = float(np.clip(value, 0.0, 0.34))

    def reanchor_origin(self) -> None:
        """Re-anchor the reset-local observation frame to the current base pose.

        Preserves any active world-frame EE target while centering
        ``observation.base_pose`` at ``(0, 0, 0)`` for subsequent arm phases.
        """
        if self.data is None:
            return
        ee_world: tuple[np.ndarray, Rotation] | None = None
        if self._target_ee is not None:
            ee_world = self._world_pose(self._target_ee)
        world_base = self._world_base_pose()
        self._origin_xy = np.asarray(world_base[:2], dtype=np.float64)
        self._origin_yaw = float(world_base[2])
        if ee_world is not None:
            pos, rot = ee_world
            self._target_ee = self._local_pose(_pose7(pos, rot.as_matrix()))
            if self._target_ee_override is not None:
                self._target_ee_override = self._target_ee

    def _sync_base_held_ee_target(self) -> None:
        if self._ee_in_base_T is None or self.data is None:
            return
        base_world = np.eye(4, dtype=np.float64)
        base_world[:3, :3] = np.asarray(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        base_world[:3, 3] = np.asarray(self.data.xpos[self._base_body], dtype=np.float64)
        ee_world = base_world @ self._ee_in_base_T
        self._target_ee = self._local_pose(_pose7(ee_world[:3, 3], ee_world[:3, :3]))

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

    def set_gripper_target(self, values: tuple[float, float]) -> None:
        """Hold a measured finger opening after contact.

        A fully closed target keeps squeezing and can eject a free object during
        transport; this holds the contact opening (plus a small squeeze offset).
        """
        target = np.asarray(values, dtype=np.float64)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("gripper target must contain two finite values")
        self._gripper_target = target
        self._grip_command = target.copy()
        self._force_gripper = True

    def solve_base_target_qpos(self) -> np.ndarray | None:
        """Solve the planar qpos that realizes the current base target."""
        if self._target_base is None or self.data is None:
            return None
        rotation = Rotation.from_euler("z", self._origin_yaw)
        target_xy = rotation.apply(
            np.r_[np.asarray(self._target_base[:2], dtype=np.float64), 0.0]
        )[:2] + self._origin_xy
        target_yaw = _wrap(float(self._target_base[2]) + self._origin_yaw)

        def pose_for(q: np.ndarray) -> tuple[np.ndarray, float]:
            self.data.qpos[self._base_q] = q
            self.env.sim.forward()
            world = self._world_base_pose()
            return np.asarray(world[:2], dtype=np.float64), float(world[2])

        current_q = np.asarray(self.data.qpos[self._base_q], dtype=np.float64).copy()
        q = current_q.copy()
        for _ in range(12):
            xy, yaw = pose_for(q)
            error = np.r_[target_xy - xy, _wrap(target_yaw - yaw)]
            if float(np.linalg.norm(error[:2])) <= 1e-4 and abs(float(error[2])) <= 1e-4:
                break
            jac = np.zeros((3, 3), dtype=np.float64)
            for index in range(3):
                probe = q.copy()
                probe[index] += 1e-4
                probe_xy, probe_yaw = pose_for(probe)
                jac[:2, index] = (probe_xy - xy) / 1e-4
                jac[2, index] = _wrap(probe_yaw - yaw) / 1e-4
            try:
                step = np.linalg.lstsq(jac, error, rcond=None)[0]
            except np.linalg.LinAlgError:
                self.data.qpos[self._base_q] = current_q
                self.env.sim.forward()
                return None
            q += np.clip(step, -0.25, 0.25)
        xy, yaw = pose_for(q)
        solved = bool(
            np.linalg.norm(target_xy - xy) <= 5e-3 and abs(_wrap(target_yaw - yaw)) <= 5e-3
        )
        self.data.qpos[self._base_q] = current_q
        self.data.qvel[self._base_v] = 0.0
        self.env.sim.forward()
        return q if solved else None

    def apply_base_qpos(self, q: np.ndarray) -> None:
        """Kinematic base move used by the teacher's transport ramp."""
        self.data.qpos[self._base_q] = q
        self.data.qvel[self._base_v] = 0.0
        self.env.sim.forward()
        self._base_command[:] = 0.0

    def teleport_base_target(self) -> bool:
        """Privileged teacher convenience: snap the planar base to its target."""
        q = self.solve_base_target_qpos()
        if q is None:
            return False
        self.apply_base_qpos(q)
        return True

    def set_local_ee_target(self, pose: tuple[float, ...] | None) -> None:
        """Set an end-effector target in the reset-anchored robot frame."""
        if pose is not None and len(pose) != 7:
            raise ValueError("EE target must be a 7-value position/quaternion pose")
        self._target_ee_override = None if pose is None else tuple(float(v) for v in pose)
        self._ee_in_base_T = None
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
        self._ee_in_base_T = None
        self._teacher_base_speed_scale = 1.0
        self._target_torso = float(_READY_TORSO)
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
        torso_torque = float(
            np.clip(1000.0 * (self._target_torso - torso) - 200.0 * torso_qd + torso_bias, -2000.0, 2000.0)
        )
        self.data.ctrl[self._torso_act] = torso_torque
        base = np.asarray(command.base_velocity, dtype=np.float64).reshape(-1)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("RoboCasaPandaBackend requires a finite three-axis base velocity")
        limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        desired_base = np.clip(base, -limits, limits)
        self._base_command = self._base_command + np.clip(
            desired_base - self._base_command, -self._base_step, self._base_step
        )
        qdot_des = self._body_twist_to_joint_vel(
            float(self._base_command[0]), float(self._base_command[1]), float(self._base_command[2])
        )
        base_ctrl = np.where(
            np.abs(qdot_des) > 1e-3,
            qdot_des + self._base_stiction_ctrl * np.tanh(qdot_des / 0.015),
            0.0,
        )
        for actuator, value in zip(self._base_act, base_ctrl, strict=True):
            low, high = self.model.actuator_ctrlrange[actuator]
            self.data.ctrl[actuator] = float(np.clip(value, low, high))
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
        self._sync_base_held_ee_target()
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

    def contact_breakdown(self, target_body: str) -> dict[str, float]:
        """Privileged diagnostic: split finger load by target vs environment.

        Never call this from the actor path; the target body is GT metadata.
        """
        from emet.rl.contact import solved_contact_force

        target = int(self.model.body(target_body).id)
        finger_bodies = set().union(*self._contact_sensor.finger_bodies_set)
        out = {
            "finger_target_N": 0.0,
            "finger_environment_N": 0.0,
            "robot_environment_N": 0.0,
            "other_N": 0.0,
        }
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            normal, _ = solved_contact_force(self.model, self.data, index)
            first_finger = first in finger_bodies
            second_finger = second in finger_bodies
            if first_finger or second_finger:
                other = second if first_finger else first
                if other == target:
                    out["finger_target_N"] += normal
                elif other not in self._contact_sensor.robot_bodies:
                    out["finger_environment_N"] += normal
            elif first in self._contact_sensor.robot_bodies or second in self._contact_sensor.robot_bodies:
                out["robot_environment_N"] += normal
            else:
                out["other_N"] += normal
        return out

    def finger_target_forces(self, target_body: str) -> tuple[float, float]:
        """Privileged diagnostic: split target load by left/right finger.

        Returns ``(left_N, right_N)``. Never call this from the actor path; the
        target body is GT metadata.
        """
        from emet.rl.contact import solved_contact_force

        target = int(self.model.body(target_body).id)
        finger_sets = self._contact_sensor.finger_bodies_set
        totals = [0.0] * len(finger_sets)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            first = int(self.model.geom_bodyid[contact.geom1])
            second = int(self.model.geom_bodyid[contact.geom2])
            if target not in (first, second):
                continue
            other = first if second == target else second
            finger = next((i for i, bodies in enumerate(finger_sets) if other in bodies), None)
            if finger is None:
                continue
            normal, _ = solved_contact_force(self.model, self.data, index)
            totals[finger] += normal
        return float(totals[0]), float(totals[1])

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
        self._sync_base_held_ee_target()
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
        torso_torque = float(
            np.clip(1000.0 * (self._target_torso - torso) - 200.0 * torso_qd + torso_bias, -2000.0, 2000.0)
        )
        base_velocity = (0.0, 0.0, 0.0)
        if self._target_base is not None:
            base = np.asarray(self._base_pose(), dtype=np.float64)
            target = np.asarray(self._target_base, dtype=np.float64)
            dx, dy = target[0] - base[0], target[1] - base[1]
            yaw = _wrap(float(target[2] - base[2]))
            if float(np.hypot(dx, dy)) > 0.025 or abs(yaw) > 0.04:
                self._teacher_phase = "base"
                forward = dx * np.cos(base[2]) + dy * np.sin(base[2])
                lateral = -dx * np.sin(base[2]) + dy * np.cos(base[2])
                limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64) * self._teacher_base_speed_scale
                base_velocity = (
                    float(np.clip(1.5 * forward, -limits[0], limits[0])) if abs(forward) > 0.01 else 0.0,
                    float(np.clip(1.5 * lateral, -limits[1], limits[1])) if abs(lateral) > 0.01 else 0.0,
                    float(np.clip(1.8 * yaw, -limits[2], limits[2])) if abs(yaw) > 0.015 else 0.0,
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

    def _body_twist_to_joint_vel(self, vx: float, vy: float, wz: float) -> np.ndarray:
        """Map chassis-frame twist ``(vx, vy, wz)`` to Omron planar joint rates.

        In ``omron_mobile_base.xml`` the two slide joints precede ``joint_mobile_yaw``
        and ``joint_mobile_yaw`` is anchored at ``(-0.21, 0, 0)``, so the slide axes
        do not rotate with heading and yaw rate sweeps the chassis origin.
        """
        rotation = np.asarray(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        forward_axis, left_axis = rotation[:2, 0], rotation[:2, 1]
        slide_0 = np.asarray(self.data.xaxis[self._base_jnt[0], :2], dtype=np.float64)
        slide_1 = np.asarray(self.data.xaxis[self._base_jnt[1], :2], dtype=np.float64)
        lever = np.asarray(self.data.xpos[self._base_body, :2] - self.data.xanchor[self._base_jnt[2], :2], dtype=np.float64)
        v_world = vx * forward_axis + vy * left_axis
        v_slide = v_world - wz * np.array([-lever[1], lever[0]], dtype=np.float64)
        return np.array([float(np.dot(v_slide, slide_0)), float(np.dot(v_slide, slide_1)), float(wz)], dtype=np.float64)

    def _joint_vel_to_body_twist(self) -> tuple[float, float, float]:
        rotation = np.asarray(self.data.xmat[self._base_body], dtype=np.float64).reshape(3, 3)
        forward_axis, left_axis = rotation[:2, 0], rotation[:2, 1]
        slide_0 = np.asarray(self.data.xaxis[self._base_jnt[0], :2], dtype=np.float64)
        slide_1 = np.asarray(self.data.xaxis[self._base_jnt[1], :2], dtype=np.float64)
        lever = np.asarray(self.data.xpos[self._base_body, :2] - self.data.xanchor[self._base_jnt[2], :2], dtype=np.float64)
        qvel = np.asarray(self.data.qvel[self._base_v], dtype=np.float64)
        v_world = qvel[0] * slide_0 + qvel[1] * slide_1 + qvel[2] * np.array([-lever[1], lever[0]], dtype=np.float64)
        return (float(np.dot(v_world, forward_axis)), float(np.dot(v_world, left_axis)), float(qvel[2]))

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
        base_twist = self._joint_vel_to_body_twist()
        return WbcObservation(
            base_pose=self._base_pose(),
            base_twist=base_twist,
            torso=torso,
            arms=(arm,),
            occupancy=None,
            base_target=self._target_base,
        )


__all__ = ["RoboCasaPandaBackend"]
