# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo whole-body backend for the Sourccey Mk.V.

The contract still speaks torques, but the packaged Sourccey MJCF exposes
position actuators for the lift, arms, and grippers plus planar velocity
actuators for the base. :class:`SourcceyWbcBackend` is therefore the plant side
of the existing :class:`~emet.controller.manipulation.torque_adapter.PositionServoAdapter`
loop: the adapter turns a policy torque into a bounded joint-position lead and
this backend realizes that target with the plant's own position servos.

Base commands follow the planar contract ``(vx, omega)``. ``vx`` is body-forward
and is rotated into the world-aligned ``base_x``/``base_y`` slide actuators with
the measured yaw; ``omega`` drives ``base_yaw_act``. The plant has no lateral
velocity channel, so a planar policy cannot strafe.

All goals are sampled in the world frame because ``sourccey_sim_profile``
declares ``target_frame="world"`` for both arms.
"""

from __future__ import annotations

import hashlib
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
    JointGroupObservation,
    WbcAction,
    WbcObservation,
    normalized_action_vector,
)
from emet.controller.manipulation.wbc_profiles import sourccey_sim_profile
from emet.rl.manipulation.wbc_contact import MuJoCoContactSensor
from emet.utils.assets import get_robot_mjcf_path

_ARM_SUFFIXES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
_BASE_JOINTS = ("base_x", "base_y", "base_yaw")
_BASE_ACTUATORS = ("base_x_act", "base_y_act", "base_yaw_act")

# Sampling and command-shaping constants. These are simulator-side limits, not
# contract fields: the contract owns velocity/torque envelopes, the plant owns
# how fast a position target may move.
_MIN_EE_DISPLACEMENT = 0.03  # m, so an "arm goal" is not a no-op
_GRIPPER_SPEED = 2.0  # rad/s
_BASE_STAGE_TIMEOUT_S = 2.0


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _pose7(position: np.ndarray, rotation: np.ndarray) -> tuple[float, ...]:
    """Return ``(xyz, quat_xyzw)`` from a MuJoCo body pose."""
    quat = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat()
    return tuple(float(v) for v in np.r_[position, quat])


class SourcceyWbcBackend:
    """MuJoCo plant for the Sourccey sim profile with sampled whole-body goals."""

    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        seed: int = 0,
        control_dt: float = 0.05,
        target_range: float = 0.15,
        arm_goal_probability: float = 1.0,
        base_goal_probability: float = 0.5,
        base_goal_mode: str = "forward",
        active_arms: tuple[str, ...] | None = None,
    ) -> None:
        self.profile = sourccey_sim_profile() if profile is None else profile
        if self.profile.meta("robot") != "sourccey":
            raise ValueError("SourcceyWbcBackend requires a sourccey embodiment profile")
        if self.profile.base.kind != "planar":
            raise ValueError("SourcceyWbcBackend requires a planar contract base")
        if not np.isfinite([control_dt, arm_goal_probability, base_goal_probability]).all():
            raise ValueError("backend timing/probability values must be finite")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        if not 0 <= arm_goal_probability <= 1 or not 0 <= base_goal_probability <= 1:
            raise ValueError("goal probabilities must be in [0, 1]")
        if arm_goal_probability > 0 and target_range <= 0:
            raise ValueError("target_range must be positive when arm goals are enabled")
        if base_goal_mode not in {"forward", "any"}:
            raise ValueError("base_goal_mode must be forward or any")
        if tuple(arm.name for arm in self.profile.arms) != ("left", "right"):
            raise ValueError("SourcceyWbcBackend requires left and right arm profiles")
        if active_arms is not None:
            if not active_arms or len(set(active_arms)) != len(active_arms):
                raise ValueError("active_arms must be non-empty and unique")
            unknown_arms = set(active_arms) - {"left", "right"}
            if unknown_arms:
                raise ValueError(f"unknown active arms: {sorted(unknown_arms)}")
            self.active_arms = tuple(active_arms)
        else:
            self.active_arms = ("left", "right")
        if self.profile.torso is None or self.profile.torso.joint_names != ("lift",):
            raise ValueError("SourcceyWbcBackend requires the single-joint lift torso group")
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            expected = tuple(f"{arm_spec.name}_{suffix}" for suffix in _ARM_SUFFIXES)
            if group.joint_names != expected:
                raise ValueError(f"{arm_spec.name} joint order does not match the Sourccey MJCF")
            if arm_spec.gripper is None or arm_spec.gripper.joint_names != (f"{arm_spec.name}_gripper",):
                raise ValueError(f"{arm_spec.name} gripper does not match the Sourccey MJCF")
            if arm_spec.target_frame != "world":
                raise ValueError("SourcceyWbcBackend emits world-frame EE poses; use target_frame='world'")

        self.control_dt = float(control_dt)
        self.arm_goal_probability = float(arm_goal_probability)
        self.base_goal_probability = float(base_goal_probability)
        self.base_goal_mode = base_goal_mode
        self.target_range = float(target_range)
        self._arm_target_range = float(target_range) if target_range > 0 else 0.0
        self.rng = np.random.default_rng(seed)

        mjcf_path = get_robot_mjcf_path("sourccey")
        if mjcf_path is None or not mjcf_path.is_file():
            raise RuntimeError("Sourccey MJCF not found; install the emet package data or use a source checkout")
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.model_fingerprint = hashlib.sha256(mjcf_path.read_bytes()).hexdigest()
        self.data = mujoco.MjData(self.model)
        step = float(self.model.opt.timestep)
        if step <= 0:
            raise ValueError("Sourccey MJCF must declare a positive timestep")
        self.substeps = int(round(self.control_dt / step))
        if self.substeps < 1 or abs(self.substeps * step - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer number of MJCF timesteps")

        self._base_q = np.asarray([self.model.joint(name).qposadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_v = np.asarray([self.model.joint(name).dofadr[0] for name in _BASE_JOINTS], dtype=int)
        self._base_act = np.asarray([self.model.actuator(name).id for name in _BASE_ACTUATORS], dtype=int)

        self._joint_q: dict[str, int] = {}
        self._joint_v: dict[str, int] = {}
        self._joint_act: dict[str, int] = {}
        self._joint_step: dict[str, float] = {}
        self._limits: dict[str, tuple[float, float]] = {}
        for joint in self.profile.torso.joints:
            self._bind_profile_joint(joint.name, joint.position_limit, joint.velocity_limit)
        for arm_spec in self.profile.arms:
            for joint in self.profile.joint_group(arm_spec.joint_group).joints:
                self._bind_profile_joint(joint.name, joint.position_limit, joint.velocity_limit)

        self._gripper_limits: dict[str, tuple[float, float]] = {}
        self._gripper_step: dict[str, float] = {}
        self._gripper_act: dict[str, int] = {}
        self._gripper_open: dict[str, tuple[float, ...]] = {}
        self._gripper_joint_names: dict[str, tuple[str, ...]] = {}
        for arm_spec in self.profile.arms:
            gripper = arm_spec.gripper
            assert gripper is not None
            names = gripper.joint_names
            self._gripper_joint_names[arm_spec.name] = names
            command_limits = gripper.command_limits
            opens: list[float] = []
            for index, name in enumerate(names):
                model_lo, model_hi = self._model_joint_limits(name)
                if command_limits is not None:
                    model_lo = max(model_lo, float(command_limits[index][0]))
                    model_hi = min(model_hi, float(command_limits[index][1]))
                if model_lo >= model_hi:
                    raise ValueError(f"{name} command limits do not intersect the MJCF joint range")
                self._gripper_limits[name] = (model_lo, model_hi)
                self._gripper_step[name] = _GRIPPER_SPEED * self.control_dt
                self._gripper_act[name] = self.model.actuator(f"{name}_act").id
                self._joint_q[name] = int(self.model.joint(name).qposadr[0])
                self._joint_v[name] = int(self.model.joint(name).dofadr[0])
                opens.append(model_hi)
            self._gripper_open[arm_spec.name] = tuple(opens)

        self._ee_body = {arm.name: int(self.model.body(arm.ee_link).id) for arm in self.profile.arms}
        support_bodies = tuple(
            str(self.model.body(index).name)
            for index in range(self.model.nbody)
            if "wheel" in str(self.model.body(index).name).lower()
        )
        self._contact_sensor = MuJoCoContactSensor(
            self.model,
            base_body="base_root",
            ee_body=self.profile.arms[0].ee_link,
            hand_bodies=tuple(arm.ee_link for arm in self.profile.arms),
            finger_bodies=("Gripper_Finger_v1_1", "Gripper_Finger_v1_2"),
            support_bodies=support_bodies,
        )
        self.collision_measured = True

        velocity_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        if self.profile.base.acceleration_limits is None:
            # Reach the declared velocity envelope in roughly half a second.
            self._base_accel = 2.0 * velocity_limits
        else:
            self._base_accel = np.asarray(self.profile.base.acceleration_limits, dtype=np.float64)

        self._position_command: dict[str, float] = {}
        self._gripper_command: dict[str, float] = {}
        self._base_command = np.zeros(2, dtype=np.float64)
        self._start_q: dict[str, np.ndarray] = {}
        self._target_q: dict[str, np.ndarray] = {}
        self._target_ee: dict[str, tuple[float, ...] | None] = {}
        self._target_base: tuple[float, float, float] | None = None
        self._lift_target = 0.0
        self._teacher_phase = "arm"
        self._elapsed = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _bind_profile_joint(
        self,
        name: str,
        position_limit: tuple[float, float],
        velocity_limit: float,
    ) -> None:
        model_lo, model_hi = self._model_joint_limits(name)
        lo = max(model_lo, float(position_limit[0]))
        hi = min(model_hi, float(position_limit[1]))
        if lo >= hi:
            raise ValueError(f"{name} profile limits {position_limit} do not intersect the MJCF joint range")
        self._joint_q[name] = int(self.model.joint(name).qposadr[0])
        self._joint_v[name] = int(self.model.joint(name).dofadr[0])
        self._joint_act[name] = int(self.model.actuator(f"{name}_act").id)
        self._joint_step[name] = float(velocity_limit) * self.control_dt
        self._limits[name] = (lo, hi)

    def _model_joint_limits(self, name: str) -> tuple[float, float]:
        joint = self.model.joint(name)
        if not self.model.jnt_limited[joint.id]:
            return -np.inf, np.inf
        lo, hi = self.model.jnt_range[joint.id]
        return float(lo), float(hi)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._base_q] = 0.0

        lift_lo, lift_hi = self._limits["lift"]
        # The scene is collision-free at the raised home posture; resetting
        # there keeps the arm task learnable instead of sampling extreme poses.
        self.data.qpos[self._joint_q["lift"]] = float(np.clip(lift_hi - 0.05, lift_lo + 0.02, lift_hi - 0.02))
        for arm_spec in self.profile.arms:
            for joint in self.profile.joint_group(arm_spec.joint_group).joints:
                lo, hi = self._limits[joint.name]
                self.data.qpos[self._joint_q[joint.name]] = float(np.clip(self.rng.uniform(-0.05, 0.05), lo, hi))
        for arm_spec in self.profile.arms:
            for name, value in zip(
                self._gripper_joint_names[arm_spec.name], self._gripper_open[arm_spec.name], strict=True
            ):
                self.data.qpos[self._joint_q[name]] = value
        mujoco.mj_forward(self.model, self.data)

        self._position_command = {
            name: float(self.data.qpos[self._joint_q[name]]) for name in self.profile.controlled_joint_names
        }
        self._gripper_command = {name: float(self.data.qpos[self._joint_q[name]]) for name in self._gripper_act}
        self._start_q = {
            arm_spec.name: np.asarray(
                [self.data.qpos[self._joint_q[j.name]] for j in self.profile.joint_group(arm_spec.joint_group).joints],
                dtype=np.float64,
            )
            for arm_spec in self.profile.arms
        }
        self._lift_target = float(self.data.qpos[self._joint_q["lift"]])
        self._base_command = np.zeros(2, dtype=np.float64)
        self._elapsed = 0.0
        self._target_base = self._sample_base_target()

        if self.rng.random() < self.arm_goal_probability:
            self._target_q, self._target_ee = self._sample_arm_targets(self._target_base)
        else:
            self._target_q = {name: q.copy() for name, q in self._start_q.items()}
            self._target_ee = {arm_spec.name: None for arm_spec in self.profile.arms}

        self._teacher_phase = "base" if self._target_base is not None else "arm"
        self._apply_control()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def _sample_base_target(self) -> tuple[float, float, float] | None:
        if self.rng.random() >= self.base_goal_probability:
            return None
        x, y, yaw = self._base_pose()
        if self.base_goal_mode == "forward":
            distance = float(self.rng.uniform(0.12, 0.35))
            bearing = float(self.rng.uniform(-0.35, 0.35))
            heading = yaw + bearing
            target_x = x + distance * np.cos(heading)
            target_y = y + distance * np.sin(heading)
            target_yaw = heading + self.rng.uniform(-0.05, 0.05)
        else:
            target_x = x + self.rng.uniform(-0.15, 0.15)
            target_y = y + self.rng.uniform(-0.15, 0.15)
            target_yaw = yaw + self.rng.uniform(-0.3, 0.3)
        return (float(target_x), float(target_y), float(target_yaw))

    def _sample_arm_targets(
        self,
        base_target: tuple[float, float, float] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, ...] | None]]:
        saved_base = self.data.qpos[self._base_q].copy()
        saved_arms = {name: q.copy() for name, q in self._start_q.items()}
        if base_target is not None:
            self.data.qpos[self._base_q] = base_target
            mujoco.mj_forward(self.model, self.data)

        targets: dict[str, np.ndarray] = {}
        poses: dict[str, tuple[float, ...] | None] = {}
        try:
            for arm_spec in self.profile.arms:
                if arm_spec.name not in self.active_arms:
                    targets[arm_spec.name] = self._start_q[arm_spec.name].copy()
                    poses[arm_spec.name] = None
                    continue
                group = self.profile.joint_group(arm_spec.joint_group)
                q0 = np.asarray(
                    [self.data.qpos[self._joint_q[joint.name]] for joint in group.joints],
                    dtype=np.float64,
                )
                ee_body = self._ee_body[arm_spec.name]
                ee0 = self.data.xpos[ee_body].copy()
                best_q = q0.copy()
                best_pose = _pose7(self.data.xpos[ee_body], self.data.xmat[ee_body])
                for _ in range(32):
                    sample = np.clip(
                        q0 + self.rng.uniform(-self._arm_target_range, self._arm_target_range, size=q0.size),
                        [self._limits[joint.name][0] for joint in group.joints],
                        [self._limits[joint.name][1] for joint in group.joints],
                    )
                    for joint, value in zip(group.joints, sample, strict=True):
                        self.data.qpos[self._joint_q[joint.name]] = value
                    mujoco.mj_forward(self.model, self.data)
                    best_q = sample
                    best_pose = _pose7(self.data.xpos[ee_body], self.data.xmat[ee_body])
                    if np.linalg.norm(self.data.xpos[ee_body] - ee0) >= _MIN_EE_DISPLACEMENT:
                        break
                targets[arm_spec.name] = best_q.copy()
                poses[arm_spec.name] = best_pose
        finally:
            for arm_spec in self.profile.arms:
                group = self.profile.joint_group(arm_spec.joint_group)
                for joint, value in zip(group.joints, saved_arms[arm_spec.name], strict=True):
                    self.data.qpos[self._joint_q[joint.name]] = value
            self.data.qpos[self._base_q] = saved_base
            mujoco.mj_forward(self.model, self.data)
        return targets, poses

    # ------------------------------------------------------------------- step

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if not self._position_command:
            raise RuntimeError("reset required before step")
        if abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")
        base = np.asarray(command.base_velocity, dtype=np.float64)
        if base.shape != (self.profile.base.action_dim,) or not np.isfinite(base).all():
            raise ValueError("SourcceyWbcBackend requires a finite planar base velocity")
        velocity_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        desired_base = np.clip(base, -velocity_limits, velocity_limits)
        self._base_command += np.clip(desired_base - self._base_command, -self._base_accel * dt, self._base_accel * dt)

        unknown = set(command.joint_positions) - set(self.profile.controlled_joint_names)
        if unknown:
            raise ValueError(f"joint position command references unknown joints: {sorted(unknown)}")
        for name in self.profile.controlled_joint_names:
            desired = float(command.joint_positions.get(name, self._position_command[name]))
            if not np.isfinite(desired):
                raise ValueError(f"{name} position target is not finite")
            lo, hi = self._limits[name]
            desired = float(np.clip(desired, lo, hi))
            slew = float(
                np.clip(desired - self._position_command[name], -self._joint_step[name], self._joint_step[name])
            )
            self._position_command[name] = float(np.clip(self._position_command[name] + slew, lo, hi))

        unknown_grippers = set(command.grippers) - set(self.profile.arm_names)
        if unknown_grippers:
            raise ValueError(f"gripper command references unknown arms: {sorted(unknown_grippers)}")
        for arm_spec in self.profile.arms:
            gripper_command = command.grippers.get(arm_spec.name)
            if gripper_command is None:
                continue
            if gripper_command.mode != "position":
                raise ValueError("SourcceyWbcBackend only accepts position gripper commands")
            names = self._gripper_joint_names[arm_spec.name]
            if len(gripper_command.values) != len(names):
                raise ValueError(f"{arm_spec.name} gripper command has the wrong DOF")
            for name, value in zip(names, gripper_command.values, strict=True):
                if not np.isfinite(value):
                    raise ValueError(f"{name} gripper target is not finite")
                lo, hi = self._gripper_limits[name]
                desired = float(np.clip(value, lo, hi))
                slew = float(
                    np.clip(desired - self._gripper_command[name], -self._gripper_step[name], self._gripper_step[name])
                )
                self._gripper_command[name] = float(np.clip(self._gripper_command[name] + slew, lo, hi))

        self._apply_control()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._elapsed += self.control_dt
        invalid = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        if not invalid:
            mujoco.mj_forward(self.model, self.data)
        contact = self._contact_sensor.read(self.data)
        info = {
            "contact_force": contact.collision_force,
            "collision": contact.self_collision,
            "collision_measured": True,
            "hand_contact_force": contact.hand_force,
            "elapsed_s": float(self._elapsed),
            "sim_time": float(self.data.time),
            "invalid_physics": bool(invalid),
            "ncon": int(self.data.ncon),
            "model_fingerprint": self.model_fingerprint,
        }
        return self._observation(), bool(invalid), info

    def _apply_control(self) -> None:
        for name, value in self._position_command.items():
            self.data.ctrl[self._joint_act[name]] = self._position_ctrl(name, value)
        for name, value in self._gripper_command.items():
            self.data.ctrl[self._gripper_act[name]] = self._position_ctrl(name, value)
        _, _, yaw = self._base_pose()
        forward = float(self._base_command[0])
        self.data.ctrl[self._base_act[0]] = forward * np.cos(yaw)
        self.data.ctrl[self._base_act[1]] = forward * np.sin(yaw)
        self.data.ctrl[self._base_act[2]] = float(self._base_command[1])

    def _position_ctrl(self, name: str, desired: float) -> float:
        """Realize a position target with the model's bias force feedforward.

        The Sourccey lift carries roughly 280 N while the position-servo
        adapter's ``max_lead`` at its matched ``kp_eff`` can trim only about
        150 N, so a legal lift target would otherwise creep toward its stop.
        Adding the plant's own ``qfrc_bias / kp`` keeps the requested position
        without changing the contract or the adapter's safety envelope.
        """
        actuator = self._joint_act[name] if name in self._joint_act else self._gripper_act[name]
        gain = float(self.model.actuator_gainprm[actuator][0])
        ctrl = float(desired)
        if gain > 0:
            ctrl += float(self.data.qfrc_bias[self._joint_v[name]]) / gain
        if self.model.actuator_ctrllimited[actuator]:
            lo, hi = (float(v) for v in self.model.actuator_ctrlrange[actuator])
            ctrl = float(np.clip(ctrl, lo, hi))
        return ctrl

    # -------------------------------------------------------------- observation

    def _base_pose(self) -> tuple[float, float, float]:
        x, y, yaw = (float(self.data.qpos[index]) for index in self._base_q)
        return x, y, _wrap_angle(yaw)

    def _base_twist(self) -> tuple[float, float]:
        _, _, yaw = self._base_pose()
        vx_world = float(self.data.qvel[self._base_v[0]])
        vy_world = float(self.data.qvel[self._base_v[1]])
        vx_body = np.cos(yaw) * vx_world + np.sin(yaw) * vy_world
        return float(vx_body), float(self.data.qvel[self._base_v[2]])

    def _observation(self) -> WbcObservation:
        torso_group = self.profile.torso
        assert torso_group is not None
        lift_v = self._joint_v["lift"]
        torso = JointGroupObservation(
            name=torso_group.name,
            position=(float(self.data.qpos[self._joint_q["lift"]]),),
            velocity=(float(self.data.qvel[lift_v]),),
            torque=(float(self.data.qfrc_actuator[lift_v]),),
        )
        arms: list[ArmObservation] = []
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            positions = tuple(float(self.data.qpos[self._joint_q[joint.name]]) for joint in group.joints)
            velocities = tuple(float(self.data.qvel[self._joint_v[joint.name]]) for joint in group.joints)
            torques = tuple(float(self.data.qfrc_actuator[self._joint_v[joint.name]]) for joint in group.joints)
            ee_body = self._ee_body[arm_spec.name]
            gripper_names = self._gripper_joint_names[arm_spec.name]
            target = self._target_ee.get(arm_spec.name)
            gripper = GripperObservation(
                position=tuple(float(self.data.qpos[self._joint_q[name]]) for name in gripper_names),
                velocity=tuple(float(self.data.qvel[self._joint_v[name]]) for name in gripper_names),
                force=tuple(float(self.data.qfrc_actuator[self._joint_v[name]]) for name in gripper_names),
                target=tuple(float(self._gripper_command[name]) for name in gripper_names),
            )
            arms.append(
                ArmObservation(
                    name=arm_spec.name,
                    joint_position=positions,
                    joint_velocity=velocities,
                    joint_torque=torques,
                    ee_pose=_pose7(self.data.xpos[ee_body], self.data.xmat[ee_body]),
                    ee_target=target,
                    target_mask=(True, True) if target is not None else (False, False),
                    gripper=gripper,
                )
            )
        return WbcObservation(
            base_pose=self._base_pose(),
            base_twist=self._base_twist(),
            arms=tuple(arms),
            occupancy=None,
            torso=torso,
            base_target=self._target_base,
        )

    # ----------------------------------------------------------------- teacher

    def teacher_action(self) -> np.ndarray:
        """Privileged PD teacher for behavior cloning.

        If a base goal exists, the base drives and settles before the arms move;
        otherwise the arms move immediately. The teacher uses the sampled
        joint-space target and the profile's identified kp/kd/torque limits.
        The grippers hold open.
        """
        if not self._target_q:
            raise RuntimeError("reset required before teacher_action")
        base_velocity = self._teacher_base_velocity()
        lift_joint = self.profile.torso.joints[0]  # type: ignore[union-attr]
        lift_q = float(self.data.qpos[self._joint_q["lift"]])
        lift_qd = float(self.data.qvel[self._joint_v["lift"]])
        lift_torque = float(
            np.clip(
                (lift_joint.kp_eff or 0.0) * (self._lift_target - lift_q) - (lift_joint.kd_eff or 0.0) * lift_qd,
                -lift_joint.torque_limit,
                lift_joint.torque_limit,
            )
        )

        arm_actions: list[ArmAction] = []
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            q = np.asarray([self.data.qpos[self._joint_q[joint.name]] for joint in group.joints])
            qd = np.asarray([self.data.qvel[self._joint_v[joint.name]] for joint in group.joints])
            if self._teacher_phase == "arm":
                q_des = self._target_q[arm_spec.name]
            else:
                q_des = self._start_q[arm_spec.name]
            kp = np.asarray([joint.kp_eff or 0.0 for joint in group.joints], dtype=np.float64)
            kd = np.asarray([joint.kd_eff or 0.0 for joint in group.joints], dtype=np.float64)
            limits = np.asarray([joint.torque_limit for joint in group.joints], dtype=np.float64)
            torque = np.clip(kp * (q_des - q) - kd * qd, -limits, limits)
            gripper = arm_spec.gripper
            assert gripper is not None
            arm_actions.append(
                ArmAction(
                    name=arm_spec.name,
                    joint_torques=tuple(float(v) for v in torque),
                    gripper=GripperCommand(
                        values=self._gripper_open[arm_spec.name],
                        mode=gripper.command_mode,
                    ),
                )
            )
        action = WbcAction(
            base_velocity=(float(base_velocity[0]), float(base_velocity[1])),
            torso_torques=(lift_torque,),
            arms=tuple(arm_actions),
        )
        return np.clip(normalized_action_vector(action, self.profile), -1.0, 1.0).astype(np.float32)

    def _teacher_base_velocity(self) -> np.ndarray:
        x, y, yaw = self._base_pose()
        if self._target_base is None:
            return np.zeros(2, dtype=np.float64)
        dx = self._target_base[0] - x
        dy = self._target_base[1] - y
        distance = float(np.hypot(dx, dy))
        heading_error = _wrap_angle(self._target_base[2] - yaw)
        vx, wz = self._base_twist()
        quiet = abs(vx) < 0.01 and abs(wz) < 0.05
        if self._teacher_phase == "base":
            if (distance <= 0.02 and abs(heading_error) <= 0.03 and quiet) or (self._elapsed >= _BASE_STAGE_TIMEOUT_S):
                self._teacher_phase = "arm"
        max_v, max_w = (float(v) for v in self.profile.base.velocity_limits)
        if distance > 1e-6:
            bearing_error = _wrap_angle(float(np.arctan2(dy, dx)) - yaw)
            forward = float(np.clip(2.0 * distance, 0.0, max_v)) * max(0.0, float(np.cos(bearing_error)))
        else:
            bearing_error = 0.0
            forward = 0.0
        # While far from the goal, steer at the goal (pure pursuit); once there,
        # settle onto the requested final heading.
        steer_error = bearing_error if distance > 0.03 else heading_error
        yaw_rate = float(np.clip(2.0 * steer_error - 0.2 * wz, -max_w, max_w))
        return np.asarray([forward, yaw_rate], dtype=np.float64)

    def close(self) -> None:
        return None


__all__ = ["SourcceyWbcBackend"]
