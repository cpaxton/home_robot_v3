# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Direct-MuJoCo whole-body backend for the RB-Y1 / Galaxea R1 proxy.

The contract base is holonomic ``(vx, vy, yaw_rate)``. The packaged plant has a
free-floating ``base_link`` plus three steer/wheel modules, so this backend maps
the contract velocity onto the module contact velocities and drives the
plant's own steer position servos and wheel velocity servos. Torso, arm, and
gripper channels use the MJCF position actuators and the existing
:class:`~emet.controller.manipulation.torque_adapter.PositionServoAdapter`
loop; a policy torque becomes a bounded joint-position lead before it reaches
this class.

Targets and observations use the world/odometry frame because
``rby1_sim_profile`` declares ``target_frame="world"`` for both arms. The base
starts at the origin, so world and odometry coincide for an episode.

Model preparation (in memory; the packaged XML is not modified):

- A worldbody floor plane is inserted when the MJCF has none.
- The packaged wheel geoms are visual-only, and the base hull's collision mesh
  reaches the ground. Without wheel collision proxies the robot would rest on
  the hull and the wheel actuators could not move it. Cylinder proxies with the
  module's rolling radius are added to each wheel body. Their contact masks keep
  them from colliding with the base hull while still contacting the floor.

Deliberate limitations:

- Wheel friction and the retained velocity servos make the measured base
  velocity differ from the kinematic module command. The observation reports
  measured body-frame twist, not the command. The wheel servos are limited to
  +/-6 actuator units with a gear of 3, so the plant saturates near 0.19 m/s
  even though the profile declares 0.4 m/s.
- Steer scrubbing on a rigid three-wheel platform can lift the base a couple of
  centimetres during a large in-place steer change. Slower commands reduce it.
- Collision evidence is measured from the proxy's MuJoCo contacts. Wheel-floor
  contacts are support; chassis/arm contacts with the scene and non-gripper
  self-collisions are reported as collisions. The proxy has limited upper-body
  collision geometry, so this measures what the model contains, not hardware.
- Arm targets are sampled by FK at the sampled base goal, then both are
  restored to the reset state. A teacher can therefore drive the base first and
  still reach the world-frame arm targets, but a policy that moves the base
  differently sees a fixed world target.
"""

from __future__ import annotations

import hashlib
from typing import Any
from xml.etree import ElementTree

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
    JointSpec,
    WbcAction,
    WbcObservation,
    normalized_action_vector,
)
from emet.controller.manipulation.wbc_profiles import rby1_sim_profile
from emet.rl.manipulation.wbc_contact import MuJoCoContactSensor
from emet.utils.assets import get_robot_mjcf_path

_TIMESTEP_S = 0.002
_WHEEL_JOINTS = tuple(f"wheel_motor_joint{index}" for index in range(1, 4))
_STEER_JOINTS = tuple(f"steer_motor_joint{index}" for index in range(1, 4))
_WHEEL_BODIES = tuple(f"wheel_motor_link{index}" for index in range(1, 4))
_STEER_BODIES = tuple(f"steer_motor_link{index}" for index in range(1, 4))
_WHEEL_ACTUATORS = tuple(f"wheel{index}" for index in range(1, 4))
_STEER_ACTUATORS = tuple(f"steer{index}" for index in range(1, 4))
_BASE_BODY = "base_link"

# The wheel body hangs one rolling radius below its steer module. The prompt's
# ~0.09525 m radius is consistent with the packaged joint offset; we read the
# offset from the MJCF and only use this value as a consistency check.
_EXPECTED_WHEEL_RADIUS_M = 0.09525
_WHEEL_HALF_WIDTH_M = 0.03

# Collision bitmasks: floor collides with the hull (1) and the wheel proxies
# (4); the wheel proxies do not collide with the hull, which they overlap.
_FLOOR_CONTYPE = 5
_FLOOR_CONAFFINITY = 5
_WHEEL_CONTYPE = 4
_WHEEL_CONAFFINITY = 4

_STEER_LIMIT_RAD = 1.57
_STEER_RATE_RAD_S = 4.0
_GRIPPER_SPEED_M_S = 0.05
_GRIPPER_OPEN_M = 0.04
_MIN_MODULE_SPEED_M_S = 1e-4

# Base-goal envelope: a short forward cone keeps a reach-after-drive episode
# reachable for the 6-DOF arms after the base settles.
_BASE_GOAL_DISTANCE_M = (0.05, 0.15)
_BASE_GOAL_BEARING_RAD = 0.35
_BASE_GOAL_YAW_RAD = 0.15
_BASE_STAGE_TIMEOUT_S = 3.0
_BASE_SETTLE_DISTANCE_M = 0.02
_BASE_SETTLE_HEADING_RAD = 0.05
_BASE_SETTLE_SPEED_M_S = 0.02
_BASE_SETTLE_YAW_RATE_RAD_S = 0.05
_BASE_POSITION_GAIN = 2.0
_BASE_YAW_GAIN = 2.0
_BASE_YAW_DAMPING = 0.2

# Arm-target sampling is a joint-space perturbation whose FK pose becomes the
# world target. A minimum displacement keeps an "arm goal" from being a no-op.
_ARM_TARGET_RANGE_RAD = 0.15
_MIN_EE_DISPLACEMENT_M = 0.03
_TARGET_SAMPLE_ATTEMPTS = 64

# The profile omits acceleration limits, so the plant owns a conservative
# executor-side envelope. These are simulator command-shaping limits, not
# hardware identification.
_BASE_ACCELERATION = np.array([1.5, 1.5, 2.0], dtype=np.float64)


def _wrap_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def _pose7(position: np.ndarray, rotation: np.ndarray) -> tuple[float, ...]:
    """Return ``(xyz, quat_xyzw)`` from a MuJoCo body pose."""
    quat = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat()
    return tuple(float(v) for v in np.r_[position, quat])


def _element_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    identifier = int(mujoco.mj_name2id(model, objtype, name))
    if identifier < 0:
        raise ValueError(f"MJCF does not contain {objtype.name} {name!r}")
    return identifier


def _add_collision_floor(root: ElementTree.Element) -> None:
    """Insert a worldbody floor plane when the MJCF has none."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if any(geom.get("type") == "plane" for geom in worldbody.findall("geom")):
        return
    worldbody.insert(
        0,
        ElementTree.Element(
            "geom",
            {
                "name": "emet_floor",
                "type": "plane",
                "size": "0 0 0.05",
                "rgba": "0.55 0.55 0.55 1",
                "contype": str(_FLOOR_CONTYPE),
                "conaffinity": str(_FLOOR_CONAFFINITY),
            },
        ),
    )


def _add_wheel_collision_proxies(root: ElementTree.Element, wheel_radius: float) -> None:
    """Give the visual-only wheel meshes a floor-contacting cylinder proxy.

    The proxies overlap the base hull, so their masks only contact the floor.
    """
    for index, body_name in enumerate(_WHEEL_BODIES, start=1):
        bodies = root.findall(f".//body[@name='{body_name}']")
        if len(bodies) != 1:
            raise ValueError(f"MJCF must contain exactly one body {body_name!r}")
        body = bodies[0]
        name = f"emet_wheel_collision{index}"
        if any(geom.get("name") == name for geom in body.findall("geom")):
            continue
        ElementTree.SubElement(
            body,
            "geom",
            {
                "name": name,
                "type": "cylinder",
                "size": f"{wheel_radius} {_WHEEL_HALF_WIDTH_M}",
                # MuJoCo cylinders point along +z; rotate the axis onto +y, the
                # wheel joint axis.
                "quat": "0.7071067811865476 0.7071067811865476 0 0",
                "friction": "1.0 0.02 0.001",
                "rgba": "0 0 0 0",
                "contype": str(_WHEEL_CONTYPE),
                "conaffinity": str(_WHEEL_CONAFFINITY),
            },
        )


def _load_model() -> tuple[mujoco.MjModel, str]:
    """Compile the packaged R1 MJCF with floor and wheel contact proxies."""
    mjcf_path = get_robot_mjcf_path("rby1")
    if mjcf_path is None or not mjcf_path.is_file():
        raise RuntimeError("RB-Y1 MJCF not found; install the emet package data or use a source checkout")
    root = ElementTree.parse(mjcf_path).getroot()
    _add_collision_floor(root)
    _add_wheel_collision_proxies(root, _EXPECTED_WHEEL_RADIUS_M)
    xml = ElementTree.tostring(root, encoding="unicode")
    mesh_dir = mjcf_path.parent / "meshes"
    assets = {path.name: path.read_bytes() for path in mesh_dir.iterdir() if path.is_file()}
    model = mujoco.MjModel.from_xml_string(xml, assets)
    model.opt.timestep = _TIMESTEP_S
    return model, hashlib.sha256(xml.encode("utf-8")).hexdigest()


class Rby1WbcBackend:
    """Plant-side :class:`~emet.rl.manipulation.wbc_env.WbcBackend` for RB-Y1."""

    def __init__(
        self,
        profile: EmbodimentProfile | None = None,
        seed: int = 0,
        control_dt: float = 0.05,
        arm_goal_probability: float = 1.0,
        base_goal_probability: float = 0.5,
        target_range: float = 0.15,
        base_goal_mode: str = "forward",
        active_arms: tuple[str, ...] | None = None,
    ) -> None:
        self.profile = rby1_sim_profile() if profile is None else profile
        if self.profile.meta("robot") != "rby1":
            raise ValueError("Rby1WbcBackend requires an rby1 embodiment profile")
        if self.profile.base.kind != "holonomic" or self.profile.base.action_dim != 3:
            raise ValueError("Rby1WbcBackend requires a holonomic three-axis base")
        if tuple(arm.name for arm in self.profile.arms) != ("left", "right"):
            raise ValueError("Rby1WbcBackend requires left and right arm profiles")
        if active_arms is not None:
            if not active_arms or len(set(active_arms)) != len(active_arms):
                raise ValueError("active_arms must be non-empty and unique")
            unknown_arms = set(active_arms) - {"left", "right"}
            if unknown_arms:
                raise ValueError(f"unknown active arms: {sorted(unknown_arms)}")
            self.active_arms = tuple(active_arms)
        else:
            self.active_arms = ("left", "right")
        if self.profile.torso is None:
            raise ValueError("Rby1WbcBackend requires the four-joint torso group")
        if any(arm.target_frame != "world" for arm in self.profile.arms):
            raise ValueError("Rby1WbcBackend emits world-frame EE poses; use target_frame='world'")
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

        self.control_dt = float(control_dt)
        self.arm_goal_probability = float(arm_goal_probability)
        self.base_goal_probability = float(base_goal_probability)
        self.base_goal_mode = base_goal_mode
        self.target_range = float(target_range)
        self._arm_target_range = float(target_range) * (_ARM_TARGET_RANGE_RAD / 0.15) if target_range > 0 else 0.0
        self.rng = np.random.default_rng(seed)
        self._elapsed = 0.0

        self.model, self.model_fingerprint = _load_model()
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(self.control_dt / _TIMESTEP_S))
        if self.substeps < 1 or abs(self.substeps * _TIMESTEP_S - self.control_dt) > 1e-9:
            raise ValueError("control_dt must be an integer multiple of the 0.002 s MJCF timestep")

        self._base_q = int(self.model.joint("base_freejoint").qposadr[0])
        self._base_v = int(self.model.joint("base_freejoint").dofadr[0])
        self._base_body = _element_id(self.model, mujoco.mjtObj.mjOBJ_BODY, _BASE_BODY)

        self._steer_q = np.asarray([int(self.model.joint(name).qposadr[0]) for name in _STEER_JOINTS], dtype=int)
        self._steer_act = np.asarray(
            [_element_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in _STEER_ACTUATORS], dtype=int
        )
        self._wheel_act = np.asarray(
            [_element_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in _WHEEL_ACTUATORS], dtype=int
        )
        steer_bodies = np.asarray(
            [_element_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _STEER_BODIES], dtype=int
        )
        wheel_bodies = np.asarray(
            [_element_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in _WHEEL_BODIES], dtype=int
        )
        self._module_xy = np.asarray(self.model.body_pos[steer_bodies, :2], dtype=np.float64)
        wheel_offsets = np.asarray(self.model.body_pos[wheel_bodies, 2], dtype=np.float64)
        wheel_center_z = float(self.model.body_pos[steer_bodies[0], 2] + wheel_offsets[0])
        self._wheel_radius = float(-wheel_offsets[0])
        if self._wheel_radius <= 0 or not np.allclose(wheel_offsets, -self._wheel_radius):
            raise ValueError("unexpected RB-Y1 wheel geometry in the MJCF")
        if abs(self._wheel_radius - _EXPECTED_WHEEL_RADIUS_M) > 1e-6:
            raise ValueError(
                f"MJCF wheel radius {self._wheel_radius:.6f} m does not match the expected "
                f"{_EXPECTED_WHEEL_RADIUS_M:.6f} m"
            )
        self._base_rest_z = self._wheel_radius - wheel_center_z
        self._wheel_gear = float(self.model.actuator_gear[self._wheel_act[0], 0])
        if self._wheel_gear <= 0:
            raise ValueError("RB-Y1 wheel actuators must declare a positive gear")
        self._wheel_ctrl_low = np.asarray(self.model.actuator_ctrlrange[self._wheel_act, 0], dtype=np.float64)
        self._wheel_ctrl_high = np.asarray(self.model.actuator_ctrlrange[self._wheel_act, 1], dtype=np.float64)

        # Bind every contract joint to its qpos/qvel/actuator address and to the
        # intersection of the profile and MJCF position limits.
        self._joint_q: dict[str, int] = {}
        self._joint_v: dict[str, int] = {}
        self._joint_act: dict[str, int] = {}
        self._joint_specs: dict[str, JointSpec] = {}
        self._joint_limits: dict[str, tuple[float, float]] = {}
        self._joint_step: dict[str, float] = {}
        for group in self.model_groups():
            for joint in group.joints:
                self._bind_joint(joint)

        self._gripper_q: dict[str, int] = {}
        self._gripper_v: dict[str, int] = {}
        self._gripper_act: dict[str, int] = {}
        self._gripper_limits: dict[str, tuple[float, float]] = {}
        self._gripper_step: dict[str, float] = {}
        self._gripper_open: dict[str, float] = {}
        self._gripper_names: dict[str, tuple[str, ...]] = {}
        for arm_spec in self.profile.arms:
            gripper = arm_spec.gripper
            assert gripper is not None
            self._gripper_names[arm_spec.name] = gripper.joint_names
            command_limits = gripper.command_limits
            opens: list[float] = []
            for index, name in enumerate(gripper.joint_names):
                joint_id = _element_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                model_lo, model_hi = self._model_joint_limits(joint_id)
                if command_limits is not None:
                    model_lo = max(model_lo, float(command_limits[index][0]))
                    model_hi = min(model_hi, float(command_limits[index][1]))
                if model_lo >= model_hi:
                    raise ValueError(f"{name} command limits do not intersect the MJCF joint range")
                self._gripper_q[name] = int(self.model.jnt_qposadr[joint_id])
                self._gripper_v[name] = int(self.model.jnt_dofadr[joint_id])
                self._gripper_act[name] = self._actuator_for_joint(joint_id)
                self._gripper_limits[name] = (model_lo, model_hi)
                self._gripper_step[name] = _GRIPPER_SPEED_M_S * self.control_dt
                opens.append(float(np.clip(_GRIPPER_OPEN_M, model_lo, model_hi)))
            self._gripper_open[arm_spec.name] = tuple(opens)

        self._ee_body = {
            arm.name: _element_id(self.model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_link) for arm in self.profile.arms
        }
        self._contact_sensor = MuJoCoContactSensor(
            self.model,
            base_body=_BASE_BODY,
            ee_body=self.profile.arms[0].ee_link,
            hand_bodies=tuple(arm.ee_link for arm in self.profile.arms),
            finger_bodies=(
                "left_gripper_finger_link1",
                "left_gripper_finger_link2",
                "right_gripper_finger_link1",
                "right_gripper_finger_link2",
            ),
            support_bodies=_WHEEL_BODIES,
        )
        self.collision_measured = True

        velocity_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        if self.profile.base.acceleration_limits is None:
            self._base_acceleration = np.minimum(_BASE_ACCELERATION, 4.0 * velocity_limits)
        else:
            self._base_acceleration = np.asarray(self.profile.base.acceleration_limits, dtype=np.float64)
        self._base_step = self._base_acceleration * self.control_dt
        self._steer_step = _STEER_RATE_RAD_S * self.control_dt

        self._base_command = np.zeros(3, dtype=np.float64)
        self._steer_command = np.zeros(3, dtype=np.float64)
        self._wheel_command = np.zeros(3, dtype=np.float64)
        self._joint_command: dict[str, float] = {}
        self._gripper_command: dict[str, float] = {}
        self._torso_target = np.zeros(len(self.profile.torso.joints), dtype=np.float64)
        self._start_q: dict[str, np.ndarray] = {}
        self._target_q: dict[str, np.ndarray] = {}
        self._target_ee: dict[str, tuple[float, ...] | None] = {}
        self._target_base: tuple[float, float, float] | None = None
        self._teacher_phase = "arm"
        self._ready = False
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ setup
    def model_groups(self):
        """Yield the controlled joint groups in profile order."""
        assert self.profile.torso is not None
        yield self.profile.torso
        for arm_spec in self.profile.arms:
            yield self.profile.joint_group(arm_spec.joint_group)

    def _model_joint_limits(self, joint_id: int) -> tuple[float, float]:
        if not self.model.jnt_limited[joint_id]:
            return -np.inf, np.inf
        low, high = self.model.jnt_range[joint_id]
        return float(low), float(high)

    def _actuator_for_joint(self, joint_id: int) -> int:
        for actuator in range(self.model.nu):
            if self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            if int(self.model.actuator_trnid[actuator, 0]) == joint_id:
                return actuator
        raise ValueError(f"MJCF joint id {joint_id} has no joint actuator")

    def _bind_joint(self, joint: JointSpec) -> None:
        joint_id = _element_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint.name)
        model_lo, model_hi = self._model_joint_limits(joint_id)
        low = max(model_lo, float(joint.position_limit[0]))
        high = min(model_hi, float(joint.position_limit[1]))
        if low >= high:
            raise ValueError(f"{joint.name} profile limits do not intersect the MJCF joint range")
        self._joint_q[joint.name] = int(self.model.jnt_qposadr[joint_id])
        self._joint_v[joint.name] = int(self.model.jnt_dofadr[joint_id])
        self._joint_act[joint.name] = self._actuator_for_joint(joint_id)
        self._joint_specs[joint.name] = joint
        self._joint_limits[joint.name] = (low, high)
        self._joint_step[joint.name] = float(joint.velocity_limit) * self.control_dt

    # ----------------------------------------------------------- WBC protocol
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> WbcObservation:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._base_q : self._base_q + 7] = (0.0, 0.0, self._base_rest_z, 1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0

        for group in self.model_groups():
            for joint in group.joints:
                low, high = self._joint_limits[joint.name]
                # The packaged home posture is collision-free. Reset near it
                # instead of sampling the full joint range: a 12-DOF arm at a
                # random extreme is a far harder imitation target.
                self.data.qpos[self._joint_q[joint.name]] = float(np.clip(self.rng.uniform(-0.05, 0.05), low, high))
        for name, value in self._gripper_open.items():
            for joint_name, joint_value in zip(self._gripper_names[name], value, strict=True):
                self.data.qpos[self._gripper_q[joint_name]] = joint_value
        mujoco.mj_forward(self.model, self.data)

        self._start_q = {
            arm_spec.name: np.asarray(
                [
                    self.data.qpos[self._joint_q[joint.name]]
                    for joint in self.profile.joint_group(arm_spec.joint_group).joints
                ],
                dtype=np.float64,
            )
            for arm_spec in self.profile.arms
        }
        self._target_base = self._sample_base_target()
        if self.rng.random() < self.arm_goal_probability:
            self._target_q, self._target_ee = self._sample_arm_targets(self._target_base)
        else:
            self._target_q = {name: q.copy() for name, q in self._start_q.items()}
            self._target_ee = {arm_spec.name: None for arm_spec in self.profile.arms}
        self._joint_command = {name: float(self.data.qpos[address]) for name, address in self._joint_q.items()}
        self._gripper_command = {name: float(self.data.qpos[address]) for name, address in self._gripper_q.items()}
        self._torso_target = np.asarray(
            [self.data.qpos[self._joint_q[joint.name]] for joint in self.profile.torso.joints],  # type: ignore[union-attr]
            dtype=np.float64,
        )
        self._base_command = np.zeros(3, dtype=np.float64)
        self._steer_command = np.asarray([self.data.qpos[address] for address in self._steer_q], dtype=np.float64)
        self._wheel_command = np.zeros(3, dtype=np.float64)
        self._teacher_phase = "base" if self._target_base is not None else "arm"
        self._elapsed = 0.0
        self._ready = True
        self._apply_control()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, command: ServoCommand, dt: float) -> tuple[WbcObservation, bool, dict[str, Any]]:
        if not self._ready:
            raise RuntimeError("reset required")
        if abs(float(dt) - self.control_dt) > 1e-9:
            raise ValueError("backend step dt does not match its configured control period")

        base = np.asarray(command.base_velocity, dtype=np.float64).reshape(-1)
        if base.shape != (3,) or not np.isfinite(base).all():
            raise ValueError("Rby1WbcBackend requires a finite holonomic three-axis base velocity")
        desired_base = np.clip(base, -np.asarray(self.profile.base.velocity_limits), self.profile.base.velocity_limits)
        self._base_command += np.clip(desired_base - self._base_command, -self._base_step, self._base_step)
        self._update_module_commands()

        known = set(self._joint_q) | set(self._gripper_q)
        unknown = set(command.joint_positions) - known
        if unknown:
            raise ValueError(f"joint position command references unknown joints: {sorted(unknown)}")
        for name in self._joint_q:
            if name not in command.joint_positions:
                continue
            desired = float(command.joint_positions[name])
            if not np.isfinite(desired):
                raise ValueError(f"{name} position target is not finite")
            low, high = self._joint_limits[name]
            desired = float(np.clip(desired, low, high))
            slew = float(np.clip(desired - self._joint_command[name], -self._joint_step[name], self._joint_step[name]))
            self._joint_command[name] = float(np.clip(self._joint_command[name] + slew, low, high))
        # A caller may send gripper positions through either channel; the
        # command.grippers channel wins when both are present.
        for name in self._gripper_q:
            if name not in command.joint_positions:
                continue
            desired = float(command.joint_positions[name])
            if not np.isfinite(desired):
                raise ValueError(f"{name} position target is not finite")
            low, high = self._gripper_limits[name]
            desired = float(np.clip(desired, low, high))
            slew = float(
                np.clip(desired - self._gripper_command[name], -self._gripper_step[name], self._gripper_step[name])
            )
            self._gripper_command[name] = float(np.clip(self._gripper_command[name] + slew, low, high))

        unknown_grippers = set(command.grippers) - set(self.profile.arm_names)
        if unknown_grippers:
            raise ValueError(f"gripper command references unknown arms: {sorted(unknown_grippers)}")
        for arm_spec in self.profile.arms:
            gripper_command = command.grippers.get(arm_spec.name)
            if gripper_command is None:
                continue
            if gripper_command.mode != "position":
                raise ValueError("Rby1WbcBackend only accepts position gripper commands")
            names = self._gripper_names[arm_spec.name]
            if len(gripper_command.values) != len(names):
                raise ValueError(f"{arm_spec.name} gripper command has the wrong DOF")
            for name, value in zip(names, gripper_command.values, strict=True):
                if not np.isfinite(value):
                    raise ValueError(f"{name} gripper target is not finite")
                low, high = self._gripper_limits[name]
                desired = float(np.clip(value, low, high))
                slew = float(
                    np.clip(desired - self._gripper_command[name], -self._gripper_step[name], self._gripper_step[name])
                )
                self._gripper_command[name] = float(np.clip(self._gripper_command[name] + slew, low, high))

        safe_qpos = self.data.qpos.copy()
        safe_qvel = self.data.qvel.copy()
        safe_time = float(self.data.time)
        self._apply_control()
        invalid = False
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
                self.data.qpos[:] = safe_qpos
                self.data.qvel[:] = safe_qvel
                self.data.time = safe_time
                invalid = True
                break
            safe_qpos = self.data.qpos.copy()
            safe_qvel = self.data.qvel.copy()
            safe_time = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)
        contact = self._contact_sensor.read(self.data)
        self._elapsed += self.control_dt
        info = {
            "elapsed_s": float(self._elapsed),
            "sim_time": float(self.data.time),
            "invalid_physics": bool(invalid),
            "collision": contact.self_collision,
            "collision_measured": True,
            "contact_force": contact.collision_force,
            "hand_contact_force": contact.hand_force,
            "ncon": int(self.data.ncon),
            "model_fingerprint": self.model_fingerprint,
            "base_command": tuple(float(v) for v in self._base_command),
            "steer_command": tuple(float(v) for v in self._steer_command),
            "wheel_command": tuple(float(v) for v in self._wheel_command),
        }
        return self._observation(), bool(invalid), info

    def close(self) -> None:
        return None

    # --------------------------------------------------------- command mapping
    def _update_module_commands(self) -> None:
        """Map body-frame base velocity to the three swerve modules.

        Module ``i`` at base-frame ``(x_i, y_i)`` needs contact velocity
        ``(vx - wz*y_i, vy + wz*x_i)``. The steer target is its direction and
        the wheel speed is its magnitude over the rolling radius. A direction
        can also be represented by steering half a turn and reversing the
        wheel; the candidate closest to the current steer angle wins, which
        keeps the modules inside their +/-1.57 rad range without large sweeps.
        """
        vx, vy, wz = (float(v) for v in self._base_command)
        for index, (x, y) in enumerate(self._module_xy):
            module_vx = vx - wz * y
            module_vy = vy + wz * x
            speed = float(np.hypot(module_vx, module_vy))
            if speed < _MIN_MODULE_SPEED_M_S:
                steer_target = float(self._steer_command[index])
                wheel_speed = 0.0
            else:
                angle = float(np.arctan2(module_vy, module_vx))
                candidates = (
                    (angle, speed),
                    (_wrap_angle(angle - np.pi), -speed),
                    (_wrap_angle(angle + np.pi), -speed),
                )
                steer_target, wheel_speed = min(
                    candidates,
                    key=lambda candidate: abs(_wrap_angle(candidate[0] - self._steer_command[index])),
                )
                steer_target = float(np.clip(steer_target, -_STEER_LIMIT_RAD, _STEER_LIMIT_RAD))
            slew = float(np.clip(steer_target - self._steer_command[index], -self._steer_step, self._steer_step))
            self._steer_command[index] = float(
                np.clip(self._steer_command[index] + slew, -_STEER_LIMIT_RAD, _STEER_LIMIT_RAD)
            )
            # MuJoCo velocity actuators are commanded in actuator space, so the
            # joint speed is scaled by the actuator gear (3 for this MJCF).
            wheel_ctrl = self._wheel_gear * wheel_speed / self._wheel_radius
            self._wheel_command[index] = float(
                np.clip(wheel_ctrl, self._wheel_ctrl_low[index], self._wheel_ctrl_high[index])
            )

    def _apply_control(self) -> None:
        self.data.ctrl[self._steer_act] = self._steer_command
        self.data.ctrl[self._wheel_act] = self._wheel_command
        for name, actuator in self._joint_act.items():
            self.data.ctrl[actuator] = self._joint_command[name]
        for name, actuator in self._gripper_act.items():
            self.data.ctrl[actuator] = self._gripper_command[name]

    # ------------------------------------------------------------- observation
    def _base_pose(self) -> tuple[float, float, float]:
        position = self.data.xpos[self._base_body]
        rotation = self.data.xmat[self._base_body].reshape(3, 3)
        return float(position[0]), float(position[1]), float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _base_twist(self) -> tuple[float, float, float]:
        """Measured base velocity in the body frame: ``(vx, vy, yaw_rate)``."""
        x, _, yaw = self._base_pose()
        qvel = self.data.qvel[self._base_v : self._base_v + 3]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vx = cos_yaw * float(qvel[0]) + sin_yaw * float(qvel[1])
        vy = -sin_yaw * float(qvel[0]) + cos_yaw * float(qvel[1])
        return float(vx), float(vy), float(self.data.qvel[self._base_v + 5])

    def _observation(self) -> WbcObservation:
        torso_group = self.profile.torso
        assert torso_group is not None
        torso = JointGroupObservation(
            name=torso_group.name,
            position=tuple(float(self.data.qpos[self._joint_q[joint.name]]) for joint in torso_group.joints),
            velocity=tuple(float(self.data.qvel[self._joint_v[joint.name]]) for joint in torso_group.joints),
            torque=tuple(float(self.data.qfrc_actuator[self._joint_v[joint.name]]) for joint in torso_group.joints),
            target=tuple(float(v) for v in self._torso_target),
            target_mask=(True,) * len(torso_group.joints),
        )
        arms: list[ArmObservation] = []
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            names = self._gripper_names[arm_spec.name]
            ee_body = self._ee_body[arm_spec.name]
            target = self._target_ee.get(arm_spec.name)
            arms.append(
                ArmObservation(
                    name=arm_spec.name,
                    joint_position=tuple(float(self.data.qpos[self._joint_q[joint.name]]) for joint in group.joints),
                    joint_velocity=tuple(float(self.data.qvel[self._joint_v[joint.name]]) for joint in group.joints),
                    joint_torque=tuple(
                        float(self.data.qfrc_actuator[self._joint_v[joint.name]]) for joint in group.joints
                    ),
                    ee_pose=_pose7(self.data.xpos[ee_body], self.data.xmat[ee_body]),
                    ee_target=target,
                    target_mask=(True, True) if target is not None else (False, False),
                    gripper=GripperObservation(
                        position=tuple(float(self.data.qpos[self._gripper_q[name]]) for name in names),
                        velocity=tuple(float(self.data.qvel[self._gripper_v[name]]) for name in names),
                        force=tuple(float(self.data.qfrc_actuator[self._gripper_v[name]]) for name in names),
                    ),
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

    # ----------------------------------------------------------- teacher/BC
    def teacher_action(self) -> np.ndarray:
        """Privileged normalized action for behavior cloning.

        With a base goal, the base P controller drives and settles before the
        arms move. The arms track the sampled joint-space targets with the
        profile's identified kp/kd/torque limits; the torso holds its reset
        posture and the grippers hold open. The teacher is deterministic and
        never consumes the RNG or advances physics.
        """
        if not self._ready:
            raise RuntimeError("reset required")
        if self._teacher_phase == "base" and self._base_goal_reached():
            self._teacher_phase = "arm"
        base_velocity = self._teacher_base_velocity()

        torso_joints = self.profile.torso.joints  # type: ignore[union-attr]
        torso_q = np.asarray([self.data.qpos[self._joint_q[joint.name]] for joint in torso_joints], dtype=np.float64)
        torso_qd = np.asarray([self.data.qvel[self._joint_v[joint.name]] for joint in torso_joints], dtype=np.float64)
        torso_torque = self._pd_torque(torso_joints, self._torso_target, torso_q, torso_qd)

        arm_actions: list[ArmAction] = []
        gripper_specs = {arm.name: arm.gripper for arm in self.profile.arms}
        for arm_spec in self.profile.arms:
            group = self.profile.joint_group(arm_spec.joint_group)
            q = np.asarray([self.data.qpos[self._joint_q[joint.name]] for joint in group.joints], dtype=np.float64)
            qd = np.asarray([self.data.qvel[self._joint_v[joint.name]] for joint in group.joints], dtype=np.float64)
            q_des = self._target_q[arm_spec.name] if self._teacher_phase == "arm" else self._start_q[arm_spec.name]
            torque = self._pd_torque(group.joints, q_des, q, qd)
            gripper = gripper_specs[arm_spec.name]
            assert gripper is not None
            arm_actions.append(
                ArmAction(
                    name=arm_spec.name,
                    joint_torques=tuple(float(v) for v in torque),
                    gripper=GripperCommand(values=self._gripper_open[arm_spec.name], mode=gripper.command_mode),
                )
            )
        action = WbcAction(
            base_velocity=tuple(float(v) for v in base_velocity),
            arms=tuple(arm_actions),
            torso_torques=tuple(float(v) for v in torso_torque),
        )
        return normalized_action_vector(action, self.profile).astype(np.float32)

    @staticmethod
    def _pd_torque(
        joints: tuple[JointSpec, ...],
        target: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        kp = np.asarray([float(joint.kp_eff or 0.0) for joint in joints], dtype=np.float64)
        kd = np.asarray([float(joint.kd_eff or 0.0) for joint in joints], dtype=np.float64)
        limits = np.asarray([joint.torque_limit for joint in joints], dtype=np.float64)
        return np.clip(kp * (target - position) - kd * velocity, -limits, limits)

    def _base_goal_reached(self) -> bool:
        if self._target_base is None:
            return True
        if self._elapsed >= _BASE_STAGE_TIMEOUT_S:
            return True
        x, y, yaw = self._base_pose()
        distance = float(np.hypot(self._target_base[0] - x, self._target_base[1] - y))
        heading = abs(_wrap_angle(self._target_base[2] - yaw))
        vx, vy, wz = self._base_twist()
        return bool(
            distance <= _BASE_SETTLE_DISTANCE_M
            and heading <= _BASE_SETTLE_HEADING_RAD
            and abs(vx) <= _BASE_SETTLE_SPEED_M_S
            and abs(vy) <= _BASE_SETTLE_SPEED_M_S
            and abs(wz) <= _BASE_SETTLE_YAW_RATE_RAD_S
        )

    def _teacher_base_velocity(self) -> np.ndarray:
        if self._target_base is None:
            return np.zeros(3, dtype=np.float64)
        x, y, yaw = self._base_pose()
        dx = self._target_base[0] - x
        dy = self._target_base[1] - y
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        heading_error = _wrap_angle(self._target_base[2] - yaw)
        velocity_limits = np.asarray(self.profile.base.velocity_limits, dtype=np.float64)
        _, _, wz = self._base_twist()
        vx = float(np.clip(_BASE_POSITION_GAIN * forward, -velocity_limits[0], velocity_limits[0]))
        vy = float(np.clip(_BASE_POSITION_GAIN * lateral, -velocity_limits[1], velocity_limits[1]))
        yaw_rate = float(
            np.clip(
                _BASE_YAW_GAIN * heading_error - _BASE_YAW_DAMPING * wz,
                -velocity_limits[2],
                velocity_limits[2],
            )
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float64)

    # ------------------------------------------------------------- sampling
    def _sample_base_target(self) -> tuple[float, float, float] | None:
        if self.rng.random() >= self.base_goal_probability:
            return None
        x, y, yaw = self._base_pose()
        if self.base_goal_mode == "forward":
            distance = float(self.rng.uniform(*_BASE_GOAL_DISTANCE_M))
            bearing = float(self.rng.uniform(-_BASE_GOAL_BEARING_RAD, _BASE_GOAL_BEARING_RAD))
            heading = yaw + bearing
            target_x = x + distance * np.cos(heading)
            target_y = y + distance * np.sin(heading)
            target_yaw = yaw + self.rng.uniform(-_BASE_GOAL_YAW_RAD, _BASE_GOAL_YAW_RAD)
        else:
            target_x = x + self.rng.uniform(-0.15, 0.15)
            target_y = y + self.rng.uniform(-0.15, 0.15)
            target_yaw = yaw + self.rng.uniform(-0.3, 0.3)
        return (float(target_x), float(target_y), float(_wrap_angle(target_yaw)))

    def _sample_arm_targets(
        self,
        base_target: tuple[float, float, float] | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, ...] | None]]:
        """Sample FK targets at the base goal, then restore the reset state.

        Sampling at the goal keeps a world-frame target reachable after the
        teacher's base stage. Targets are joint-space perturbations of the
        reset posture, so the model's own FK defines reachability.
        """
        saved_qpos = self.data.qpos.copy()
        if base_target is not None:
            self.data.qpos[self._base_q] = base_target[0]
            self.data.qpos[self._base_q + 1] = base_target[1]
            self.data.qpos[self._base_q + 2] = self._base_rest_z
            half = 0.5 * base_target[2]
            self.data.qpos[self._base_q + 3 : self._base_q + 7] = (np.cos(half), 0.0, 0.0, np.sin(half))
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
                ee_body = self._ee_body[arm_spec.name]
                low = np.asarray([self._joint_limits[joint.name][0] for joint in group.joints], dtype=np.float64)
                high = np.asarray([self._joint_limits[joint.name][1] for joint in group.joints], dtype=np.float64)
                q0 = np.asarray([self.data.qpos[self._joint_q[joint.name]] for joint in group.joints], dtype=np.float64)
                ee0 = self.data.xpos[ee_body].copy()
                best_q = q0.copy()
                best_pose = _pose7(self.data.xpos[ee_body], self.data.xmat[ee_body])
                for _ in range(_TARGET_SAMPLE_ATTEMPTS):
                    sample = np.clip(
                        q0 + self.rng.uniform(-self._arm_target_range, self._arm_target_range, q0.size), low, high
                    )
                    for joint, value in zip(group.joints, sample, strict=True):
                        self.data.qpos[self._joint_q[joint.name]] = value
                    mujoco.mj_forward(self.model, self.data)
                    best_q = sample
                    best_pose = _pose7(self.data.xpos[ee_body], self.data.xmat[ee_body])
                    if np.linalg.norm(self.data.xpos[ee_body] - ee0) >= _MIN_EE_DISPLACEMENT_M:
                        break
                targets[arm_spec.name] = best_q.copy()
                poses[arm_spec.name] = best_pose
                for joint, value in zip(group.joints, q0, strict=True):
                    self.data.qpos[self._joint_q[joint.name]] = value
        finally:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
        return targets, poses


__all__ = ["Rby1WbcBackend"]
