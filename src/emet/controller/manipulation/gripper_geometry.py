# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Encoder-conditioned whole-gripper volumes over unmeasured joint intervals."""

import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.interval_geometry import subtree_geometry_intervals
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.native_scene import model_signature
from emet.controller.manipulation.observed_effects import RobotPoseSample
from emet.motion.mujoco_collision import body_subtree
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.synchronized_capture import robot_kinematic_signature


def _geometry_signature(model):
    return hashlib.sha256(
        model_signature(model).encode()
        + robot_kinematic_signature(model).encode()
        + model.geom_aabb.tobytes()
        + model.jnt_limited.tobytes()
    ).hexdigest()


def _frozen(array, dtype=float):
    array = np.asarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class GripperGeometryEnvelope:
    bounds_world: np.ndarray
    geom_ids: tuple[int, ...]
    model_signature: str
    observation_id: str
    received_monotonic: float
    world_from_ee: np.ndarray
    motor_positions: np.ndarray
    joint_intervals: tuple[tuple[str, float, float], ...]
    assumptions: tuple[str, ...]
    execution_qualified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "bounds_world", _frozen(self.bounds_world))
        object.__setattr__(self, "world_from_ee", _frozen(self.world_from_ee))
        object.__setattr__(self, "motor_positions", _frozen(self.motor_positions))

    def robot_brackets(self, estimate, model, *, now):
        """Consume the bound in existing visual-effect frames using ONLY motors.

        This does not supply object pose/error or clock/alignment qualification.
        The enclosing source must qualify those and the declared model assumptions.
        """
        capture = estimate.motor_capture.capture
        capture.require(now=now)
        if (
            capture.observation_id != self.observation_id
            or capture.received_monotonic != self.received_monotonic
            or self.model_signature != _geometry_signature(model)
            or not np.array_equal(estimate.world_from_ee, self.world_from_ee)
            or not np.array_equal(estimate.motor_capture.positions, self.motor_positions)
        ):
            raise ObservationUnavailable("gripper_geometry_changed")
        return tuple(
            RobotPoseSample(estimate.world_from_ee, estimate.motor_capture.positions, t, self.bounds_world)
            for t in (capture.earliest, capture.latest)
        )

    def as_dict(self):
        return {
            "construction": "encoder_conditioned_model_joint_envelope",
            "observation_id": self.observation_id,
            "bounds_count": len(self.bounds_world),
            "joint_intervals": [list(item) for item in self.joint_intervals],
            "assumptions": list(self.assumptions),
            "execution_qualified": False,
        }


def bound_stretch_gripper(
    model,
    estimate,
    *,
    root_body="link_SG3_gripper_body",
    joint_intervals=None,
    calibration_id=None,
    position_error_m=0.0,
    rotation_error_rad=0.0,
):
    """Analytic enclosure of housing, cameras, fingers and rubber collision geoms.

    Default unknown hinges use their complete MODEL limits (a full turn for
    unlimited hinges). This is conditional on the model's mechanical validity;
    MuJoCo soft constraints may overrun limits. Overrides/overtravel need an
    explicit declared calibration ID, not a bound inferred from two controls.
    Motor aperture is measured. Other passive qpos values are never read from a
    robot/debug packet. A calibrated pose-error margin can enlarge the envelope.
    """
    joint_intervals = {} if joint_intervals is None else dict(joint_intervals)
    if (
        not np.isfinite([position_error_m, rotation_error_rad]).all()
        or position_error_m < 0
        or not 0 <= rotation_error_rad <= np.pi
        or (joint_intervals and not calibration_id)
    ):
        raise ValueError("gripper_geometry_calibration_required")
    if estimate.motor_capture.kinematic_signature != robot_kinematic_signature(model):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, STRETCH_ARM_CHAIN.ee_body)
    if root < 0 or ee < 0:
        raise ObservationUnavailable("gripper_geometry_unknown")
    bid = ee
    while bid != root:
        if bid <= 0 or model.body_jntnum[bid]:
            raise ObservationUnavailable("gripper_anchor_not_rigid")
        bid = int(model.body_parentid[bid])
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    nominal_root, nominal_ee = np.eye(4), np.eye(4)
    nominal_root[:3, :3], nominal_root[:3, 3] = data.xmat[root].reshape(3, 3), data.xpos[root]
    nominal_ee[:3, :3], nominal_ee[:3, 3] = data.xmat[ee].reshape(3, 3), data.xpos[ee]
    # This rigid ratio is independent of arm split and all passive finger state.
    root_pose = estimate.world_from_ee @ np.linalg.inv(nominal_ee) @ nominal_root
    ids = body_subtree(model, root_body)
    ranges, reported = {}, []
    known_aperture = float(estimate.motor_capture.as_dict()["gripper"])
    names = set()
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) not in ids or int(model.jnt_bodyid[jid]) == root:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            raise ObservationUnavailable("robot_joint_interval_identity_unavailable")
        names.add(name)
        if name == "joint_gripper_slide":
            if name in joint_intervals:
                raise ValueError("measured_motor_interval_override")
            limits = known_aperture, known_aperture
        elif name in joint_intervals:
            limits = tuple(joint_intervals[name])
        elif model.jnt_limited[jid]:
            limits = tuple(model.jnt_range[jid])
        elif model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            reference = model.qpos0[model.jnt_qposadr[jid]]
            limits = reference - np.pi, reference + np.pi
        else:
            raise ObservationUnavailable("robot_joint_interval_unavailable")
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] > limits[1]:
            raise ValueError("invalid_joint_interval")
        ranges[jid] = limits
        reported.append((name, float(limits[0]), float(limits[1])))
    if set(joint_intervals) - names:
        raise ValueError("unknown_gripper_joint_interval")
    geom_ids, boxes = subtree_geometry_intervals(model, root_id=root, world_from_root=root_pose, joint_intervals=ranges)
    # Angular error is around the measured EE origin, including housing offset.
    farthest = np.max(np.abs(boxes - estimate.world_from_ee[:3, 3]), axis=1)
    radius = np.linalg.norm(farthest, axis=-1).max()
    margin = position_error_m + 2 * radius * np.sin(rotation_error_rad / 2)
    boxes[:, 0] -= margin
    boxes[:, 1] += margin
    return GripperGeometryEnvelope(
        boxes,
        geom_ids,
        _geometry_signature(model),
        estimate.motor_capture.capture.observation_id,
        estimate.motor_capture.capture.received_monotonic,
        estimate.world_from_ee,
        estimate.motor_capture.positions,
        tuple(reported),
        (
            "calibrated_robot_geometry",
            "model_joint_limits" if not joint_intervals else "declared_intervals:" + str(calibration_id),
        ),
    )
