# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Whole-robot outer geometry from Stretch motors and declared mechanical limits.

These are possible occupied volumes, never captured inner material or free-space
evidence. Passive finger/rubber/wheel state and individual telescope lengths are
not sensors. Mechanical limits remain model assumptions, including soft limits.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.gripper_geometry import _geometry_signature
from emet.controller.manipulation.interval_geometry import subtree_geometry_intervals
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import pose_uncertainty_margin, rigid_copy
from emet.motion.mujoco_collision import body_subtree
from emet.simulation.synchronized_capture import robot_kinematic_signature


@dataclass(frozen=True)
class StretchGeometryEnvelope:
    geom_ids: tuple[int, ...]
    bounds_world: np.ndarray
    joint_intervals: tuple[tuple[str, float, float], ...]
    observation_id: str
    model_signature: str
    assumptions: tuple[str, ...]
    execution_qualified: bool = False

    def __post_init__(self):
        boxes = np.asarray(self.bounds_world, float)
        object.__setattr__(self, "bounds_world", np.frombuffer(boxes.tobytes(), dtype=float).reshape(boxes.shape))


def bound_stretch_robot(model, estimate, *, now, position_error_m, rotation_error_rad):
    """Enclose every collidable geom over all compatible passive configurations.

    Each telescope interval is intersected with the measured total extension
    constraint. The distal arm body's rigid pose uses that total exactly, after
    checking the calibrated collinear chain. Intermediate links retain interval
    uncertainty. No body pose is read from simulator/debug scene feedback.
    Explicit localization/calibration margins dilate every bound about the base.
    """
    motors = estimate.motor_capture
    motors.capture.require(now=now)
    if motors.kinematic_signature != robot_kinematic_signature(model):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    if (
        not np.isfinite([position_error_m, rotation_error_rad]).all()
        or position_error_m < 0
        or not 0 <= rotation_error_rad <= np.pi
    ):
        raise ValueError("invalid_robot_geometry_error")
    ids = body_subtree(model, "base_link")
    if not ids or any(int(model.geom_bodyid[g]) not in ids for g in range(model.ngeom)):
        raise ObservationUnavailable("observed_robot_only_model_required")
    data = mujoco.MjData(model)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rj = int(model.body_jntadr[root])
    if model.body_jntnum[root] != 1 or model.jnt_type[rj] != mujoco.mjtJoint.mjJNT_FREE:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    pose = rigid_copy(estimate.world_from_base)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, pose[:3, :3].ravel())
    address = int(model.jnt_qposadr[rj])
    data.qpos[address : address + 7] = np.r_[pose[:3, 3], quat]
    values, intervals = motors.as_dict(), {}
    for motor in ("lift", "wrist_yaw", "wrist_pitch", "wrist_roll", "head_pan", "head_tilt", "gripper"):
        name = "joint_gripper_slide" if motor == "gripper" else "joint_" + motor
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)
        if (
            j < 0
            or a < 0
            or model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[a, 0] != j
            or model.actuator_gear[a, 0] != 1
        ):
            raise ObservationUnavailable("robot_capture_model_mismatch")
        data.qpos[model.jnt_qposadr[j]] = values[motor]
        intervals[j] = (values[motor], values[motor])
    measured = set(intervals)
    arm = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_arm_l{i}") for i in range(4)]
    if any(j < 0 or not model.jnt_limited[j] or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE for j in arm):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    total = values["arm"]
    limits = model.jnt_range[arm]
    if not limits[:, 0].sum() <= total <= limits[:, 1].sum():
        raise ObservationUnavailable("robot_joint_interval_unavailable")
    # A feasible representative is only a kinematic construction. It is not a
    # measured telescope split and is never published as sensor feedback.
    remaining = total - limits[:, 0].sum()
    for i, j in enumerate(arm):
        low = max(limits[i, 0], total - np.delete(limits[:, 1], i).sum())
        high = min(limits[i, 1], total - np.delete(limits[:, 0], i).sum())
        intervals[j] = (low, high)
        delta = min(remaining, limits[i, 1] - limits[i, 0])
        data.qpos[model.jnt_qposadr[j]] = limits[i, 0] + delta
        remaining -= delta
    for j in range(model.njnt):
        if j == rj or j in intervals:
            continue
        if int(model.jnt_bodyid[j]) not in ids:
            raise ObservationUnavailable("observed_robot_only_model_required")
        if model.jnt_limited[j]:
            intervals[j] = tuple(model.jnt_range[j])
        elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            ref = model.qpos0[model.jnt_qposadr[j]]
            intervals[j] = (ref - np.pi, ref + np.pi)
        else:
            raise ObservationUnavailable("robot_joint_interval_unavailable")
    mujoco.mj_kinematics(model, data)
    if not np.allclose(data.xaxis[arm], data.xaxis[arm[0]], atol=1e-9, rtol=0):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    # The order/ancestry contract matters: axis alignment alone cannot prove
    # that a measured sum controls one serial distal frame.
    distal = int(model.jnt_bodyid[arm[0]])
    found = []
    bid = distal
    while bid > 0 and bid != root:
        if any(
            j not in measured and j not in arm
            for j in range(int(model.body_jntadr[bid]), int(model.body_jntadr[bid] + model.body_jntnum[bid]))
        ):
            raise ObservationUnavailable("robot_frame_has_unmeasured_joints")
        found.extend(
            j
            for j in range(int(model.body_jntadr[bid]), int(model.body_jntadr[bid] + model.body_jntnum[bid]))
            if j in arm
        )
        bid = int(model.body_parentid[bid])
    if bid != root or found != arm:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    rotation, translation = data.xmat[distal].reshape(3, 3), data.xpos[distal]
    overrides = {distal: ((rotation, rotation), (translation, translation))}
    geom_ids, boxes = subtree_geometry_intervals(
        model, root_id=root, world_from_root=pose, joint_intervals=intervals, body_overrides=overrides
    )
    radius = np.linalg.norm(np.max(np.abs(boxes - pose[:3, 3]), axis=1), axis=-1).max()
    margin = pose_uncertainty_margin(radius, position_error_m, rotation_error_rad)
    boxes[:, 0] -= margin
    boxes[:, 1] += margin
    reported = tuple(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), float(a), float(b))
        for j, (a, b) in sorted(intervals.items())
    )
    return StretchGeometryEnvelope(
        geom_ids,
        boxes,
        reported,
        motors.capture.observation_id,
        _geometry_signature(model),
        ("calibrated_robot_geometry", "model_joint_limits", "measured_total_extension", "bounded_localization"),
    )
