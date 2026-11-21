# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Stretch EE FK from motor channels and localized camera pose, without passive truth."""

from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.encoder_capture import (
    STRETCH_CAPTURE_MOTORS,
    MotorCapture,
    read_local_sim_motor_capture,
)
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import rigid_copy
from emet.motion.mujoco_collision import body_subtree
from emet.robots.stretch import STRETCH_ARM_CHAIN
from emet.simulation.synchronized_capture import camera_opencv_pose, robot_kinematic_signature


@dataclass(frozen=True)
class EncoderKinematicEstimate:
    world_from_ee: np.ndarray
    motor_capture: object
    inferred_joint_names: tuple[str, ...]
    world_from_base: np.ndarray
    # Joint coupling is a kinematic estimate, never individual motor readouts.
    gripper_geometry_qualified: bool = False
    execution_qualified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "world_from_ee", rigid_copy(self.world_from_ee))
        object.__setattr__(self, "world_from_base", rigid_copy(self.world_from_base))


def _motor_frame_kinematics(motors: MotorCapture, model, *, include_ee: bool):
    if motors.names != STRETCH_CAPTURE_MOTORS or motors.units != "actuator_length":
        raise ObservationUnavailable("invalid_joint_feedback")
    if motors.kinematic_signature != robot_kinematic_signature(model):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    ids = body_subtree(model, "base_link")
    if any(int(model.geom_bodyid[g]) not in ids for g in range(model.ngeom)) or any(
        int(model.jnt_bodyid[j]) not in ids for j in range(model.njnt)
    ):
        raise ObservationUnavailable("observed_robot_only_model_required")
    data = mujoco.MjData(model)
    values = motors.as_dict()
    assigned = set()
    for motor in ("lift", "wrist_yaw", "wrist_pitch", "wrist_roll", "head_pan", "head_tilt", "gripper"):
        joint = "joint_gripper_slide" if motor == "gripper" else "joint_" + motor
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor)
        if (
            jid < 0
            or aid < 0
            or model.actuator_gear[aid, 0] != 1.0
            or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[aid, 0] != jid
        ):
            raise ObservationUnavailable("robot_capture_model_mismatch")
        data.qpos[model.jnt_qposadr[jid]] = values[motor]
        assigned.add(joint)
    for i in range(4):
        name = f"joint_arm_l{i}"
        data.joint(name).qpos[0] = values["arm"] / 4
    # The runtime model must retain the calibrated collinear telescope chain.
    # Its kinematic signature is tied to the motor producer, not scene objects.
    root = model.body("base_link")
    root_joint = int(root.jntadr[0])
    if int(root.jntnum[0]) != 1 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    base_addr = int(model.jnt_qposadr[root_joint])
    data.qpos[base_addr : base_addr + 7] = [0, 0, 0, 1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    arm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_arm_l{i}") for i in range(4)]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_SLIDE for j in arm_ids) or not np.allclose(
        data.xaxis[arm_ids], data.xaxis[arm_ids[0]], atol=1e-9, rtol=0
    ):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "d435i_camera_rgb")
    if cid < 0:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    permitted = assigned | {f"joint_arm_l{i}" for i in range(4)}
    frames = [int(model.cam_bodyid[cid])]
    if include_ee:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, STRETCH_ARM_CHAIN.ee_body)
        if ee_id < 0:
            raise ObservationUnavailable("robot_capture_model_mismatch")
        frames.append(ee_id)
    for bid in frames:
        while bid > 0:
            for jid in range(int(model.body_jntadr[bid]), int(model.body_jntadr[bid] + model.body_jntnum[bid])):
                if jid != root_joint and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) not in permitted:
                    raise ObservationUnavailable("robot_frame_has_unmeasured_joints")
            bid = int(model.body_parentid[bid])
    return data, base_addr, cid, assigned


def stretch_base_from_head_camera(motors: MotorCapture, model) -> np.ndarray:
    """Return the upright head camera pose from calibrated, qualified motor readouts."""
    data, _, camera, _ = _motor_frame_kinematics(motors, model, include_ee=False)
    return rigid_copy(camera_opencv_pose(model, data, camera, upright=True))


def estimate_stretch_ee_from_encoders(observation, model, *, now=None):
    """Estimate the rigid grasp frame from motors and a localized camera pose.

    Dividing total extension across the collinear telescope preserves rigid FK.
    Individual-link collision geometry and localization errors need their own bounds.
    """
    motors = read_local_sim_motor_capture(observation, now=now)
    data, base_addr, cid, assigned = _motor_frame_kinematics(motors, model, include_ee=True)
    root_from_camera = camera_opencv_pose(model, data, cid, upright=True)
    world_from_root = rigid_copy(observation.camera_pose) @ np.linalg.inv(root_from_camera)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, world_from_root[:3, :3].ravel())
    data.qpos[base_addr : base_addr + 3], data.qpos[base_addr + 3 : base_addr + 7] = world_from_root[:3, 3], quat
    mujoco.mj_kinematics(model, data)
    ee = data.body(STRETCH_ARM_CHAIN.ee_body)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = ee.xmat.reshape(3, 3), ee.xpos
    inferred = tuple(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "@base_link:0")
        for j in range(model.njnt)
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) not in assigned
    )
    return EncoderKinematicEstimate(pose, motors, inferred, world_from_root)
