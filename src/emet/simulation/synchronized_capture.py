# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""One simulator sample for images and robot-only kinematic debug feedback.

Only the renderer sees the private scene snapshot. Published metadata contains
robot configuration and robot-mounted camera FK, never environment joint state.
Full configuration includes unmeasured passive joints. It is debug truth, not a
declaration that hardware has encoders for those joints. This ideal simulator
sampling contract does not calibrate hardware clocks/noise.
"""

import hashlib
import json
import time

import mujoco
import numpy as np

from emet.controller.manipulation.capture_clock import local_monotonic_clock_id
from emet.controller.manipulation.encoder_capture import STRETCH_CAPTURE_MOTORS
from emet.motion.mujoco_collision import body_subtree

JOINT_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}


def capture_stretch_motors(model, probe, sample):
    """The same eight motor-position channels used by Stretch's normal driver.

    These lengths are encoder-equivalent measurements in the simulator's motor
    calibration. They do not measure passive fingers or telescope split lengths.
    """
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in STRETCH_CAPTURE_MOTORS]
    if any(aid < 0 for aid in ids):
        return None  # Explicitly unsupported motor profile, never full-qpos fallback.
    robot_ids = body_subtree(model, "base_link")
    for name, aid in zip(STRETCH_CAPTURE_MOTORS, ids, strict=True):
        if model.actuator_gear[aid, 0] != 1.0:
            raise ValueError("unsupported_stretch_encoder_calibration")
        if name == "arm":
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "extend")
            if (
                tid < 0
                or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON
                or model.actuator_trnid[aid, 0] != tid
            ):
                raise ValueError("unsupported_stretch_encoder_calibration")
            adr, count = int(model.tendon_adr[tid]), int(model.tendon_num[tid])
            expected = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_arm_l{i}") for i in range(4)}
            if (
                count != 4
                or -1 in expected
                or set(model.wrap_objid[adr : adr + count]) != expected
                or not np.all(model.wrap_type[adr : adr + count] == mujoco.mjtWrap.mjWRAP_JOINT)
                or not np.all(model.wrap_prm[adr : adr + count] == 1.0)
                or any(model.qpos0[model.jnt_qposadr[j]] != 0.0 for j in expected)
                or any(int(model.jnt_bodyid[j]) not in robot_ids for j in expected)
            ):
                raise ValueError("unsupported_stretch_encoder_calibration")
        else:
            joint_name = "joint_gripper_slide" if name == "gripper" else "joint_" + name
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if (
                jid < 0
                or int(model.jnt_bodyid[jid]) not in robot_ids
                or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[aid, 0] != jid
                or model.qpos0[model.jnt_qposadr[jid]] != 0.0
            ):
                raise ValueError("unsupported_stretch_encoder_calibration")
    mujoco.mj_tendon(model, probe)
    mujoco.mj_transmission(model, probe)
    return {
        "schema_version": 1,
        "source": "simulator_stretch_motor_encoders",
        "clock_id": sample["clock_id"],
        "interval_ns": list(sample["interval_ns"]),
        "units": "actuator_length",
        "kinematic_signature": sample["kinematic_signature"],
        "positions": {
            name: float(probe.actuator_length[aid]) for name, aid in zip(STRETCH_CAPTURE_MOTORS, ids, strict=True)
        },
    }


def robot_joint_layout(model, root_body="base_link"):
    ids = body_subtree(model, root_body)
    if not ids:
        raise ValueError("robot_capture_model_unavailable")
    layout = {}
    for bid in sorted(ids):
        for local in range(int(model.body_jntnum[bid])):
            jid = int(model.body_jntadr[bid]) + local
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            key = name or f"@{body}:{local}"
            if key in layout or body is None:
                raise ValueError("robot_capture_joint_identity_ambiguous")
            layout[key] = (int(model.jnt_qposadr[jid]), JOINT_WIDTH[int(model.jnt_type[jid])], jid)
    if not layout:
        raise ValueError("robot_capture_model_unavailable")
    return layout


def robot_kinematic_signature(model, root_body="base_link"):
    ids = body_subtree(model, root_body)
    indices = {bid: index for index, bid in enumerate(sorted(ids))}
    body_rows = []
    for bid in sorted(ids):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        parent = int(model.body_parentid[bid])
        body_rows.append((name, indices.get(parent), model.body_pos[bid].tolist(), model.body_quat[bid].tolist()))
    joints = []
    for name, (_, width, jid) in sorted(robot_joint_layout(model, root_body).items()):
        joints.append(
            (
                name,
                int(model.jnt_type[jid]),
                indices[int(model.jnt_bodyid[jid])],
                width,
                model.jnt_pos[jid].tolist(),
                model.jnt_axis[jid].tolist(),
                model.qpos0[model.jnt_qposadr[jid] : model.jnt_qposadr[jid] + width].tolist() if width != 7 else None,
            )
        )
    cameras = [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, c),
            indices[int(model.cam_bodyid[c])],
            model.cam_pos[c].tolist(),
            model.cam_quat[c].tolist(),
        )
        for c in range(model.ncam)
        if int(model.cam_bodyid[c]) in ids
    ]
    motors = []
    for name in STRETCH_CAPTURE_MOTORS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            continue
        kind, target = int(model.actuator_trntype[aid]), int(model.actuator_trnid[aid, 0])
        obj = mujoco.mjtObj.mjOBJ_JOINT if kind == mujoco.mjtTrn.mjTRN_JOINT else mujoco.mjtObj.mjOBJ_TENDON
        target_name = mujoco.mj_id2name(model, obj, target) if target >= 0 else None
        motors.append((name, kind, target_name, model.actuator_gear[aid].tolist()))
    return hashlib.sha256(json.dumps((body_rows, joints, cameras, motors), sort_keys=True).encode()).hexdigest()


def camera_opencv_pose(model, data, camera_id, *, upright=False):
    pose = np.eye(4)
    pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    if upright:
        pose[:3, :3] = pose[:3, :3] @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, 3] = data.cam_xpos[camera_id]
    return pose


def capture_scene_for_sensors(model, data, physics_lock, *, cameras, root_body="base_link"):
    """Copy under the physics lock, then render/FK that private immutable sample.

    Capture time bounds the copy, not subsequent rendering or IPC publication.
    Physics can advance while the sampled pixels are rendered. All cameras and
    robot feedback still describe the same sample, including during motion.
    """
    probe = mujoco.MjData(model)
    with physics_lock:
        start = time.monotonic_ns()
        mujoco.mj_copyData(probe, model, data)
        end = time.monotonic_ns()
    mujoco.mj_kinematics(model, probe)
    mujoco.mj_camlight(model, probe)
    layout = robot_joint_layout(model, root_body)
    joints = {key: probe.qpos[addr : addr + width].tolist() for key, (addr, width, _) in layout.items()}
    robot_ids = body_subtree(model, root_body)
    poses = {}
    for key, (camera_name, upright) in cameras.items():
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cid < 0 or int(model.cam_bodyid[cid]) not in robot_ids:
            raise ValueError("robot_capture_camera_unavailable")
        poses[key] = camera_opencv_pose(model, probe, cid, upright=upright).tolist()
    packet = {
        "schema_version": 1,
        "source": "simulator_robot_kinematics_debug",
        "clock_id": local_monotonic_clock_id(),
        "interval_ns": [start, end],
        "root_body": root_body,
        "kinematic_signature": robot_kinematic_signature(model, root_body),
        "joint_positions": joints,
        "camera_poses": poses,
    }
    return probe, packet
