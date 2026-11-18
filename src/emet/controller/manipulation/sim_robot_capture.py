# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Robot-only synchronized debug truth; never qualify observed robot feedback."""

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from emet.controller.manipulation.capture_clock import local_monotonic_clock_id, local_sim_capture_interval
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval, RobotPoseSample
from emet.motion.mujoco_collision import body_subtree
from emet.simulation.synchronized_capture import camera_opencv_pose, robot_joint_layout, robot_kinematic_signature


@dataclass(frozen=True)
class SynchronizedRobotCapture:
    capture: CaptureInterval
    robot_before: RobotPoseSample
    robot_after: RobotPoseSample
    camera_pose: np.ndarray
    # Exact co-sampling is a simulator property, not calibrated hardware noise.
    alignment_error_m: float = 0.0
    alignment_error_rad: float = 0.0
    source: str = "simulator_robot_kinematics_debug"
    execution_qualified: bool = False


def read_synchronized_robot_capture(
    observation, model, *, ee_body, gripper_bodies, root_body="base_link", camera_name="d435i_camera_rgb", now=None
):
    """Verify a local shared-state sample against a ROBOT-ONLY calibrated model.

    No scene snapshots, object bodies or contact metadata are read. A packet from
    a remote clock, ordinary asynchronous image or hardware client is rejected.
    Published camera pose must agree with the sampled robot FK, not later state.
    """
    now = time.monotonic() if now is None else now
    if observation.is_simulation is not True:
        raise ObservationUnavailable("simulation_capture_required")
    timing = (observation.image_timing or {}).get("head_cam_left")
    if not isinstance(timing, dict) or timing.get("source") != "renderer_synchronized_snapshot":
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    stamp = timing.get("timestamp_ns")
    if type(stamp) is not int or stamp <= 0:
        raise ObservationUnavailable("capture_interval_unavailable")
    capture = local_sim_capture_interval(
        timing, received_monotonic=observation.received_monotonic, observation_id=f"capture:{stamp}"
    )
    capture.require(now=now)
    packet = timing.get("robot_sample")
    expected = {
        "schema_version",
        "source",
        "clock_id",
        "interval_ns",
        "root_body",
        "kinematic_signature",
        "joint_positions",
        "camera_poses",
    }
    if (
        not isinstance(packet, dict)
        or set(packet) != expected
        or packet["schema_version"] != 1
        or packet["source"] != "simulator_robot_kinematics_debug"
        or packet["clock_id"] != local_monotonic_clock_id()
        or packet["root_body"] != root_body
        or packet["interval_ns"] != timing.get("capture_interval_ns")
        or packet["interval_ns"] != timing.get("depth_capture_interval_ns")
        or stamp != timing["capture_interval_ns"][0]
        or not isinstance(packet["camera_poses"], dict)
    ):
        raise ObservationUnavailable("capture_motion_alignment_unknown")
    robot_ids = body_subtree(model, root_body)
    if any(int(model.geom_bodyid[g]) not in robot_ids for g in range(model.ngeom)) or any(
        int(model.jnt_bodyid[j]) not in robot_ids for j in range(model.njnt)
    ):
        raise ObservationUnavailable("observed_robot_only_model_required")
    if packet["kinematic_signature"] != robot_kinematic_signature(model, root_body):
        raise ObservationUnavailable("robot_capture_model_mismatch")
    layout = robot_joint_layout(model, root_body)
    positions = packet["joint_positions"]
    if not isinstance(positions, dict) or set(positions) != set(layout):
        raise ObservationUnavailable("invalid_joint_feedback")
    data = mujoco.MjData(model)
    for name, (addr, width, _jid) in layout.items():
        try:
            q = np.asarray(positions[name], dtype=float)
        except (TypeError, ValueError) as error:
            raise ObservationUnavailable("invalid_joint_feedback") from error
        if q.shape != (width,) or not np.isfinite(q).all():
            raise ObservationUnavailable("invalid_joint_feedback")
        if width in (4, 7) and not np.isclose(np.linalg.norm(q[-4:]), 1.0, atol=1e-6):
            raise ObservationUnavailable("invalid_joint_feedback")
        data.qpos[addr : addr + width] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cid < 0 or int(model.cam_bodyid[cid]) not in robot_ids:
        raise ObservationUnavailable("robot_capture_model_mismatch")
    camera_pose = camera_opencv_pose(model, data, cid, upright=True)
    published = np.asarray(observation.camera_pose, float)
    sampled = np.asarray(packet["camera_poses"].get("cam_d435i_rgb"), float)
    if (
        published.shape != (4, 4)
        or sampled.shape != (4, 4)
        or not np.allclose(camera_pose, sampled, atol=1e-6, rtol=0)
        or not np.allclose(camera_pose, published, atol=1e-6, rtol=0)
    ):
        raise ObservationUnavailable("capture_camera_pose_mismatch")
    before = RobotPoseSample.from_model(
        model,
        data,
        ee_body=ee_body,
        gripper_bodies=gripper_bodies,
        received_monotonic=capture.earliest,
        source="simulator_robot_kinematics_debug",
    )
    after = RobotPoseSample.from_model(
        model,
        data,
        ee_body=ee_body,
        gripper_bodies=gripper_bodies,
        received_monotonic=capture.latest,
        source="simulator_robot_kinematics_debug",
    )
    camera_pose = np.frombuffer(camera_pose.tobytes(), dtype=float).reshape(4, 4)
    return SynchronizedRobotCapture(capture, before, after, camera_pose)
