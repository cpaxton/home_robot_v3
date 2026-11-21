# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).
"""Stretch navigation trial with registered depth and capture-aligned encoders."""

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from emet.controller.manipulation.encoder_capture import read_local_sim_motor_capture, read_local_sim_wheel_capture
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.stretch_encoder_kinematics import stretch_base_from_head_camera
from emet.core.navigation_result import NavigationPoseUncertainty
from emet.core.zmq_protocol import (
    CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
    EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY,
    EMET_ZMQ_SESSION_KEY,
)
from emet.mapping.wheel_motion_guard import WheelMotionGuard
from emet.mapping.wheel_odometry import DifferentialDriveOdometry
from emet.robots.head_capability import STRETCH_LEGACY_HEAD, HeadCapability, mujoco_head_capability
from emet.simulation.env_flags import env_sim_nav_teleport
from emet.simulation.mujoco_server_stretch import MujocoZmqServer
from emet.simulation.sim_manipulation import set_free_body_pose
from emet.simulation.stretch_mujoco.config import robot_settings
from emet.simulation.stretch_mujoco.enums.stretch_cameras import StretchCameras
from emet.simulation.stretch_mujoco.mujoco_server import MujocoServer
from emet.utils.geometry import xyt2sophus


@dataclass(frozen=True)
class EncoderNavigationFrame:
    timestamp_ns: int
    base_xyt: np.ndarray
    camera_pose: np.ndarray


class StretchChangedObjectPhysics(MujocoServer):
    """Physics-process fixture intervention with no robot-client command."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervention_dir = Path(os.environ["EMET_EVAL_CHANGED_OBJECT_DIR"])
        self.intervened = False

    def _ctrl_callback(self, model, data):
        super()._ctrl_callback(model, data)
        if self.intervened or not (self.intervention_dir / "scoring-intervention.request").is_file():
            return
        self.intervened = True
        ok = set_free_body_pose(model, data, "object1", [-.13, -.75, .55])
        pending = self.intervention_dir / "scoring-intervention.tmp"
        pending.write_text(json.dumps({"ok": ok, "monotonic": time.monotonic()}))
        pending.replace(self.intervention_dir / "scoring-intervention.json")


class StretchEncoderNavigationServer(MujocoZmqServer):
    """Use one local encoder frame for RGB-D, waypoint control, and arrival."""

    def __init__(
        self, *args, encoder_model_path, arrival_pose_uncertainty: NavigationPoseUncertainty | None = None,
        motion_reference: Callable[[dict], np.ndarray | None] | None = None, **kwargs,
    ):
        if env_sim_nav_teleport():
            raise ValueError("encoder navigation requires physical wheel drive")
        self._encoder_model = mujoco.MjModel.from_xml_path(str(encoder_model_path))
        self._arrival_pose_uncertainty = arrival_pose_uncertainty
        if motion_reference is not None and arrival_pose_uncertainty is None:
            raise ValueError("independent motion comparison requires calibrated tolerances")
        self._motion_reference = motion_reference
        self._motion_guard = (
            WheelMotionGuard(arrival_pose_uncertainty, max_age_s=.75) if motion_reference is not None else None
        )
        self._motion_check = None
        self._motion_stamp = None
        self._wheel_odometry = DifferentialDriveOdometry(
            wheel_radius_m=robot_settings["wheel_diameter"] / 2,
            track_width_m=robot_settings["wheel_separation"],
        )
        self._encoder_frame = None
        self._encoder_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def _navigation_frame_origin(self):
        return np.zeros(3)

    def _build_emet_session_stretch(self, *, robocasa):
        head = mujoco_head_capability(self._encoder_model, ("joint_head_pan", "joint_head_tilt"))
        if head is None:
            raise ValueError("encoder navigation requires a calibrated head")
        head = HeadCapability(
            (max(head.pan[0], STRETCH_LEGACY_HEAD.pan[0]), min(head.pan[1], STRETCH_LEGACY_HEAD.pan[1])),
            head.tilt,
            head.source,
        )
        return {
            "schema_version": CURRENT_EMET_ZMQ_SESSION_SCHEMA_VERSION,
            "runtime_kind": "stretch_mujoco_sim",
            "is_simulation": True,
            "emet_robot_id": self.get_robot_spec().name,
            "capabilities": {
                "depth": True, "num_cameras": 1, "dof": int(self.get_robot_spec().dof),
                "head_motion": head.as_dict(), "teleport_base": False,
            },
            "pose_source": "stretch_wheel_and_head_encoders",
        }

    def _capture_encoder_frame(self, imagery):
        if imagery is None:
            return None
        timing = imagery.image_timing_for(StretchCameras.cam_d435i_rgb)
        if timing is None:
            return None
        timing.pop("robot_sample", None)
        with self._encoder_lock:
            now = time.monotonic()
            stamp = timing["timestamp_ns"]
            if not 0 <= now - stamp / 1e9 <= 1.0:
                return None
            previous = self._encoder_frame
            if previous is not None and stamp <= previous.timestamp_ns:
                return previous if stamp == previous.timestamp_ns else None
            obs = SimpleNamespace(
                is_simulation=True, received_monotonic=now, image_timing={"head_cam_left": timing}
            )
            try:
                wheels = read_local_sim_wheel_capture(obs, now=now)
                motors = read_local_sim_motor_capture(obs, now=now)
                base_from_camera = stretch_base_from_head_camera(motors, self._encoder_model)
                xyt = self._wheel_odometry.update(wheels)
            except ObservationUnavailable:
                return None
            camera_pose = xyt2sophus(xyt).matrix() @ base_from_camera
            self._encoder_frame = EncoderNavigationFrame(stamp, xyt, camera_pose)
            return self._encoder_frame

    def get_base_pose(self):
        frame = self._capture_encoder_frame(self._camera_data)
        return frame.base_xyt if frame is not None else None

    def _state_poses(self):
        # Keep command receipts flowing when the pose source stops publishing.
        return self.get_base_pose(), None

    def _full_observation_poses(self, cam_data):
        frame = self._capture_encoder_frame(cam_data)
        if frame is None:
            return None
        return frame.base_xyt, frame.camera_pose, None

    def get_servo_message(self):
        return self.get_full_observation_message()

    def _check_independent_motion(self, imagery, frame):
        """Check each fixed-camera packet once against the same snapshot's wheel pose."""
        now = time.monotonic_ns()
        if self._motion_stamp == frame.timestamp_ns and now - frame.timestamp_ns <= 750_000_000:
            return self._motion_check
        reference = imagery.fixed_rgbd_sample
        wheel_packet = imagery.synchronized_wheel_sample
        if (
            reference is not None and wheel_packet is not None
            and reference.get("clock_id") == wheel_packet.get("clock_id")
            and reference.get("interval_ns") == wheel_packet.get("interval_ns")
        ):
            reference_xyt = self._motion_reference(reference)
            reference_stamp = reference["interval_ns"][0]
        else:
            reference_xyt, reference_stamp = None, None
        self._motion_check = self._motion_guard.update(
            frame.base_xyt, reference_xyt, wheel_timestamp_ns=frame.timestamp_ns,
            reference_timestamp_ns=reference_stamp, now_ns=now,
        )
        self._motion_stamp = frame.timestamp_ns
        return self._motion_check

    def navigation_policy_measurement(self):
        imagery = self._camera_data
        frame = self._capture_encoder_frame(imagery)
        if frame is None:
            return {"pose": None, "timestamp": None, "stopped": False}
        if self._motion_guard is not None:
            check = self._check_independent_motion(imagery, frame)
            if not check.ok:
                return {"failure": check.status, "failure_details": {"motion_check": asdict(check)}}
        return {
            "pose": frame.base_xyt,
            "timestamp": frame.timestamp_ns / 1e9,
            "stopped": self.base_controller_at_goal() and not self.active,
        }

    def _use_molmospaces_nav_teleport(self):
        return False

    def start_navigation_command(self, action):
        imagery = self._camera_data
        frame = self._capture_encoder_frame(imagery)
        if frame is None or action.get("nav_teleport") or env_sim_nav_teleport():
            raise RuntimeError("encoder navigation requires fresh wheel feedback and physical wheel drive")
        uncertainty = self._arrival_pose_uncertainty
        if uncertainty is not None and not action.get("nav_policy"):
            raise ValueError("bounded encoder navigation requires a navigation policy")
        if self._motion_guard is not None:
            check = self._check_independent_motion(imagery, frame)
            if not check.ok:
                raise ValueError(check.status)
        context = super().start_navigation_command(action)
        if uncertainty is not None:
            context["arrival_pose_uncertainty"] = asdict(uncertainty)
        if self._motion_guard is not None:
            context["independent_motion_check"] = "fixed_rgbd"
        return context

    def _navigation_control_policy(self, name):
        policy = super()._navigation_control_policy(name)
        uncertainty = self._arrival_pose_uncertainty
        if uncertainty is None:
            return policy
        xy, yaw = uncertainty.usable_tolerances(policy.xy_tolerance, policy.yaw_tolerance)
        return replace(policy, xy_tolerance=xy, yaw_tolerance=yaw)

    def check_base_poses(self, poses):
        raise RuntimeError("encoder navigation uses observed clearance")

    def handle_action(self, action):
        if (
            EMET_ACTION_MUJOCO_GROUND_TRUTH_KEY in action
            or action.get("nav_teleport")
            or any(key.startswith("sim_") for key in action)
        ):
            raise RuntimeError("simulator state actions are unavailable in encoder navigation")
        return super().handle_action(action)

    def _attach_emet_session(self, message):
        session = self._emet_session
        if session is not None:
            message[EMET_ZMQ_SESSION_KEY] = {**session, "navigation_origin_xyt": [0., 0., 0.]}
        for key in ("ee_pose", "base_xyz", "base_up_dot_world_z", "actuator_targets", "actuator_names"):
            message.pop(key, None)
        timing = message.get("head_cam_left/image_timing")
        if timing is not None:
            timing.pop("robot_sample", None)
        return message
