# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""XLeRobot dual-arm mobile manipulator — https://github.com/Vector-Wangel/XLeRobot."""

from pathlib import Path

import numpy as np

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "robot" / "xlerobot"
_MJCF_PATH = str(_ASSETS_DIR / "xlerobot.xml")

XLEROBOT_JOINT_NAMES = [
    "slide_joint_x",
    "slide_joint_y",
    "hinge_joint_z",
    "ST3215_Servo_Motor-v1-2_Hub---Servo",
    "ST3215_Servo_Motor-v1-1_Hub-2---Servo",
    "ST3215_Servo_Motor-v1_Revolute-40",
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
    "Jaw_L",
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
    "Jaw_R",
    "head_pan_joint",
    "head_tilt_joint",
]

XLEROBOT_ACTUATOR_NAMES = [
    "slider_actuator_x",
    "slider_actuator_y",
    "hinge_actuator_z",
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
    "Jaw_R",
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
    "Jaw_L",
    "head_pan",
    "head_tilt",
    "wheel1",
    "wheel2",
    "wheel3",
]

# Jaw position actuators (``Jaw`` class ctrlrange ≈ -0.2 … 2.0 rad).
XLEROBOT_JAW_CLOSED = 0.0
XLEROBOT_JAW_OPEN = 1.6
XLEROBOT_GRIPPER_JOINTS = {"left": "Jaw_L", "right": "Jaw_R"}
XLEROBOT_GRIPPER_ACTUATORS = {"left": "Jaw_L", "right": "Jaw_R"}
XLEROBOT_HEAD_JOINTS = ("head_pan_joint", "head_tilt_joint")
XLEROBOT_HEAD_ACTUATORS = ("head_pan", "head_tilt")


def jaw_angle_from_normalized(value: float) -> float:
    """Map Stretch-style gripper scalar (~0 closed, ~1 open) to jaw radians."""
    t = float(np.clip(value, 0.0, 1.0))
    return XLEROBOT_JAW_CLOSED + t * (XLEROBOT_JAW_OPEN - XLEROBOT_JAW_CLOSED)


def jaw_normalized_from_angle(angle: float) -> float:
    span = XLEROBOT_JAW_OPEN - XLEROBOT_JAW_CLOSED
    if span <= 1e-9:
        return 0.0
    return float(np.clip((float(angle) - XLEROBOT_JAW_CLOSED) / span, 0.0, 1.0))


def parse_xlerobot_gripper_side(gripper_name: str) -> str:
    """Return ``left`` or ``right`` from a gripper name / alias."""
    g = gripper_name.lower().replace("-", "_")
    if "right" in g:
        return "right"
    return "left"


XLEROBOT_CAMERA_NAMES = ["head_camera_left", "head_camera_right"]

# Tucked arms + slight head down for navigation (avoids zero-pose arm penetration in merged scenes).
XLEROBOT_NAVIGATION_JOINT_QPOS: dict[str, float] = {
    "Rotation_L": 0.5,
    "Pitch_L": -0.8,
    "Elbow_L": 2.0,
    "Wrist_Pitch_L": -1.0,
    "Wrist_Roll_L": 0.0,
    "Jaw_L": 0.5,
    "Rotation_R": -0.5,
    "Pitch_R": -0.8,
    "Elbow_R": 2.0,
    "Wrist_Pitch_R": -1.0,
    "Wrist_Roll_R": 0.0,
    "Jaw_R": 0.5,
    "head_pan_joint": 0.0,
    "head_tilt_joint": -0.35,
}


def apply_xlerobot_navigation_joint_pose(model, data) -> None:
    """Set arm/head ``qpos`` to :data:`XLEROBOT_NAVIGATION_JOINT_QPOS` (planar base unchanged)."""
    import mujoco

    for jname, val in XLEROBOT_NAVIGATION_JOINT_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr] = float(val)
        vadr = int(model.jnt_dofadr[jid])
        if vadr >= 0:
            data.qvel[vadr] = 0.0
    mujoco.mj_forward(model, data)


class XLeRobotBackend(RobotBackend):
    """XLeRobot dual-arm mobile platform (planar slide+yaw base)."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="xlerobot",
            dof=len(XLEROBOT_JOINT_NAMES),
            joint_names=list(XLEROBOT_JOINT_NAMES),
            camera_names=list(XLEROBOT_CAMERA_NAMES),
            urdf_path=None,
            mjcf_path=_MJCF_PATH,
            actuator_names=list(XLEROBOT_ACTUATOR_NAMES),
            base_link_name="chassis",
            footprint=Footprint(width=0.45, length=0.45, width_offset=0.0, length_offset=0.0),
            planar_base_joint_names=("slide_joint_x", "slide_joint_y", "hinge_joint_z"),
            planar_spawn_xy_extra_margin_m=0.35,
            planar_spawn_clip_guard_body_names=("Jaw_L", "Jaw_R"),
            planar_spawn_clip_guard_pad_m=0.25,
            robosuite_rgb_depth_ops=("flipud",),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        raise NotImplementedError("XLeRobot kinematic model not yet implemented in emet.")


__all__ = [
    "XLEROBOT_ACTUATOR_NAMES",
    "XLEROBOT_CAMERA_NAMES",
    "XLEROBOT_GRIPPER_ACTUATORS",
    "XLEROBOT_GRIPPER_JOINTS",
    "XLEROBOT_HEAD_ACTUATORS",
    "XLEROBOT_HEAD_JOINTS",
    "XLEROBOT_JAW_CLOSED",
    "XLEROBOT_JAW_OPEN",
    "XLEROBOT_NAVIGATION_JOINT_QPOS",
    "XLeRobotBackend",
    "apply_xlerobot_navigation_joint_pose",
    "jaw_angle_from_normalized",
    "jaw_normalized_from_angle",
    "parse_xlerobot_gripper_side",
]
