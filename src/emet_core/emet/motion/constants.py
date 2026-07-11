# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import logging
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from emet.utils.config import get_full_config_path

logger = logging.getLogger(__name__)


def _ensure_stretch_urdf(urdf_path: str) -> str:
    """If stretch.urdf doesn't exist, generate it from the stretch_urdf package."""
    if os.path.isfile(urdf_path):
        return urdf_path

    try:
        import importlib.resources

        pkg_path = str(importlib.resources.files("stretch_urdf"))
    except (ImportError, ModuleNotFoundError):
        logger.warning(
            "stretch.urdf not found at %s and stretch_urdf package is not installed. "
            "IK will fail. Copy a calibrated URDF or install stretch_urdf.",
            urdf_path,
        )
        return urdf_path

    model_name = "SE3"
    tool_name = "eoa_wrist_dw3_tool_sg3"
    src_urdf = os.path.join(pkg_path, model_name, f"stretch_description_{model_name}_{tool_name}.urdf")
    mesh_dir = os.path.join(pkg_path, model_name, "meshes")

    if not os.path.isfile(src_urdf):
        logger.warning("Stock URDF not found at %s", src_urdf)
        return urdf_path

    logger.info("Generating default stretch.urdf from stretch_urdf package -> %s", urdf_path)

    with open(src_urdf, "r") as f:
        urdf_text = f.read()

    for match in re.finditer(r'filename="(.+?)"', urdf_text):
        orig = match.group(1)
        fn = orig.split("/")[-1]
        urdf_text = urdf_text.replace(orig, os.path.join(mesh_dir, fn))

    os.makedirs(os.path.dirname(urdf_path), exist_ok=True)

    tree = ET.ElementTree(ET.fromstring(urdf_text))
    root = tree.getroot()

    has_fake = any(j.get("name") == "joint_fake" for j in root.findall("joint"))
    if not has_fake:
        snippet = ET.fromstring(
            "<root>"
            '<link name="fake_link_x">'
            "  <inertial>"
            '    <origin rpy="0 0 0" xyz="0 0 0"/>'
            '    <mass value="0.749143203376"/>'
            '    <inertia ixx="0.071" ixy="-0.004" ixz="0" iyy="0.0004" iyz="-0.003" izz="0.071"/>'
            "  </inertial>"
            "</link>"
            '<joint name="joint_fake" type="prismatic">'
            '  <origin rpy="0 0 0" xyz="0 0 0"/>'
            '  <axis xyz="1 0 0"/>'
            '  <parent link="base_link"/>'
            '  <child link="fake_link_x"/>'
            '  <limit effort="100" lower="-1" upper="1.1" velocity="1"/>'
            "</joint>"
            "</root>"
        )
        for elem in snippet:
            root.append(elem)

        for joint in root.findall("joint"):
            if joint.get("name") == "joint_mast":
                parent = joint.find("parent")
                if parent is not None:
                    parent.set("link", "fake_link_x")
                break

    tree.write(urdf_path, xml_declaration=True, encoding="utf-8")
    logger.info("Generated %s", urdf_path)
    return urdf_path


# Stretch stuff
MANIP_STRETCH_URDF = _ensure_stretch_urdf(get_full_config_path("urdf/stretch.urdf"))

# This is the gripper, and the distance in the gripper frame to where the fingers will roughly meet
STRETCH_GRASP_FRAME = "link_grasp_center"
STRETCH_CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"

# Offsets required for "link_straight_gripper" grasp frame
STRETCH_STANDOFF_DISTANCE = 0.235
STRETCH_STANDOFF_WITH_MARGIN = 0.25
# Offset from a predicted grasp point to STRETCH_GRASP_FRAME
STRETCH_GRASP_OFFSET = np.eye(4)
STRETCH_GRASP_OFFSET[:3, 3] = np.array([0, 0, -1 * STRETCH_STANDOFF_DISTANCE])
# Offset from STRETCH_GRASP_FRAME to predicted grasp point
STRETCH_TO_GRASP = np.eye(4)
STRETCH_TO_GRASP[:3, 3] = np.array([0, 0, STRETCH_STANDOFF_DISTANCE])

# For EXTEND_ARM action
STRETCH_ARM_EXTENSION = 0.8
STRETCH_ARM_LIFT = 0.8

STRETCH_HEAD_CAMERA_ROTATIONS = 3  # number of counterclockwise rotations for the head camera

# For EXTEND_ARM action
STRETCH_ARM_EXTENSION = 0.8
STRETCH_ARM_LIFT = 0.8

look_at_ee = np.array([-np.pi / 2, -np.pi / 4])
# Forward, slight down — room-scale view (not floor). Matches kinematics.look_front.
look_front = np.array([0.0, math.radians(-30)])
look_ahead = np.array([0.0, 0.0])
look_close = np.array([0.0, math.radians(-45)])
look_down = np.array([0.0, math.radians(-58)])


STRETCH_HOME_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.2,  # lift
        0.057,  # arm
        0.0,  # gripper rpy
        0.0,
        0.0,
        3.0,  # wrist,
        0.0,
        0.0,
    ]
)

# look down in navigation mode for doing manipulation post-navigation
STRETCH_POSTNAV_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        0.0,
        math.radians(-45),
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.78,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper pointed down, for a top-down grasp
STRETCH_DEMO_PREGRASP_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)

# Gripper straight out, lowered arm for clear vision
STRETCH_PREDEMO_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.4,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        0.0,  # wrist pitch
        0.0,  # wrist yaw
        -np.pi / 2,  # head pan, camera to face the arm
        -np.pi / 4,
    ]
)
# Navigation should not be fully folded up against the arm - in case its holding something.
# Head uses look_front (not look_down): agent/describe need a usable camera, not floor.
STRETCH_NAVIGATION_Q = np.array(
    [
        0,  # x
        0,  # y
        0,  # theta
        0.6,  # lift
        0.01,  # arm
        0.0,  # gripper rpy
        0.0,  # wrist roll
        -1.5,  # wrist pitch
        0.0,  # wrist yaw
        look_front[0],
        look_front[1],
    ]
)


PIN_CONTROLLED_JOINTS = [
    "base_x_joint",
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]

ROS_ARM_JOINTS = ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]
ROS_LIFT_JOINT = "joint_lift"
ROS_GRIPPER_FINGER = "joint_gripper_finger_left"
# ROS_GRIPPER_FINGER2 = "joint_gripper_finger_right"
ROS_HEAD_PAN = "joint_head_pan"
ROS_HEAD_TILT = "joint_head_tilt"
ROS_WRIST_YAW = "joint_wrist_yaw"
ROS_WRIST_PITCH = "joint_wrist_pitch"
ROS_WRIST_ROLL = "joint_wrist_roll"

stretch_degrees_of_freedom = 3 + 2 + 4 + 2
default_gripper_open_threshold: float = 0.3
