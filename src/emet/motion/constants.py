# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import math
import os

import numpy as np

from emet.utils.config import get_full_config_path

# Stretch stuff: use config urdf if present, else generate from hello-robot-stretch-urdf package
_CONFIG_STRETCH_URDF = get_full_config_path("urdf/stretch.urdf")

# XML snippet added by dynamem (OK-Robot manipulation stack): joint_fake so joint_mast is child
_JOINT_FAKE_SNIPPET = """
    <link name="fake_link_x">
        <inertial>
            <origin rpy="0.0 0.0 0." xyz="0. 0. 0."/>
            <mass value="0.749143203376"/>
            <inertia ixx="0.0709854511955" ixy="-0.00433428742758" ixz="-0.000186110788698" iyy="0.000437922053343" iyz="-0.00288788257713" izz="0.0711048085017"/>
        </inertial>
    </link>
    <joint name="joint_fake" type="prismatic">
        <origin rpy="0. 0. 0." xyz="0. 0. 0."/>
        <axis xyz="1.0 0.0 0.0"/>
        <parent link="base_link"/>
        <child link="fake_link_x"/>
        <limit effort="100.0" lower="-1.0" upper="1.1" velocity="1.0"/>
    </joint>
"""


def _generate_dynamem_stretch_urdf(package_urdf_path: str, out_path: str) -> None:
    """Write a Dynamem-modified Stretch URDF (joint_fake + mesh paths) to out_path."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(package_urdf_path)
    root = tree.getroot()
    pkg_dir = os.path.dirname(os.path.abspath(package_urdf_path))

    # Resolve mesh paths relative to package so the generated file works from any cwd
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and not os.path.isabs(fn):
            mesh.set("filename", os.path.normpath(os.path.join(pkg_dir, fn)))

    # Add joint_fake if not present (same logic as config/dynamem_urdf.py)
    has_fake = any(j.get("name") == "joint_fake" for j in root.findall("joint"))
    if not has_fake:
        snippet_root = ET.fromstring(f"<root>{_JOINT_FAKE_SNIPPET}</root>")
        for el in snippet_root:
            root.append(el)
        for joint in root.findall("joint"):
            if joint.get("name") == "joint_mast":
                parent = joint.find("parent")
                if parent is not None:
                    parent.set("link", "fake_link_x")
                break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tree.write(out_path, xml_declaration=True, encoding="utf-8", default_namespace="")


def get_stretch_urdf_path() -> str:
    """Path to a valid Stretch URDF for kinematics (config or generated with joint_fake)."""
    if os.path.isfile(_CONFIG_STRETCH_URDF):
        return _CONFIG_STRETCH_URDF
    try:
        import stretch_urdf

        pkg_dir = os.path.dirname(stretch_urdf.__file__)
        # Prefer dex_wrist URDF so joint_wrist_pitch and joint_wrist_roll exist (Dynamem default_manip_mode_controlled_joints)
        for rel in (
            "RE1V0/stretch_description_RE1V0_tool_stretch_dex_wrist.urdf",
            "RE2V0/stretch_description_RE2V0_tool_stretch_dex_wrist.urdf",
            "RE1V0/stretch_description_RE1V0_tool_stretch_gripper.urdf",
        ):
            fallback = os.path.join(pkg_dir, rel)
            if os.path.isfile(fallback):
                _generate_dynamem_stretch_urdf(fallback, _CONFIG_STRETCH_URDF)
                return _CONFIG_STRETCH_URDF
        re1 = os.path.join(pkg_dir, "RE1V0")
        if os.path.isdir(re1):
            for name in sorted(os.listdir(re1)):
                if name.endswith(".urdf"):
                    _generate_dynamem_stretch_urdf(os.path.join(re1, name), _CONFIG_STRETCH_URDF)
                    return _CONFIG_STRETCH_URDF
    except Exception:
        pass
    return _CONFIG_STRETCH_URDF


MANIP_STRETCH_URDF = get_stretch_urdf_path()

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
look_front = np.array([0.0, -np.pi / 4])
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
# Navigation should not be fully folded up against the arm - in case its holding something
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
        0.0,
        math.radians(-65),
        # look_close[1],
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
