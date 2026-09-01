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

"""Sourccey mobile manipulator (Vulcan Robotics) — vendored MJCF + URDF + ZMQ client.

Assets derived from the updated official hardware repo
https://github.com/vulcan-forge/sourccey-hardware (``URDF/ArmLeft/ArmLeft.urdf`` arm
kinematics + meshes; converted to ``sourccey.xml`` by ``scripts/robot_assets/``).
See ``src/emet/assets/robot/sourccey/NOTICE.md`` and ``docs/robots/sourccey.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from emet.robots.base import ArmChain, RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend


def _sourccey_mjcf_path() -> str:
    """Same file as :func:`get_robot_mjcf_path` (importlib resource path)."""
    p = get_robot_mjcf_path("sourccey")
    if p is None or not p.is_file():
        raise RuntimeError(
            "Sourccey MJCF not found. Use a full emet install with package data, or a checkout where "
            "src/emet/assets/robot/sourccey/sourccey.xml exists."
        )
    return str(p.resolve())


def _sourccey_urdf_path() -> str | None:
    """Vendored official left-arm URDF (canonical; the right arm is its code-side mirror)."""
    mjcf = Path(_sourccey_mjcf_path())
    urdf = mjcf.parent / "urdf" / "ArmLeft" / "ArmLeft.urdf"
    return str(urdf.resolve()) if urdf.is_file() else None


# Planar base + lift + dual 5-DOF arms + grippers. Order matches ``sourccey.xml`` joints.
SOURCCEY_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "lift",
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]

SOURCCEY_ACTUATOR_NAMES = [
    "base_x_act",
    "base_y_act",
    "base_yaw_act",
    "lift_act",
    "left_shoulder_pan_act",
    "left_shoulder_lift_act",
    "left_elbow_flex_act",
    "left_wrist_flex_act",
    "left_wrist_roll_act",
    "left_gripper_act",
    "right_shoulder_pan_act",
    "right_shoulder_lift_act",
    "right_elbow_flex_act",
    "right_wrist_flex_act",
    "right_wrist_roll_act",
    "right_gripper_act",
]

SOURCCEY_CAMERA_NAMES = ["front_left", "front_right", "wrist_left", "wrist_right"]

# Gripper joint -> actuator name (single revolute gear per side).
SOURCCEY_GRIPPER_JOINTS = {"left": "left_gripper", "right": "right_gripper"}
SOURCCEY_GRIPPER_ACTUATORS = {"left": "left_gripper_act", "right": "right_gripper_act"}

# Home keyframe used by robosuite_load_utils / spawns (arms tucked).
SOURCCEY_HOME_KEYFRAME = "sourccey_home"

# Per-arm IK chain (shoulder_pan … wrist_roll). The gripper is a separate actuator
# so position IK cannot chew the fingers; ``_set_gripper`` drives ``{side}_gripper_act``.
_ARM_IK_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def _sourccey_arm_chain(side: str) -> ArmChain:
    joints = tuple(f"{side}_{s}" for s in _ARM_IK_SUFFIXES)
    acts = tuple(f"{side}_{s}_act" for s in (*_ARM_IK_SUFFIXES, "gripper"))
    return ArmChain(
        joint_names=joints,
        ee_body=f"{side}_Gripper-Finger",
        actuator_names=acts,
        link_bodies=(
            f"{side}_Arm-Base-Shoulder",
            f"{side}_Arm-Bicep",
            f"{side}_Arm-Forearm",
            f"{side}_Arm-Wrist",
            f"{side}_Gripper-Base",
            f"{side}_Gripper-Finger",
        ),
        gripper_bodies=(f"{side}_Gripper-Finger",),
    )


class SourcceyBackend(RobotBackend):
    """Sourccey: planar base + vertical lift + dual 5-DOF arms + grippers.

    Sim runs through :class:`~emet.simulation.robosuite_server.RobosuiteZmqServer` on the
    vendored MJCF; kinematic pick/place (``capabilities.kinematic_manip``) is advertised.
    Real-hardware ZMQ support is a stub for now; ``create_client`` returns a
    ``GenericZmqClient`` so joint/gripper/head plumbing has a target once a real
    bridge exists.
    """

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="sourccey",
            dof=len(SOURCCEY_JOINT_NAMES),
            joint_names=list(SOURCCEY_JOINT_NAMES),
            camera_names=list(SOURCCEY_CAMERA_NAMES),
            urdf_path=_sourccey_urdf_path(),
            mjcf_path=_sourccey_mjcf_path(),
            actuator_names=list(SOURCCEY_ACTUATOR_NAMES),
            base_link_name="base_root",
            footprint=Footprint(width=0.42, length=0.42, width_offset=0.0, length_offset=0.0),
            planar_base_joint_names=("base_x", "base_y", "base_yaw"),
            arm_chain=_sourccey_arm_chain("left"),
            arm_chains={"left": _sourccey_arm_chain("left"), "right": _sourccey_arm_chain("right")},
            advertise_kinematic_manip=True,
            # arm meshes are visual-only in the MJCF; inflate clip erosion + require EE XY inside floor.
            planar_spawn_xy_extra_margin_m=0.25,
            planar_spawn_clip_guard_body_names=(
                "left_Gripper-Base",
                "right_Gripper-Base",
                "left_Arm-Wrist",
                "right_Arm-Wrist",
            ),
            planar_spawn_clip_guard_pad_m=0.25,
            planar_spawn_robocasa_first_clearance_m=0.068,
            robosuite_rgb_depth_ops=("flipud",),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def get_emote_backend(self) -> EmoteBackend:
        from emet.controller.emotes.backend import GenericEmoteBackend

        return GenericEmoteBackend(self.get_spec().name)

    def create_model(self, **kwargs):
        from emet.robots.spec_robot_model import SpecRobotModel

        return SpecRobotModel(self.get_spec())


__all__ = [
    "SOURCCEY_ACTUATOR_NAMES",
    "SOURCCEY_CAMERA_NAMES",
    "SOURCCEY_GRIPPER_ACTUATORS",
    "SOURCCEY_GRIPPER_JOINTS",
    "SOURCCEY_HOME_KEYFRAME",
    "SOURCCEY_JOINT_NAMES",
    "SourcceyBackend",
    "_sourccey_arm_chain",
]
