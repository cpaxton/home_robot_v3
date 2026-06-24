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

"""Franka FR3 tabletop arm — MuJoCo Menagerie assets."""

from pathlib import Path

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "robot" / "franka_fr3"
_MJCF_PATH = str(_ASSETS_DIR / "fr3.xml")

FRANKA_FR3_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

FRANKA_FR3_ACTUATOR_NAMES = list(FRANKA_FR3_JOINT_NAMES)
FRANKA_FR3_CAMERA_NAMES = ["wrist_camera"]


class FrankaFR3Backend(RobotBackend):
    """Franka FR3 fixed-base manipulator (MolmoBot tabletop data alignment)."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="franka_fr3",
            dof=7,
            joint_names=list(FRANKA_FR3_JOINT_NAMES),
            camera_names=list(FRANKA_FR3_CAMERA_NAMES),
            urdf_path=None,
            mjcf_path=_MJCF_PATH,
            actuator_names=list(FRANKA_FR3_ACTUATOR_NAMES),
            base_link_name="base",
            footprint=Footprint(width=0.20, length=0.20, width_offset=0.0, length_offset=0.0),
            robosuite_rgb_depth_ops=("flipud",),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        raise NotImplementedError("Franka FR3 kinematic model not yet implemented in emet.")


__all__ = [
    "FRANKA_FR3_ACTUATOR_NAMES",
    "FRANKA_FR3_CAMERA_NAMES",
    "FRANKA_FR3_JOINT_NAMES",
    "FrankaFR3Backend",
]
