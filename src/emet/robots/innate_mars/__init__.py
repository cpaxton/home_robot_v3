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

"""Innate Mars mobile manipulator — MuJoCo assets vendored as Maurice; real robot via innate_mars_bridge."""

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path


def _innate_mars_mjcf_path() -> str:
    """Same file as :func:`get_robot_mjcf_path` (importlib resource path), not a second path via __file__."""
    p = get_robot_mjcf_path("innate_mars")
    if p is None or not p.is_file():
        raise RuntimeError(
            "Innate Mars MJCF not found. Use a full emet install with package data, or a checkout where "
            "src/emet/assets/robot/innate_mars/innate_mars.xml exists."
        )
    return str(p.resolve())

# Matches `maurice.mjcf`: planar base + arm + mimic gripper joint.
INNATE_MARS_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint6M",
]

INNATE_MARS_ACTUATOR_NAMES = [
    "base_x",
    "base_y",
    "base_yaw",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

INNATE_MARS_CAMERA_NAMES = ["head_left", "head_right", "camera_arm"]


class InnateMarsBackend(RobotBackend):
    """Innate Mars / Maurice-style mobile manipulator."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="innate_mars",
            dof=len(INNATE_MARS_JOINT_NAMES),
            joint_names=list(INNATE_MARS_JOINT_NAMES),
            camera_names=list(INNATE_MARS_CAMERA_NAMES),
            urdf_path=None,
            mjcf_path=_innate_mars_mjcf_path(),
            actuator_names=list(INNATE_MARS_ACTUATOR_NAMES),
            base_link_name="base_link",
            footprint=Footprint(width=0.48, length=0.48, width_offset=0.0, length_offset=0.0),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        raise NotImplementedError(
            "Innate Mars kinematic model not yet implemented. Use MuJoCo-based planning or a third-party IK solver."
        )


__all__ = ["InnateMarsBackend", "INNATE_MARS_JOINT_NAMES"]
