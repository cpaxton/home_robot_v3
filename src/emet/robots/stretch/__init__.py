# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Stretch 3 robot backend — https://hello-robot.com/stretch-3-product."""

from emet.controller.zmq_client import HomeRobotZmqClient
from emet.motion.constants import MANIP_STRETCH_URDF, stretch_degrees_of_freedom
from emet.motion.kinematics import HelloStretchKinematics
from emet.motion.robot import Footprint
from emet.robots.base import RobotBackend, RobotSpec

STRETCH_JOINT_NAMES = [
    "base_x",
    "base_y",
    "base_theta",
    "lift",
    "arm",
    "gripper",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
    "head_pan",
    "head_tilt",
]
STRETCH_CAMERA_NAMES = ["head_camera", "ee_camera"]


class StretchBackend(RobotBackend):
    """Stretch 3 robot backend. Uses existing ZMQ client and kinematics."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="stretch",
            dof=stretch_degrees_of_freedom,
            joint_names=STRETCH_JOINT_NAMES,
            camera_names=STRETCH_CAMERA_NAMES,
            urdf_path=MANIP_STRETCH_URDF,
            footprint=Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1),
        )

    def create_client(self, robot_ip: str, **kwargs) -> HomeRobotZmqClient:
        return HomeRobotZmqClient(robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs) -> HelloStretchKinematics:
        return HelloStretchKinematics(**kwargs)


__all__ = ["StretchBackend"]
