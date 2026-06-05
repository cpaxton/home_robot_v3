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

"""Stretch 3 robot backend — https://hello-robot.com/stretch-3-product."""

from __future__ import annotations

from typing import TYPE_CHECKING

from emet.motion.constants import MANIP_STRETCH_URDF, stretch_degrees_of_freedom
from emet.motion.kinematics import HelloStretchKinematics
from emet.robots.base import RobotBackend, RobotSpec

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend
    from emet.controller.zmq_client import StretchZmqClient
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path

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
STRETCH_ROBOCASA_CAMERA_NAMES = ["d435i_camera_rgb"]
STRETCH_ROBOCASA_MJCF_JOINT_NAMES = [
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    "joint_head_pan",
    "joint_head_tilt",
]


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
            sim_uses_stretch_mujoco_zmq=True,
        )

    def get_robosuite_robocasa_spec(self) -> RobotSpec:
        """RobotSpec for ``RobosuiteZmqServer`` + Robocasa merged Stretch MJCF (freejoint base)."""
        mjcf = get_robot_mjcf_path("stretch")
        return RobotSpec(
            name="stretch",
            dof=len(STRETCH_ROBOCASA_MJCF_JOINT_NAMES),
            joint_names=list(STRETCH_ROBOCASA_MJCF_JOINT_NAMES),
            camera_names=list(STRETCH_ROBOCASA_CAMERA_NAMES),
            urdf_path=MANIP_STRETCH_URDF,
            mjcf_path=str(mjcf.resolve()) if mjcf is not None else None,
            actuator_names=[],
            base_link_name="base_link",
            footprint=Footprint(width=0.34, length=0.33, width_offset=0.0, length_offset=-0.1),
            robosuite_rgb_depth_ops=("flipud",),
        )

    def create_client(self, robot_ip: str, **kwargs) -> StretchZmqClient:
        from emet.controller.zmq_client import StretchZmqClient

        return StretchZmqClient(robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs) -> HelloStretchKinematics:
        return HelloStretchKinematics(**kwargs)

    def get_emote_backend(self) -> EmoteBackend:
        from emet.controller.emotes.backend import StretchEmoteBackend

        return StretchEmoteBackend()

    def create_mujoco_stationary_control(self):
        from emet.robots.stretch.sim_stationary import StretchMujocoStationary

        return StretchMujocoStationary()


__all__ = ["StretchBackend"]
