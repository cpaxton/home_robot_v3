from __future__ import annotations

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

from pathlib import Path
from typing import TYPE_CHECKING

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.utils.assets import get_robot_mjcf_path

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend


def _innate_mars_mjcf_path() -> str:
    """Same file as :func:`get_robot_mjcf_path` (importlib resource path), not a second path via __file__."""
    p = get_robot_mjcf_path("innate_mars")
    if p is None or not p.is_file():
        raise RuntimeError(
            "Innate Mars MJCF not found. Use a full emet install with package data, or a checkout where "
            "src/emet/assets/robot/innate_mars/innate_mars.xml exists."
        )
    return str(p.resolve())


def _innate_mars_urdf_path() -> str | None:
    """Vendored ``maurice.urdf`` next to ``innate_mars.xml`` (same kinematics as innate-os)."""
    urdf = Path(_innate_mars_mjcf_path()).with_name("maurice.urdf")
    return str(urdf.resolve()) if urdf.is_file() else None


# Planar base + arm + mimic gripper; MJCF joint translations match ``maurice.urdf`` (RViz / innate-os).
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
            urdf_path=_innate_mars_urdf_path(),
            mjcf_path=_innate_mars_mjcf_path(),
            actuator_names=list(INNATE_MARS_ACTUATOR_NAMES),
            base_link_name="base_link",
            footprint=Footprint(width=0.48, length=0.48, width_offset=0.0, length_offset=0.0),
            optional_uv_extras=(),
            dynamem_depth_source_hint="da3",
            planar_base_joint_names=("base_x", "base_y", "base_yaw"),
            # MJCF head cameras align with ROS optical (+X mount forward); MuJoCo Renderer buffers are upright
            # on typical EGL/GL backends. Avoid baked flip/rot here—extra ops distort stereo/overlays. If pixels
            # look upside-down on your GPU, export EMET_ROBOSUITE_RENDER_FLIPUD=1 before starting the robosuite
            # server (only applies when ops is empty; see RobosuiteZmqServer).
            # MuJoCo Renderer buffers are vertically flipped vs OpenCV / ``imshow``; match ZMQ + preview-cameras.
            robosuite_rgb_depth_ops=("flipud",),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def get_emote_backend(self) -> EmoteBackend:
        from emet.robots.innate_mars.emote_backend import InnateMarsEmoteBackend

        return InnateMarsEmoteBackend()

    def create_model(self, **kwargs):
        raise NotImplementedError(
            "Innate Mars kinematic model not yet implemented. Use MuJoCo-based planning or a third-party IK solver."
        )


from .dummy_client import DummyInnateMarsClient
from .ros_client import InnateMarsRosRobotClient

__all__ = [
    "INNATE_MARS_JOINT_NAMES",
    "DummyInnateMarsClient",
    "InnateMarsBackend",
    "InnateMarsRosRobotClient",
]
