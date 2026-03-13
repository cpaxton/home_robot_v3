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
# Rainbow RB-Y1 robot backend — same hardware as Galaxea R1 (two-armed mobile manipulator).
# MolmoSpaces uses the id "rby1". This module provides a first-class robot backend so that
# emet serve mujoco --robot rby1 and emet run dynamem --robot rby1 use GenericZmqClient
# with the same MJCF as galaxea_r1.

from __future__ import annotations

# Reuse Galaxea R1 spec (same hardware); only the backend name differs for MolmoSpaces/CLI.
from pathlib import Path

from emet.robots.base import RobotBackend, RobotSpec
from emet.robots.footprint import Footprint
from emet.robots.galaxea_r1 import (
    R1_ACTUATOR_NAMES,
    R1_CAMERA_NAMES,
    R1_JOINT_NAMES,
)

# Same MJCF as Galaxea R1 (Rainbow RB-Y1 = Galaxea R1 hardware).
_assets_dir = Path(__file__).resolve().parents[2] / "assets" / "robot" / "galaxea_r1"
_MJCF_PATH = str(_assets_dir / "galaxea_r1.xml")


class Rby1Backend(RobotBackend):
    """Rainbow RB-Y1 backend (Galaxea R1 family). Uses same MJCF and GenericZmqClient as galaxea_r1."""

    def get_spec(self) -> RobotSpec:
        return RobotSpec(
            name="rby1",
            dof=26,
            joint_names=R1_JOINT_NAMES,
            camera_names=R1_CAMERA_NAMES,
            urdf_path=None,
            mjcf_path=_MJCF_PATH,
            actuator_names=R1_ACTUATOR_NAMES,
            base_link_name="base_link",
            footprint=Footprint(width=0.56, length=0.50, width_offset=0.0, length_offset=0.0),
        )

    def create_client(self, robot_ip: str, **kwargs):
        from emet.controller.generic_zmq_client import GenericZmqClient

        return GenericZmqClient(robot_spec=self.get_spec(), robot_ip=robot_ip, **kwargs)

    def create_model(self, **kwargs):
        raise NotImplementedError(
            "RB-Y1 kinematic model not yet implemented. Use MuJoCo-based IK or a third-party IK solver."
        )


__all__ = ["Rby1Backend"]
