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

"""Robot backends for EMET — Stretch, Mobile ALOHA, Galaxea R1 / RB-Y1, Innate Mars, YOR."""

from emet.robots.base import RobotBackend, RobotSpec

ROBOT_REGISTRY = {
    "stretch": "emet.robots.stretch",
    "mobile_aloha": "emet.robots.mobile_aloha",
    "galaxea_r1": "emet.robots.galaxea_r1",
    "rby1": "emet.robots.rby1",  # Rainbow RB-Y1 (Galaxea R1 family); MolmoSpaces id
    "rb_y1": "emet.robots.rby1",  # same, for --robot rb-y1
    "innate_mars": "emet.robots.innate_mars",
    "yor": "emet.robots.yor",
}

__all__ = ["RobotBackend", "RobotSpec", "ROBOT_REGISTRY"]
