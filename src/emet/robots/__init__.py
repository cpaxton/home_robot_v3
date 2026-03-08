# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Robot backends for EMET — Stretch, Mobile ALOHA, Galaxea R1, Innate Mars, YOR."""

from emet.robots.base import RobotBackend, RobotSpec

ROBOT_REGISTRY = {
    "stretch": "emet.robots.stretch",
    "mobile_aloha": "emet.robots.mobile_aloha",
    "galaxea_r1": "emet.robots.galaxea_r1",
    "innate_mars": "emet.robots.innate_mars",
    "yor": "emet.robots.yor",
}

__all__ = ["RobotBackend", "RobotSpec", "ROBOT_REGISTRY"]
