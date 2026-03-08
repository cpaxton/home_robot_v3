# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Galaxea R1 robot — https://github.com/galaxea-robotics."""

from emet.robots.base import RobotBackend, RobotSpec


class GalaxeaR1Backend(RobotBackend):
    """Stub for Galaxea R1 robot integration."""

    def get_spec(self) -> RobotSpec:
        raise NotImplementedError(
            "Galaxea R1 integration is a stub. See https://github.com/galaxea-robotics"
        )

    def create_client(self, robot_ip: str, **kwargs):
        raise NotImplementedError("Galaxea R1 client not yet implemented")

    def create_model(self, **kwargs):
        raise NotImplementedError("Galaxea R1 model not yet implemented")


__all__ = ["GalaxeaR1Backend"]
