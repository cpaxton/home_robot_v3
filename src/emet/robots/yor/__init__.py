# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""YOR robot — https://github.com/YOR-robot/YOR."""

from emet.robots.base import RobotBackend, RobotSpec


class YORBackend(RobotBackend):
    """Stub for YOR robot integration."""

    def get_spec(self) -> RobotSpec:
        raise NotImplementedError(
            "YOR integration is a stub. See https://github.com/YOR-robot/YOR"
        )

    def create_client(self, robot_ip: str, **kwargs):
        raise NotImplementedError("YOR client not yet implemented")

    def create_model(self, **kwargs):
        raise NotImplementedError("YOR model not yet implemented")


__all__ = ["YORBackend"]
