# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Innate Mars robot — https://github.com/innate-robotics."""

from emet.robots.base import RobotBackend, RobotSpec


class InnateMarsBackend(RobotBackend):
    """Stub for Innate Mars robot integration."""

    def get_spec(self) -> RobotSpec:
        raise NotImplementedError(
            "Innate Mars integration is a stub. See https://github.com/innate-robotics"
        )

    def create_client(self, robot_ip: str, **kwargs):
        raise NotImplementedError("Innate Mars client not yet implemented")

    def create_model(self, **kwargs):
        raise NotImplementedError("Innate Mars model not yet implemented")


__all__ = ["InnateMarsBackend"]
