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

"""Mobile ALOHA robot — https://github.com/mobile-aloha/mobile-aloha."""

from emet.robots.base import RobotBackend, RobotSpec


class MobileALOHABackend(RobotBackend):
    """Stub for Mobile ALOHA robot integration."""

    def get_spec(self) -> RobotSpec:
        raise NotImplementedError(
            "Mobile ALOHA integration is a stub. See https://github.com/mobile-aloha/mobile-aloha"
        )

    def create_client(self, robot_ip: str, **kwargs):
        raise NotImplementedError("Mobile ALOHA client not yet implemented")

    def create_model(self, **kwargs):
        raise NotImplementedError("Mobile ALOHA model not yet implemented")


__all__ = ["MobileALOHABackend"]
