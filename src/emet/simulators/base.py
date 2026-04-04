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

"""Base abstraction for simulator servers."""

from abc import ABC, abstractmethod
from typing import Any

from emet.robots.base import RobotSpec


class BaseSimulatorServer(ABC):
    """Server that publishes observations and receives actions (ZMQ or other)."""

    @abstractmethod
    def get_robot_spec(self) -> RobotSpec:
        """Which robot does this simulator emulate?"""
        ...

    @abstractmethod
    def get_full_observation_message(self) -> dict[str, Any]:
        """Get the full observation message (images, depth, state, etc.)."""
        ...

    @abstractmethod
    def get_state_message(self) -> dict[str, Any]:
        """Get a compact state message (e.g. joint positions, homed status)."""
        ...

    @abstractmethod
    def get_servo_message(self) -> dict[str, Any]:
        """Get messages for visual servoing (lower-res images, ee camera)."""
        ...

    @abstractmethod
    def handle_action(self, action: dict[str, Any]) -> None:
        """Apply the received action to the simulation."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the server is running."""
        ...
