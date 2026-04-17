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

"""Base abstractions for robot backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emet.controller.emotes.backend import EmoteBackend
    from emet.core.robot import AbstractRobotClient
    from emet.motion.robot import RobotModel

from emet.robots.footprint import Footprint


@dataclass
class RobotSpec:
    """Declarative config for a robot (DOF, cameras, URDF path, etc.)."""

    name: str
    dof: int
    joint_names: list[str]
    camera_names: list[str]
    urdf_path: str | None
    footprint: Footprint
    mjcf_path: str | None = None
    actuator_names: list[str] = field(default_factory=list)
    base_link_name: str = "base_link"


class RobotBackend(ABC):
    """Base class for robot-specific logic."""

    @abstractmethod
    def get_spec(self) -> RobotSpec:
        """Return the robot specification."""
        ...

    @abstractmethod
    def create_client(self, robot_ip: str, **kwargs: Any) -> "AbstractRobotClient":
        """Create a client for communicating with the robot (real or simulated)."""
        ...

    @abstractmethod
    def create_model(self, **kwargs: Any) -> "RobotModel":
        """Create a kinematic/planning model of the robot."""
        ...

    def get_emote_backend(self) -> "EmoteBackend":
        """Emote/gesture implementation for this robot (Stretch motion vs speech-only, etc.)."""
        from emet.controller.emotes.backend import GenericEmoteBackend

        return GenericEmoteBackend(self.get_spec().name)
