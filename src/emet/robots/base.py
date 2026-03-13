# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Base abstractions for robot backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from emet.core.robot import AbstractRobotClient
    from emet.motion.robot import RobotModel

from emet.motion.robot import Footprint


@dataclass
class RobotSpec:
    """Declarative config for a robot (DOF, cameras, URDF path, etc.)."""

    name: str
    dof: int
    joint_names: List[str]
    camera_names: List[str]
    urdf_path: Optional[str]
    footprint: Footprint
    mjcf_path: Optional[str] = None
    actuator_names: List[str] = field(default_factory=list)
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
