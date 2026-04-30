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
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""Minimal :class:`emet.motion.robot.RobotModel` from :class:`RobotSpec` (footprint + DOF only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.motion.robot import Footprint, RobotModel
from emet.robots.base import RobotSpec


class SpecRobotModel(RobotModel):
    """Kinematic planning placeholder for ZMQ sim robots without a full URDF :class:`RobotModel`.

    Exposes :meth:`get_footprint` and :meth:`get_dof` from the backend spec; :meth:`set_config` is
    a no-op (planners that require Bullet are not supported on generic sim yet).
    """

    def __init__(self, spec: RobotSpec):
        super().__init__(name=spec.name, urdf_path=spec.urdf_path)
        self._spec = spec

    def get_dof(self) -> int:
        return int(self._spec.dof)

    def set_config(self, q: Any) -> None:
        return

    def get_footprint(self) -> Footprint:
        return self._spec.footprint

    def get_config(self) -> np.ndarray:
        return np.zeros(self.get_dof(), dtype=np.float32)
