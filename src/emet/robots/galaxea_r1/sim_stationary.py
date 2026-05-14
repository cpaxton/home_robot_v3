# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""MuJoCo stationary control for Galaxea R1 / RB-Y1 (shared MJCF family).

Override methods on :class:`GalaxeaR1FamilyMujocoStationary` for family-specific actuator or wheel rules
while keeping the same :class:`emet.simulation.mujoco_stationary_control.MujocoStationaryControl` interface
used by all MuJoCo registry sims.
"""

from __future__ import annotations

from emet.simulation.mujoco_stationary_control import DefaultMujocoStationaryControl


class GalaxeaR1FamilyMujocoStationary(DefaultMujocoStationaryControl):
    """Galaxea R1 / rby1 stationary ``ctrl`` + hold; defaults match generic transmission logic."""
