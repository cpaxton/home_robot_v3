# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

"""MuJoCo stationary control for Stretch (used in :class:`emet.simulation.stretch_mujoco.mujoco_server.MujocoServer`)."""

from __future__ import annotations

from emet.simulation.mujoco_stationary_control import DefaultMujocoStationaryControl


class StretchMujocoStationary(DefaultMujocoStationaryControl):
    """Stretch-specific overrides for full-model ``ctrl`` + optional :class:`~emet.robots.base.RobotSpec` hold."""
