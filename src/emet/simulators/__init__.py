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

"""Simulator backends for EMET — MuJoCo, Behavior 1k, Molmo Spaces."""

from emet.simulators.base import BaseSimulatorServer

SIMULATOR_REGISTRY = {
    "mujoco": "emet.simulators.mujoco",
    "behavior1k": "emet.simulators.behavior1k",
    "molmo_spaces": "emet.simulators.molmo_spaces",
}

__all__ = ["BaseSimulatorServer", "SIMULATOR_REGISTRY"]
