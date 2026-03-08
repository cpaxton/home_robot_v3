# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Molmo Spaces simulator — https://github.com/allenai/molmo-spaces."""

from emet.simulators.base import BaseSimulatorServer


class MolmoSpacesSimulator(BaseSimulatorServer):
    """Stub for Molmo Spaces simulator integration."""

    def get_robot_spec(self):
        raise NotImplementedError(
            "Molmo Spaces integration is a stub. See https://github.com/allenai/molmo-spaces"
        )

    def get_full_observation_message(self):
        raise NotImplementedError("Molmo Spaces not yet implemented")

    def get_state_message(self):
        raise NotImplementedError("Molmo Spaces not yet implemented")

    def get_servo_message(self):
        raise NotImplementedError("Molmo Spaces not yet implemented")

    def handle_action(self, action):
        raise NotImplementedError("Molmo Spaces not yet implemented")

    def is_running(self) -> bool:
        raise NotImplementedError("Molmo Spaces not yet implemented")


__all__ = ["MolmoSpacesSimulator"]
