# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Behavior 1k simulator — https://github.com/behavior-1k/behavior-1k."""

from emet.simulators.base import BaseSimulatorServer


class Behavior1kSimulator(BaseSimulatorServer):
    """Stub for Behavior 1k simulator integration."""

    def get_robot_spec(self):
        raise NotImplementedError(
            "Behavior 1k integration is a stub. See https://github.com/behavior-1k/behavior-1k"
        )

    def get_full_observation_message(self):
        raise NotImplementedError("Behavior 1k not yet implemented")

    def get_state_message(self):
        raise NotImplementedError("Behavior 1k not yet implemented")

    def get_servo_message(self):
        raise NotImplementedError("Behavior 1k not yet implemented")

    def handle_action(self, action):
        raise NotImplementedError("Behavior 1k not yet implemented")

    def is_running(self) -> bool:
        raise NotImplementedError("Behavior 1k not yet implemented")


__all__ = ["Behavior1kSimulator"]
