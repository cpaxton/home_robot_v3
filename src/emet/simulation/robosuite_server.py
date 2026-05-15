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
# This source code is licensed under the LICENSE file in the root directory of this source tree.

"""Backward-compatible name for the MuJoCo merged-MJCF ZMQ server (:class:`BaseMujocoZmqServer`)."""

from emet.simulation.base_mujoco_zmq_server import (
    _PRIMARY_RH,
    _PRIMARY_RW,
    _SERVO_RH,
    _SERVO_RW,
    BaseMujocoZmqServer,
)


class RobosuiteZmqServer(BaseMujocoZmqServer):
    """Legacy class name for :class:`BaseMujocoZmqServer`."""

    pass


__all__ = [
    "BaseMujocoZmqServer",
    "RobosuiteZmqServer",
    "_PRIMARY_RH",
    "_PRIMARY_RW",
    "_SERVO_RH",
    "_SERVO_RW",
]
