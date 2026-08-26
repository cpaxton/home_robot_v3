# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Controller package smoke tests

from __future__ import annotations

from threading import Lock


def make_zmq_test_client(cls):
    """Construct a StretchZmqClient via ``__new__`` (no ZMQ sockets) with locks initialized.

    Some unit tests build the client without ``__init__`` (which opens sockets); the
    methods under test touch ``_state_lock`` / ``_obs_lock`` / ``_send_lock`` (e.g.
    ``_sim_to_real_ratio`` → ``_scaled_motion_timeout``). Initialize those so the
    tests pass without a live robot.
    """
    client = cls.__new__(cls)
    for name in ("_obs_lock", "_act_lock", "_state_lock", "_servo_lock", "_send_lock", "_emet_session_lock"):
        setattr(client, name, Lock())
    client._obs = None
    client._servo = None
    client._state = None
    client._last_step = -1
    client._finish = False
    client._rerun = None
    return client
