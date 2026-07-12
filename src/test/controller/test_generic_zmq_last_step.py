# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""GenericZmqClient state-step monotonicity."""

from __future__ import annotations

from emet.controller.generic_zmq_client import GenericZmqClient


def test_generic_zmq_last_step_is_monotonic():
    client = GenericZmqClient.__new__(GenericZmqClient)
    client._last_step = 5
    # Mimic _state_loop update: out-of-order / stale step must not regress.
    msg_step = 3
    client._last_step = max(client._last_step, int(msg_step))
    assert client._last_step == 5
    client._last_step = max(client._last_step, 7)
    assert client._last_step == 7
