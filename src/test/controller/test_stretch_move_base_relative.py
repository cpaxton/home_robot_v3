# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stretch move_base_to must send absolute episode xyt (idempotent under reliable resend)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from src.test.controller import make_zmq_test_client

from emet.controller.zmq_client import StretchZmqClient
from emet.utils.geometry import xyt_base_to_global


def test_move_base_to_relative_sends_absolute_episode_xyt():
    client = make_zmq_test_client(StretchZmqClient)
    pose = np.array([1.0, 2.0, 0.5], dtype=np.float64)
    client.get_base_pose = MagicMock(return_value=pose.copy())
    client.get_emet_session = MagicMock(return_value={"capabilities": {"teleport_base": True}})
    sent: dict = {}

    def _send(action, **_kwargs):
        sent.update(action)
        return {"step": 7, **action}

    client.send_action = MagicMock(side_effect=_send)
    client._wait_for_base_motion = MagicMock()

    delta = np.array([0.0, 0.0, np.pi / 4], dtype=np.float64)
    client.move_base_to(delta, relative=True, blocking=True, timeout=2.0, reliable=True)

    assert sent.get("nav_relative") is False
    assert sent.get("nav_teleport") is True
    expected = xyt_base_to_global(delta, pose)
    np.testing.assert_allclose(np.asarray(sent["xyt"], dtype=float), expected, atol=1e-9)
    client._wait_for_base_motion.assert_called_once()
    assert client._wait_for_base_motion.call_args.kwargs["goal_angle"] == float(expected[2])
