# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from emet.core.server import BaseZmqServer


@pytest.mark.parametrize("extra", [{}, {"posture": "navigation"}, {"head_to": [0, 0]}])
def test_repeated_relative_navigation_dispatches_once(extra):
    received = []
    action = {"step": 7, "xyt": [0, 0, 0.785], "nav_relative": True, **extra}
    server = SimpleNamespace(
        verbose=False,
        skip_duplicate_steps=True,
        _last_step=-1,
        is_running=Mock(side_effect=[True, True, False]),
        recv_socket=SimpleNamespace(recv_pyobj=Mock(side_effect=[action.copy(), action.copy()])),
        handle_action=lambda a: received.append(a),
    )
    BaseZmqServer.spin_recv(server)
    assert len(received) == 1
    assert server._last_step == 7


def test_stretch_late_image_cannot_regress_acknowledgement():
    from emet.controller.zmq_client import StretchZmqClient

    client = object.__new__(StretchZmqClient)
    client._obs_lock = threading.Lock()
    client._last_step = 8
    client._iter = 9
    client._note_emet_session_from_zmq_dict = Mock()
    client._update_obs({"step": 3})
    assert client._last_step == 8


@pytest.mark.parametrize("client_kind", ["generic", "stretch"])
def test_acknowledgement_wait_is_bounded(client_kind):
    if client_kind == "generic":
        from emet.controller.generic_zmq_client import GenericZmqClient as Client
    else:
        from emet.controller.zmq_client import StretchZmqClient as Client
    client = object.__new__(Client)
    client._act_lock = threading.Lock()
    client._iter = 0
    client._last_step = -1
    client.send_message = Mock()
    with patch("time.monotonic", side_effect=[0.0, 1.0]):
        with pytest.raises(TimeoutError, match="outcome unknown"):
            client.send_action({"xyt": [0, 0, 1]}, timeout=0.5)
    client.send_message.assert_called_once()


@pytest.mark.parametrize("client_kind", ["generic", "stretch"])
def test_retry_keeps_same_identity(client_kind):
    if client_kind == "generic":
        from emet.controller.generic_zmq_client import GenericZmqClient as Client
    else:
        from emet.controller.zmq_client import StretchZmqClient as Client
    client = object.__new__(Client)
    client._act_lock = threading.Lock()
    client._iter = 0
    client._last_step = -1
    sent = []

    def send(action):
        sent.append(action.copy())
        if len(sent) == 2:
            client._last_step = action["step"]

    client.send_message = send
    with patch("time.sleep"):
        client.send_action({"xyt": [0, 0, 1]})
    assert len(sent) == 2 and sent[0] == sent[1]
