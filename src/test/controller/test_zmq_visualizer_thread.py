# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace

import pytest

from emet.controller.zmq_client import StretchZmqClient
from emet.visualization.null_visualizer import NullVisualizer


@pytest.mark.parametrize("visualizer", [None, NullVisualizer(), SimpleNamespace(enabled=True)])
def test_visualizer_thread_requires_enabled_visualizer(monkeypatch, visualizer):
    client = object.__new__(StretchZmqClient)
    client._started = False
    client._zmq_closed = False
    client._rerun = visualizer
    client._obs = {"rgb": "present"}
    client._state = {"is_homed": True, "is_runstopped": False}
    client._servo = object()
    client._obs_lock = threading.Lock()
    client._state_lock = threading.Lock()
    client._verify_emet_robot_id_stretch = lambda: True
    client._note_emet_session_from_zmq_dict = lambda _: None
    targets = []
    monkeypatch.setattr(
        "emet.controller.zmq_client.threading.Thread",
        lambda *, target, daemon: SimpleNamespace(start=lambda: targets.append(target.__name__)),
    )
    try:
        assert client.start()
        expected = ["blocking_spin", "blocking_spin_state", "blocking_spin_servo"]
        if getattr(visualizer, "enabled", False) is True:
            assert targets == expected + ["blocking_spin_rerun"]
            return
        assert targets == expected
        # Also safe if invoked directly: must not access observations or busy-loop.
        client.blocking_spin_rerun()
    finally:
        client._zmq_closed = True
