# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from emet.simulation.mujoco_server_stretch import MujocoZmqServer


@pytest.mark.parametrize("name", ["_control_thread", "_send_thread", "_recv_thread"])
def test_shutdown_does_not_join_its_calling_thread(monkeypatch, name):
    monkeypatch.setattr("emet.simulation.mujoco_server_stretch.time.sleep", lambda _: None)
    server = SimpleNamespace(robot_sim=Mock())
    setattr(server, name, threading.current_thread())
    MujocoZmqServer.stop(server)
    assert server._done
    server.robot_sim.stop.assert_called_once()
