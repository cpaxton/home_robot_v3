# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace

import numpy as np

from emet.simulation.robosuite_server import RobosuiteZmqServer


def test_generic_policy_reports_tipping_even_at_planar_goal():
    server = RobosuiteZmqServer.__new__(RobosuiteZmqServer)
    server._mj_lock = threading.RLock()
    server._spec = SimpleNamespace(base_link_name="base")
    server._mjdata = SimpleNamespace(time=12, body=lambda _: SimpleNamespace(xmat=np.diag([1, 0.7, 0.7]).ravel()))
    server._at_goal = True
    server._nav_goal_world = None
    server.get_base_xyt = lambda: [0, 0, 0]
    assert server.navigation_policy_measurement()["failure"] == "base posture unsafe"


def test_stretch_policy_uses_timestamped_pose_in_resolved_episode_frame():
    from emet.simulation.mujoco_server_stretch import MujocoZmqServer

    server = MujocoZmqServer.__new__(MujocoZmqServer)
    server._initial_xyt = np.array([2, 3, np.pi / 2])
    server._status = SimpleNamespace(time=12, base=SimpleNamespace(x=2, y=4, theta=np.pi / 2))
    server.base_controller_at_goal = lambda: True
    server.active = False
    sample = server.navigation_policy_measurement()
    np.testing.assert_allclose(sample["pose"], [1, 0, 0], atol=1e-8)
    assert sample["timestamp"] == 12
    assert sample["stopped"]
