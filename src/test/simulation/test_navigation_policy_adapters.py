# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from emet.simulation.robosuite_server import RobosuiteZmqServer


@pytest.mark.parametrize("name,xy,yaw", [(None, 0.07, 0.15), ("precision", 0.02, 0.03)])
def test_stretch_controller_tolerance_matches_command_contract(name, xy, yaw):
    from emet.simulation.mujoco_server_stretch import MujocoZmqServer

    server = MujocoZmqServer.__new__(MujocoZmqServer)
    server.controller = Mock()

    def install(action):
        server._contract_navigation_context = {"resolved_goal": action["xyt"]}

    server.handle_action = install
    action = {"xyt": [1, 2, 0]}
    if name:
        action["nav_policy"] = name
    assert server.start_navigation_command(action)["resolved_goal"] == [1, 2, 0]
    server.controller.control.set_linear_error_tolerance.assert_called_once_with(xy)
    server.controller.control.set_angular_error_tolerance.assert_called_once_with(yaw)


def test_generic_policy_reports_tipping_even_at_planar_goal():
    server = RobosuiteZmqServer.__new__(RobosuiteZmqServer)
    server._mj_lock = threading.RLock()
    server._spec = SimpleNamespace(base_link_name="base")
    server._mjdata = SimpleNamespace(time=12, body=lambda _: SimpleNamespace(xmat=np.diag([1, 0.7, 0.7]).ravel()))
    server._at_goal = True
    server._nav_goal_world = None
    server.get_base_xyt = lambda: [0, 0, 0]
    assert server.navigation_policy_measurement()["failure"] == "base posture unsafe"


@pytest.mark.parametrize("distance", [0.012, 0.052])
def test_stretch_manipulation_base_executes_fine_corrections_without_changing_navigation(distance):
    from emet.core.navigation_result import NAVIGATION_POLICIES
    from emet.simulation.mujoco_server_stretch import MujocoZmqServer

    server = MujocoZmqServer.__new__(MujocoZmqServer)
    server.control_mode = "manipulation"
    server._manip_xyt = np.zeros(3)
    server.robot_sim = Mock()
    server.controller = Mock()
    server.get_base_pose = lambda: np.zeros(3)
    server.set_goal_pose = Mock()
    server.manip_to(np.array([distance, 0.6, 0.1, 0, -0.3, 0]))
    np.testing.assert_allclose(server.set_goal_pose.call_args.args[0], [distance, 0, 0])
    server.controller.control.set_linear_error_tolerance.assert_called_once_with(0.005)
    server.controller.control.set_angular_error_tolerance.assert_called_once_with(0.03)
    assert NAVIGATION_POLICIES["precision"].xy_tolerance == 0.02
    assert server.controller.translation_only is True


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
