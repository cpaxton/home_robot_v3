"""Slow simulation must retain the controller's measured-time motion budget."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from emet.simulation.mujoco_server_stretch import MujocoZmqServer


class Status(dict):
    __getattr__ = dict.__getitem__


def test_controller_timeout_uses_simulation_time(monkeypatch):
    monkeypatch.setattr("emet.simulation.mujoco_server_stretch.timeit.default_timer", lambda: 100.0)
    controller = Mock()
    controller.compute_control.return_value = (.1, .2)
    controller.is_done.return_value = False
    controller.timeout.side_effect = lambda elapsed: elapsed > 50
    robot = Mock()
    server = SimpleNamespace(
        _status=Status(time=12., base=SimpleNamespace(x_vel=.1, theta_vel=.2)),
        debug_control_loop=False, get_base_pose=lambda: np.zeros(3), controller=controller,
        active=True, xyt_goal=np.ones(3), goal_set_t=0., goal_set_sim_t=10.,
        controller_finished=False, robot_sim=robot,
    )
    MujocoZmqServer._control_loop_callback(server)
    controller.timeout.assert_called_once_with(2.)
    robot.set_base_velocity.assert_called_once_with(v_linear=.1, omega=.2)
    assert not server.is_done and server.active
