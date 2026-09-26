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


def test_precision_position_hysteresis_preserves_final_turn(monkeypatch):
    monkeypatch.setattr("emet.simulation.mujoco_server_stretch.timeit.default_timer", lambda: 1.)
    controller = Mock()
    controller.compute_control.return_value = (0., .2)
    controller.is_done.return_value = False
    controller.timeout.return_value = False
    server = SimpleNamespace(
        _status=Status(time=1., base=SimpleNamespace(x_vel=0., theta_vel=.2)),
        debug_control_loop=False, get_base_pose=lambda: np.zeros(3), controller=controller,
        active=True, xyt_goal=np.ones(3), goal_set_t=0., goal_set_sim_t=0.,
        controller_finished=False, robot_sim=Mock(),
        _precision_xy_tolerances=(.01, .02), _precision_xy_acquired=False,
    )
    for distance, tolerance in [(.015, .01), (.009, .02), (.015, .02), (.021, .01)]:
        controller.compute_current_error.return_value = np.array([distance, 0., .7])
        MujocoZmqServer._control_loop_callback(server)
        controller.control.set_linear_error_tolerance.assert_called_with(tolerance)


def test_accepted_navigation_transitions_mode_inside_adapter():
    server = SimpleNamespace(controller=Mock())
    received = []

    def handle(action):
        received.append(action)
        server._contract_navigation_context = {"resolved_goal": action["xyt"]}

    server.handle_action = handle
    action = {"xyt": [1., 0., .5], "nav_policy": "precision"}
    context = MujocoZmqServer.start_navigation_command(server, action)
    assert context["resolved_goal"] == action["xyt"]
    assert received[0]["control_mode"] == "navigation"
    assert "control_mode" not in action
    server.controller.control.set_linear_error_tolerance.assert_called_once_with(.01)
    server.controller.control.set_angular_error_tolerance.assert_called_once_with(.015)
