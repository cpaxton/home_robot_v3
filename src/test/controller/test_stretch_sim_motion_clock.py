"""Slow simulation must retain the controller's measured-time motion budget."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

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


@pytest.mark.parametrize("policy,xy,yaw", [("precision", .01, .015), ("manipulation", .005, .0075)])
def test_accepted_navigation_transitions_mode_inside_adapter(policy, xy, yaw):
    server = SimpleNamespace(controller=Mock(), get_base_pose=lambda: np.zeros(3))
    server.controller.compute_current_error.return_value = np.array([1., 0., .5])
    received = []

    def handle(action):
        received.append(action)
        server._contract_navigation_context = {"resolved_goal": action["xyt"]}

    server.handle_action = handle
    action = {"xyt": [1., 0., .5], "nav_policy": policy}
    context = MujocoZmqServer.start_navigation_command(server, action)
    assert context["resolved_goal"] == action["xyt"]
    assert received[0]["control_mode"] == "navigation"
    assert "control_mode" not in action
    server.controller.control.set_linear_error_tolerance.assert_called_once_with(xy)
    server.controller.control.set_angular_error_tolerance.assert_called_once_with(yaw)
    assert server._precision_xy_tolerances == (xy, 2 * xy)


@pytest.mark.parametrize("fresh_x,acquired", [(0., True), (-.1, False)])
def test_final_turn_uses_fresh_position_acceptance_without_cached_success(fresh_x, acquired):
    from emet.motion.control.goto_controller import GotoVelocityController

    controller = GotoVelocityController()
    controller.update_pose_feedback(np.zeros(3))  # Deliberately stale in the second case.
    server = SimpleNamespace(controller=controller, get_base_pose=lambda: np.array([fresh_x, 0., 0.]))

    def handle(action):
        controller.update_goal(np.asarray(action["xyt"]))
        server._contract_navigation_context = {"resolved_goal": action["xyt"]}

    server.handle_action = handle
    MujocoZmqServer.start_navigation_command(server, {"xyt": [.012, 0., 2.6], "nav_policy": "precision"})
    assert server._precision_xy_acquired is acquired
    linear, angular = controller.compute_control()
    if acquired:
        assert linear == 0 and angular > 0  # Turn now, inside the unchanged 20 mm bound.
    else:
        assert linear > 0  # Fresh feedback outside the bound still requires translation.
    assert not controller.is_done()
