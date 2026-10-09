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
    controller.control.v_max = .2
    controller.cfg.max_rev_dist = 1.
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
    controller.control.v_max = .005
    controller.compute_current_error.return_value = np.array([-.009, 0., .7])
    controller.cfg.max_rev_dist = 0.
    MujocoZmqServer._control_loop_callback(server)
    assert server.robot_sim.set_base_velocity.call_args.kwargs["v_linear"] == 0.
    controller.cfg.max_rev_dist = 1.
    MujocoZmqServer._control_loop_callback(server)
    assert server.robot_sim.set_base_velocity.call_args.kwargs["v_linear"] == -.005


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


@pytest.mark.parametrize("turn_sign", [-1, 1])
@pytest.mark.parametrize("creep", [-.004, .004])
def test_precision_turn_corrects_creep_without_relaxing_arrival(monkeypatch, turn_sign, creep):
    """A small wheel-induced drift must not trigger repeated turn/translate cycles."""
    from emet.motion.control.goto_controller import GotoVelocityController

    pose = np.array([-.012, .006 * turn_sign, -.62 * turn_sign])
    goal = np.array([0., 0., 2.368 * turn_sign])
    elapsed = [0.]
    commanded = [0., 0.]
    monkeypatch.setattr("emet.simulation.mujoco_server_stretch.timeit.default_timer", lambda: elapsed[0])
    controller = GotoVelocityController()
    controller.update_pose_feedback(pose.copy())
    controller.update_goal(goal)
    controller.control.set_angular_error_tolerance(.015)

    def velocity(v_linear, omega):
        commanded[:] = [v_linear, omega]

    server = SimpleNamespace(
        _status=Status(time=0., base=SimpleNamespace(x_vel=0., theta_vel=0.)),
        debug_control_loop=False, get_base_pose=lambda: pose.copy(), controller=controller,
        active=True, xyt_goal=goal.copy(), goal_set_t=0., goal_set_sim_t=0.,
        controller_finished=False, done_since=0., done_t=0., robot_sim=SimpleNamespace(set_base_velocity=velocity),
        _precision_xy_tolerances=(.01, .02), _precision_xy_acquired=True,
    )
    errors = []
    for _ in range(1500):
        MujocoZmqServer._control_loop_callback(server)
        linear, angular = commanded
        assert abs(linear) <= min(.02, controller.control.v_max)
        # Fixed 4 mm/s creep whenever the wheel pair is turning.
        actual_linear = linear + (creep if abs(angular) > 0 else 0)
        dt = .02
        pose[:2] += dt * actual_linear * np.array([np.cos(pose[2]), np.sin(pose[2])])
        pose[2] += dt * angular
        elapsed[0] += dt
        server._status.time = elapsed[0]
        server._status.base.x_vel = actual_linear
        server._status.base.theta_vel = angular
        errors.append(np.linalg.norm(pose[:2] - goal[:2]))
        if not server.active:
            break
    assert not server.active, "Precision turn never converged under bounded creep"
    assert max(errors) <= .02  # No expanded position acceptance or excursion.
    assert abs(np.arctan2(np.sin(pose[2] - goal[2]), np.cos(pose[2] - goal[2]))) <= .03
