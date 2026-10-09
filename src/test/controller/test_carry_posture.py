# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.controller.zmq_client import StretchZmqClient
from emet.motion import HelloStretchIdx


def client():
    robot = object.__new__(StretchZmqClient)
    robot._zmq_manipulation_supported = Mock(return_value=True)
    robot.switch_to_manipulation_mode = Mock()
    robot.switch_to_navigation_mode = Mock()
    robot.arm_to = Mock(return_value=True)
    robot.send_action = Mock()
    q = np.zeros(11)
    q[HelloStretchIdx.LIFT] = 0.8
    q[HelloStretchIdx.ARM] = 0.2
    q[HelloStretchIdx.WRIST_PITCH] = -0.25
    q[HelloStretchIdx.GRIPPER] = -0.1
    robot.get_joint_positions = Mock(return_value=q)
    return robot, q


@pytest.mark.parametrize("navigation", [False, True])
def test_carry_posture_preserves_payload_wrist_and_height(navigation):
    robot, held = client()
    robot.set_carry_configuration(held)
    held[HelloStretchIdx.BASE_X] = 0.08
    method = robot.move_to_nav_posture if navigation else robot.move_to_manip_posture
    method()
    target = robot.arm_to.call_args.args[0]
    assert target[HelloStretchIdx.BASE_X] == 0.08  # fresh base reference, not captured one
    assert target[HelloStretchIdx.LIFT] == 0.8
    assert target[HelloStretchIdx.WRIST_PITCH] == -0.25
    assert target[HelloStretchIdx.ARM] == 0.01
    assert held[HelloStretchIdx.ARM] == 0.2  # no mutation of the observation
    assert "gripper" not in robot.arm_to.call_args.kwargs
    assert robot.switch_to_navigation_mode.called is navigation
    robot.send_action.assert_not_called()  # no empty-hand posture sent to bridge


def test_failed_carry_posture_does_not_start_navigation():
    robot, held = client()
    robot.set_carry_configuration(held)
    robot.arm_to.return_value = False
    with pytest.raises(RuntimeError, match="Carry posture"):
        robot.move_to_nav_posture()
    robot.switch_to_navigation_mode.assert_not_called()
    assert robot._carry_configuration is not None


def test_confirmed_open_clears_carry_constraint():
    robot, held = client()
    robot.set_carry_configuration(held)
    robot._finish = False
    robot._robot_model = Mock(GRIPPER_OPEN=0.6)
    robot.gripper_to = Mock()
    held[HelloStretchIdx.GRIPPER] = 0.6
    assert robot.open_gripper(blocking=True)
    assert robot._carry_configuration is None


def test_unconfirmed_open_keeps_conservative_carry_constraint():
    robot, held = client()
    robot.set_carry_configuration(held)
    robot._finish = False
    robot._robot_model = Mock(GRIPPER_OPEN=0.6)
    robot.gripper_to = Mock()
    with patch("emet.controller.zmq_client.timeit.default_timer", side_effect=[0, 11]):
        assert robot.open_gripper(blocking=True) is False
    assert robot._carry_configuration is not None
