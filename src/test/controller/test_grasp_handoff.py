# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.controller.operations.grasp_object import GraspObjectOperation
from emet.motion import HelloStretchIdx


def operation():
    op = object.__new__(GraspObjectOperation)
    op.agent = Mock()
    op.robot = op.agent.robot
    op.robot_model = Mock()
    op._name = "test"
    op._object_xyz = np.array([0.08, -0.53, 0.52])
    op.target_object = "red cylinder"
    op.agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, -1.4])
    op.agent.robot.move_base_to.return_value = True
    return op


def test_grounded_grasp_turns_arm_toward_target_then_reacquires():
    op = operation()
    target = SimpleNamespace(xyz=np.array([0.09, -0.52, 0.52]))
    op.agent.prepare_query_target.return_value = target
    op.align_grounded_target_for_grasp()
    pose = op.robot.move_base_to.call_args.args[0]
    np.testing.assert_allclose(pose[:2], [0, 0])
    assert pose[2] == pytest.approx(np.arctan2(-0.53, 0.08) + np.pi / 2)
    op.agent.prepare_query_target.assert_called_once_with("red cylinder")
    assert op.grounded_target is target
    np.testing.assert_allclose(op.get_object_xyz(), target.xyz)


def test_failed_alignment_does_not_reacquire_or_move_arm():
    op = operation()
    op.robot.move_base_to.return_value = False
    with pytest.raises(RuntimeError, match="orientation"):
        op.align_grounded_target_for_grasp()
    op.agent.prepare_query_target.assert_not_called()
    op.robot.arm_to.assert_not_called()


@pytest.mark.parametrize("ik", [None, "negative_arm", "nan"])
def test_invalid_pregrasp_never_sends_arm_command(ik):
    op = operation()
    op.error = Mock()
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.zeros(3), np.array([0, 0, 0, 1]))
    q = np.zeros(11)
    q[HelloStretchIdx.ARM] = -0.18 if ik == "negative_arm" else np.nan
    op.robot_model.manip_ik_for_grasp_frame.return_value = (None if ik is None else q, None, None, ik is not None, None)
    assert op.pregrasp_open_loop(op.get_object_xyz()) is False
    op.robot.arm_to.assert_not_called()


def test_servo_stops_after_failed_pregrasp():
    op = operation()
    op.intro = Mock()
    op.show_servo_gui = False
    op.gripper_aruco_detector = Mock()
    op.pregrasp_open_loop = Mock(return_value=False)
    with patch("emet.controller.operations.grasp_object.time.sleep") as sleep:
        assert op.visual_servo_to_object(None) is False
    sleep.assert_not_called()
    op.robot.get_servo_observation.assert_not_called()
