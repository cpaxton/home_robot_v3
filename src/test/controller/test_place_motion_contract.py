# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from emet.controller.operations.place_object import PlaceObjectOperation
from emet.motion import HelloStretchIdx


def operation():
    op = object.__new__(PlaceObjectOperation)
    op._name = "place_test"
    op.agent = Mock()
    op.robot = op.agent.robot
    op.robot_model = op.robot.get_robot_model.return_value
    op.robot_model.manip_fk.return_value = (np.array([0, -0.2, 0.8]), np.array([0, 0, 0, 1]))
    op.robot.get_observation.return_value = SimpleNamespace(joint=np.zeros(11))
    op.robot.get_base_pose.return_value = np.zeros(3)
    op.robot.arm_to.return_value = True
    op.robot.open_gripper.return_value = True
    op.sample_placement_position = Mock(return_value=np.array([0, -0.5, 0.6]))
    op.get_target = Mock(return_value=SimpleNamespace(point_cloud=torch.tensor([[0, -0.5, 0.6]])))
    op._get_place_joint_state = Mock(return_value=(np.zeros(11), True))
    op.talk = False
    return op


@pytest.mark.parametrize("motions", [[False], [True, False]])
def test_failed_place_motion_never_releases(motions):
    op = operation()
    op.robot.arm_to.side_effect = motions
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    op.robot.open_gripper.assert_not_called()
    op.robot.move_to_nav_posture.assert_not_called()


def test_failed_release_does_not_retreat_or_report_success():
    op = operation()
    op.robot.open_gripper.return_value = False
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    assert op.robot.arm_to.call_count == 2
    op.robot.move_to_nav_posture.assert_not_called()


def test_failed_retreat_does_not_report_success_or_start_posture_motion():
    op = operation()
    op.robot.arm_to.side_effect = [True, True, False]
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    assert op.released is True
    op.robot.move_to_nav_posture.assert_not_called()


@pytest.mark.parametrize("q,success", [(None, True), (np.full(11, np.nan), True), (np.zeros(11), False)])
def test_invalid_place_ik_never_releases(q, success):
    op = operation()
    op._get_place_joint_state.return_value = (q, success)
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    op.robot.open_gripper.assert_not_called()
    assert op.robot.arm_to.call_count == 1


def test_place_does_not_mutate_cached_observation():
    op = operation()
    joint = op.robot.get_observation.return_value.joint
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert op.was_successful()
    assert joint[HelloStretchIdx.WRIST_PITCH] == 0
    assert all(call.kwargs.get("blocking") is True for call in op.robot.arm_to.call_args_list)


def test_place_reads_joint_state_after_posture_change():
    op = operation()
    op.robot.move_to_manip_posture.side_effect = lambda: setattr(
        op.robot.get_observation.return_value, "joint", np.ones(11)
    )
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert op.robot.arm_to.call_args_list[0].args[0][HelloStretchIdx.LIFT] == 1


def test_reach_contract_is_shared_by_lazy_and_instance_controllers():
    from emet.controller.base_controller import BaseController
    from emet.controller.controller_instance_memory import RobotAgent
    from emet.controller.controller_lazy_graph import LazyGraphController

    assert RobotAgent.manipulation_radius is BaseController.manipulation_radius
    assert LazyGraphController.manipulation_radius is BaseController.manipulation_radius
    agent = object.__new__(RobotAgent)
    agent._manipulation_radius = 0.8
    assert agent.manipulation_radius == 0.8


def test_place_reacquires_after_arm_facing_rotation():
    op = operation()
    old = SimpleNamespace(xyz=np.array([0.2, -0.5, 0.6]))
    fresh = SimpleNamespace(xyz=np.array([0.21, -0.51, 0.6]))
    op.agent.prepare_query_target.side_effect = [old, fresh]
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target") as orient:
        assert op.prepare_query_target("blue cube") is fresh
    orient.assert_called_once_with(op.robot, old.xyz)
    assert op.agent.prepare_query_target.call_count == 2


def test_missing_final_object_alignment_never_releases():
    op = operation()
    op.held_query = "red cylinder"
    op.align_held_object_for_release = Mock(return_value=False)
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    op.robot.open_gripper.assert_not_called()


def test_visual_placement_corrects_observed_object_offset_and_drop_height():
    op = operation()
    op.held_query = "red cylinder"
    op.robot.get_joint_positions.return_value = np.zeros(11)
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, 0.68]
    obs = SimpleNamespace(ee_pose=pose)
    offset = np.array([[0, -0.53, 0.67], [0.02, -0.51, 0.71]])
    aligned = np.array([[-0.01, -0.51, 0.62], [0.01, -0.49, 0.66]])
    with patch(
        "emet.controller.operations.query_observation.observe_query_points", side_effect=[(obs, offset), (obs, aligned)]
    ):
        assert op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    assert op.robot.arm_to.call_count == 1
    requested = op._get_place_joint_state.call_args.args[0]
    delta = requested - np.array([0, -0.2, 0.8])
    assert delta[0] < 0 and delta[1] > 0 and delta[2] < 0
    assert np.linalg.norm(delta) == pytest.approx(0.05)


@pytest.mark.parametrize("failure", ["far_object", "far_goal", "motion", "invalid_ik"])
def test_visual_placement_rejects_untrusted_or_failed_correction(failure):
    op = operation()
    op.held_query = "red cylinder"
    op.robot.get_joint_positions.return_value = np.zeros(11)
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, 0.7]
    points = np.array([[0, -0.5, 0.67], [0.02, -0.48, 0.71]])
    goal = np.array([0, -0.5, 0.6])
    if failure == "far_object":
        points += 1
    elif failure == "far_goal":
        goal[:2] += 1
    elif failure == "motion":
        op.robot.arm_to.return_value = False
    else:
        op._get_place_joint_state.return_value = (None, False)
    with patch(
        "emet.controller.operations.query_observation.observe_query_points",
        return_value=(SimpleNamespace(ee_pose=pose), points),
    ):
        assert op.align_held_object_for_release(goal) is False
    if failure != "motion":
        op.robot.arm_to.assert_not_called()
