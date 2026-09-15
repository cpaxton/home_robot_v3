# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from emet.controller.operations.place_object import PlaceObjectOperation
from emet.memory.grounded_target import GroundedTarget
from emet.motion import HelloStretchIdx


def grounded(points):
    return GroundedTarget(1, 1, 1, np.repeat(np.atleast_2d(points), 10, axis=0))


def operation():
    op = object.__new__(PlaceObjectOperation)
    op._name = "place_test"
    op.agent = Mock()
    op.agent.manipulation_radius = 0.55
    op.agent.voxel_size = 0.05
    op.parameters = {}
    op.robot = op.agent.robot
    op.robot_model = op.robot.get_robot_model.return_value
    op.robot_model.manip_fk.return_value = (np.array([0, -0.2, 0.8]), np.array([0, 0, 0, 1]))
    op.robot.get_observation.return_value = SimpleNamespace(joint=np.zeros(11))
    op.robot.get_base_pose.return_value = np.zeros(3)
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    op.robot.arm_to.return_value = True
    op.robot.open_gripper.return_value = True
    op.sample_placement_position = Mock(return_value=np.array([0, -0.5, 0.6]))
    op.get_target = Mock(return_value=SimpleNamespace(point_cloud=torch.tensor([[0, -0.5, 0.6]])))
    op._get_place_joint_state = Mock(return_value=(np.zeros(11), True))
    op.talk = False
    return op


def test_near_support_release_preset_preserves_recovery_control():
    from emet.config.loader import load_config

    control = load_config("configs/emet/query_geometry_recovery_pilot.yaml").mapping_dict
    candidate = load_config("configs/emet/query_geometry_setdown_pilot.yaml").mapping_dict
    from emet.controller.dynamem.look import _find_phase_nav_timeout

    assert _find_phase_nav_timeout(SimpleNamespace(parameters=candidate)) == 30.0
    assert candidate.pop("find_phase_nav_step_timeout_s") == 30.0
    assert candidate["grasp"].pop("geometry_servo_tolerance_m") == 0.005
    assert candidate.pop("place") == {"release_clearance_m": 0.005, "release_z_tolerance_m": 0.005}
    assert candidate == control


@pytest.mark.parametrize(
    "key,value",
    [
        ("release_clearance_m", -0.01),
        ("release_clearance_m", float("nan")),
        ("release_clearance_m", 0.1),
        ("release_z_tolerance_m", 0),
        ("release_z_tolerance_m", float("inf")),
        ("release_z_tolerance_m", 0.02),
    ],
)
def test_release_configuration_cannot_request_penetration_or_relax_height_gate(key, value):
    op = operation()
    op.parameters = {"place": {key: value}}
    with pytest.raises(ValueError):
        op.configure(held_query="red cylinder")
    op.robot.arm_to.assert_not_called()


def test_near_support_release_requires_another_observed_correction():
    op = operation()
    op.parameters = {"place": {"release_clearance_m": 0.005, "release_z_tolerance_m": 0.005}}
    op.configure(held_query="red cylinder")
    op.robot.get_joint_positions.return_value = np.zeros(11)
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, 0.65]
    # The old 2 cm gap / 1.5 cm tolerance accepts this 1.7 cm gap. The
    # near-support row must lower by 1.2 cm, then inspect another fresh frame.
    high = np.array([[0, -0.5, 0.617]] * 20)
    low = np.array([[0, -0.5, 0.605]] * 20)
    with patch(
        "emet.controller.operations.query_observation.observe_query_points",
        side_effect=[(SimpleNamespace(ee_pose=pose), high), (SimpleNamespace(ee_pose=pose), low)],
    ) as observe:
        assert op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    assert observe.call_count == 2
    assert op.robot.arm_to.call_count == 1
    assert op._get_place_joint_state.call_args.args[0][2] == pytest.approx(0.8 - 0.012)


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


def test_place_ik_transforms_world_target_with_world_base_pose():
    op = operation()
    op.robot.get_base_pose_world.return_value = np.array([4, -2, np.pi / 2])
    op.robot.get_base_pose.return_value = np.zeros(3)
    op.sample_placement_position.return_value = np.array([4.5, -2, 0.6])
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    np.testing.assert_allclose(op._get_place_joint_state.call_args.kwargs["pos"][:2], [0, -0.5], atol=1e-10)
    op.robot.get_base_pose.assert_not_called()


def test_support_depth_outlier_does_not_raise_the_placement_approach():
    op = operation()
    points = torch.tensor([[0, -0.5, 0.56]] * 100 + [[0, -0.3, 0.9]])
    op.get_target.return_value.point_cloud = points
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert op._get_place_joint_state.call_args.kwargs["pos"][2] == pytest.approx(0.66)


def test_sampling_placement_does_not_edit_the_receptacle_cloud():
    op = operation()
    points = torch.tensor([[0.0, -0.4, 0.55], [0, -0.6, 0.55]])
    op.get_target.return_value.point_cloud = points
    before = points.clone()
    PlaceObjectOperation.sample_placement_position(op, np.zeros(3))
    assert torch.equal(points, before)


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
    old = grounded([0.2, -0.5, 0.6])
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    fresh = grounded([0.21, -0.51, 0.6])
    op.agent.prepare_query_target.side_effect = [old, fresh]
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target") as orient:
        assert op.prepare_query_target("blue cube") is fresh
    orient.assert_called_once()
    assert orient.call_args.args[0] is op.robot
    np.testing.assert_array_equal(orient.call_args.args[1], old.xyz)
    assert op.agent.prepare_query_target.call_count == 2
    op.agent.navigate_to_target_pose.assert_not_called()


@pytest.mark.parametrize("arrived", [True, False])
def test_far_receptacle_requires_safe_navigation_before_arm_alignment(arrived):
    op = operation()
    target = grounded([1, 0, 0.9])
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    fresh = grounded([1.01, 0, 0.9])
    op.agent.prepare_query_target.side_effect = [target, fresh]
    op.agent.navigate_to_target_pose.return_value = arrived
    op.robot.get_base_pose_world.side_effect = [np.zeros(3), np.array([0.3, 0, 0])]
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target") as orient:
        if arrived:
            assert op.prepare_query_target("sink") is fresh
            orient.assert_called_once()
            assert orient.call_args.args[0] is op.robot
            np.testing.assert_array_equal(orient.call_args.args[1], target.xyz)
        else:
            with pytest.raises(RuntimeError, match="placement workspace"):
                op.prepare_query_target("sink")
            orient.assert_not_called()
            assert op.agent.prepare_query_target.call_count == 1
    call = op.agent.navigate_to_target_pose.call_args
    assert call.kwargs["distance_range"] == pytest.approx((0.55, 0.8))
    np.testing.assert_array_equal(call.args[0], target.xyz)
    op.robot.move_base_to.assert_not_called()
    op.robot.open_gripper.assert_not_called()


def test_placement_workspace_rejects_motion_success_without_measured_arrival():
    op = operation()
    op.agent.prepare_query_target.return_value = grounded([1, 0, 0.9])
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    op.agent.navigate_to_target_pose.return_value = True
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target") as orient:
        with pytest.raises(RuntimeError, match="still too far"):
            op.prepare_query_target("sink")
        orient.assert_not_called()
    op.robot.open_gripper.assert_not_called()


def test_large_receptacle_approach_targets_placement_surface_not_far_center():
    op = operation()
    points = np.array([[1.0, 0, 0.9], [2.0, 0, 0.9]])
    target = grounded(points)
    op.agent.prepare_query_target.return_value = target
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    op.robot.get_base_pose_world.side_effect = [np.zeros(3), np.array([0.65, 0, 0])]
    op.agent.navigate_to_target_pose.return_value = True
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target"):
        assert op.prepare_query_target("counter") is target
    np.testing.assert_allclose(op.agent.navigate_to_target_pose.call_args.args[0], [1.25, 0, 0.9])
    np.testing.assert_array_equal(points, [[1.0, 0, 0.9], [2.0, 0, 0.9]])


def test_partial_reacquisition_preserves_the_planned_point_and_support_clearance():
    op = operation()
    target = grounded([[1, 0, 0.9], [2, 0, 0.9], [1, 0, 0.7], [2, 0, 0.7]])
    fresh = grounded([[1.6, 0, 0.7], [1.8, 0, 0.7]])
    op.agent.prepare_query_target.side_effect = [target, fresh]
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    op.robot.get_base_pose_world.side_effect = [np.zeros(3), np.array([0.65, 0, 0])]
    op.agent.navigate_to_target_pose.return_value = True
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target"):
        assert op.prepare_query_target("counter") is fresh
    op.get_target.return_value.point_cloud = torch.as_tensor(fresh.points.copy())
    planned = op.agent.navigate_to_target_pose.call_args.args[0]
    sampled = op.sample_placement_position(np.array([0.65, 0, 0]))
    np.testing.assert_array_equal(sampled, planned)
    assert op.support_top() == pytest.approx(0.9)
    sampled[0] += 10
    np.testing.assert_array_equal(op.sample_placement_position(np.zeros(3)), planned)


def test_reacquisition_of_a_different_support_does_not_reuse_old_placement():
    op = operation()
    op.sample_placement_position = PlaceObjectOperation.sample_placement_position.__get__(op)
    op.agent.prepare_query_target.side_effect = [grounded([0.5, 0, 0.7]), grounded([2, 0, 0.7])]
    with patch("emet.controller.operations.stretch_manipulation.orient_arm_toward_target"):
        with pytest.raises(ValueError, match="absent or ambiguous"):
            op.prepare_query_target("counter")
    assert op._placement_reference is None
    assert op._support_height_reference is None
    op.robot.open_gripper.assert_not_called()


@pytest.mark.parametrize("reset", ["new_query", "nonquery_configuration"])
def test_new_request_cannot_reuse_a_previous_placement_reference(reset):
    op = operation()
    op._placement_reference = np.ones(3)
    op._support_height_reference = 1.0
    if reset == "new_query":
        op.agent.prepare_query_target.side_effect = ValueError("abstained")
        with pytest.raises(ValueError, match="abstained"):
            op.prepare_query_target("counter")
    else:
        op.configure()
    assert op._placement_reference is None
    assert op._support_height_reference is None


def test_missing_final_object_alignment_never_releases():
    op = operation()
    op.held_query = "red cylinder"
    op.align_held_object_for_release = Mock(return_value=False)
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert not op.was_successful()
    op.robot.open_gripper.assert_not_called()


def test_query_place_keeps_existing_payload_orientation():
    op = operation()
    op.held_query = "red cylinder"
    op.align_held_object_for_release = Mock(return_value=False)
    op.robot.get_observation.return_value.joint[HelloStretchIdx.WRIST_PITCH] = -0.25
    with patch("emet.controller.operations.place_object.time.sleep"):
        op.run()
    assert op.robot.arm_to.call_args_list[0].args[0][HelloStretchIdx.WRIST_PITCH] == -0.25


def test_visual_placement_centers_before_lowering_to_release_height():
    op = operation()
    op.held_query = "red cylinder"
    op.robot.get_joint_positions.return_value = np.zeros(11)
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, 0.68]
    obs = SimpleNamespace(ee_pose=pose)
    offset = np.array([[0, -0.53, 0.67], [0.02, -0.51, 0.71]])
    centered = offset + [-0.01, 0.02, 0]
    aligned = np.array([[-0.01, -0.51, 0.62], [0.01, -0.49, 0.66]])
    op.robot_model.manip_fk.side_effect = [
        (np.array([0, -0.2, 0.8]), np.array([0, 0, 0, 1])),
        (np.array([-0.01, -0.18, 0.8]), np.array([0, 0, 0, 1])),
    ]
    with patch(
        "emet.controller.operations.query_observation.observe_query_points",
        side_effect=[(obs, offset), (obs, centered), (obs, aligned)],
    ):
        assert op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    assert op.robot.arm_to.call_count == 2
    first, second = [call.args[0] for call in op._get_place_joint_state.call_args_list]
    np.testing.assert_allclose(first - [0, -0.2, 0.8], [-0.01, 0.02, 0])
    np.testing.assert_allclose(second - first, [0, 0, -0.05])


@pytest.mark.parametrize("bottom", [0.600, 0.605, 0.609])
def test_lateral_alignment_near_support_does_not_drag_or_release(bottom):
    op = operation()
    op.held_query = "red cylinder"
    op.release_clearance_m = op.release_z_tolerance_m = 0.005
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, bottom + 0.03]
    points = np.array([[0.03, -0.5, bottom], [0.03, -0.5, bottom], [0.03, -0.5, bottom + 0.04]])
    with patch(
        "emet.controller.operations.query_observation.observe_query_points",
        return_value=(SimpleNamespace(ee_pose=pose), points),
    ):
        assert not op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    op.robot.arm_to.assert_not_called()
    op.robot.open_gripper.assert_not_called()


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


def test_visual_placement_compensates_tracking_bias_without_accumulating_wrist_sag():
    op = operation()
    op.held_query = "red cylinder"
    measured = np.zeros(11)
    measured[:3] = [0, -0.5, 0.70]
    op.robot.get_joint_positions.side_effect = lambda: measured.copy()
    op.robot_model.manip_fk.side_effect = lambda q: (q[:3].copy(), np.array([q[3], 0, 0, 1]))

    def ik(pos, quat, seed):
        q = seed.copy()
        q[:3] = pos
        q[3] = quat[0]
        return q, True

    def move(q, **kwargs):
        measured[:] = q
        measured[2] -= 0.016  # Steady loaded tracking error, not image noise.
        measured[3] -= 0.02  # Must not become the next orientation target.
        return True

    def observe(*args, **kwargs):
        pose = np.eye(4)
        pose[:3, 3] = measured[:3]
        points = measured[:3] + np.array([[0, 0, -0.04], [0, 0, -0.04], [0, 0, 0]])
        return SimpleNamespace(ee_pose=pose), points

    op._get_place_joint_state.side_effect = ik
    op.robot.arm_to.side_effect = move
    with patch("emet.controller.operations.query_observation.observe_query_points", side_effect=observe):
        assert op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    assert op.robot.arm_to.call_count == 2
    commands = op._get_place_joint_state.call_args_list
    assert commands[1].args[0][2] > commands[0].args[0][2]
    for command in commands:
        np.testing.assert_array_equal(command.args[1], [0, 0, 0, 1])


def test_visual_placement_bounds_unexecuted_reference_corrections():
    op = operation()
    op.held_query = "red cylinder"
    op.robot.get_joint_positions.return_value = np.zeros(11)
    pose = np.eye(4)
    pose[:3, 3] = [0, -0.5, 0.70]
    points = np.array([[0, -0.5, 0.66], [0, -0.5, 0.66], [0, -0.5, 0.7]])
    # A motion API returning success without movement cannot wind up a large
    # target: two repeated 4 cm corrections exceed the 5 cm measured budget.
    with patch(
        "emet.controller.operations.query_observation.observe_query_points",
        return_value=(SimpleNamespace(ee_pose=pose), points),
    ):
        assert not op.align_held_object_for_release(np.array([0, -0.5, 0.6]))
    assert op.robot.arm_to.call_count == 1
