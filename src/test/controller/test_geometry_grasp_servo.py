# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.controller.operations.grasp_object import GraspObjectOperation
from emet.controller.operations.stretch_manipulation import world_delta_to_model_base


def fixture(center):
    op = object.__new__(GraspObjectOperation)
    op._name = "geometry_servo"
    op.robot = Mock()
    op.robot_model = Mock()
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot_model.manip_fk.return_value = (np.zeros(3), np.array([0, 0, 0, 1]))
    op.robot_model.manip_ik_for_grasp_frame.return_value = (np.zeros(11), None, None, True, None)
    op.robot.arm_to.return_value = True
    op._grasp = Mock(return_value=True)
    points = np.tile(np.asarray(center, dtype=float), (10, 10, 1))
    servo = SimpleNamespace(get_ee_xyz_in_world_frame=lambda: points, ee_pose=np.eye(4))
    return op, servo, np.ones((10, 10), dtype=bool)


def test_centered_observed_geometry_authorizes_grasp_without_pixel_depth_constant():
    op, servo, mask = fixture([0, 0, 0.005])
    assert op.geometry_servo_step(servo, mask) is True
    op._grasp.assert_called_once()
    op.robot.arm_to.assert_not_called()


def test_tight_grasp_gate_corrects_mirrored_run_residual_before_closing():
    # Last accepted error in the weak mirrored grasp, before a delayed slip.
    error = [0.0036036, -0.010019, 0.003071]
    control, servo, mask = fixture(error)
    assert control.geometry_servo_step(servo, mask) is True
    candidate, servo, mask = fixture(error)
    candidate.geometry_servo_tolerance_m = 0.005
    assert candidate.geometry_servo_step(servo, mask) is None
    candidate._grasp.assert_not_called()
    # Lateral alignment must now meet the same tighter gate before insertion.
    np.testing.assert_allclose(candidate.robot_model.manip_ik_for_grasp_frame.call_args.args[0], [0, error[1], 0])


@pytest.mark.parametrize("tolerance", [0, -0.005, 0.013, float("nan"), float("inf")])
def test_geometry_servo_tolerance_cannot_disable_or_loosen_the_closure_gate(tolerance):
    op, _, _ = fixture([0, 0, 0])
    op.parameters = {"grasp": {"geometry_servo": True, "geometry_servo_tolerance_m": tolerance}}
    with pytest.raises(ValueError, match="tolerance"):
        op.configure(grounded_target=Mock(), servo_to_grasp=True, try_open_loop=False)
    op._grasp.assert_not_called()


def test_offset_geometry_moves_at_most_five_cm_and_does_not_close():
    op, servo, mask = fixture([0.1, 0, 0.02])
    assert op.geometry_servo_step(servo, mask) is None
    target = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    assert 0 < np.linalg.norm(target) <= 0.05
    op._grasp.assert_not_called()


@pytest.mark.parametrize("rotation", [np.eye(3), Rotation.from_euler("xyz", [0.2, -0.3, 1.1]).as_matrix()])
def test_grasp_aligns_across_opening_before_inserting_fingers(rotation):
    offset = rotation @ np.array([0.18, 0.05, -0.08])
    op, servo, mask = fixture(offset)
    servo.ee_pose[:3, :3] = rotation
    # Equal model/world grasp orientation makes the requested model delta
    # directly comparable to the world-frame measured correction.
    op.robot_model.manip_fk.return_value = (np.zeros(3), Rotation.from_matrix(rotation).as_quat())
    assert op.geometry_servo_step(servo, mask) is None
    command = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    local = rotation.T @ command
    assert local[0] == pytest.approx(0, abs=1e-10)
    assert local[1] > 0
    assert local[2] == pytest.approx(0, abs=1e-10)
    assert np.linalg.norm(command) == pytest.approx(0.05)
    op._grasp.assert_not_called()


def test_aligned_grasp_advances_without_relaxing_closure_distance():
    op, servo, mask = fixture([0.1, 0.003, -0.004])
    assert op.geometry_servo_step(servo, mask) is None
    command = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    assert command[0] > 0
    assert np.linalg.norm(command) == pytest.approx(0.05)
    op._grasp.assert_not_called()


def test_tight_grasp_tolerance_also_gates_lateral_alignment_before_insertion():
    op, servo, mask = fixture([0.1, 0.008, -0.02])
    op.geometry_servo_tolerance_m = 0.005
    assert op.geometry_servo_step(servo, mask) is None
    np.testing.assert_allclose(op.robot_model.manip_ik_for_grasp_frame.call_args.args[0], [0, 0.008, 0])
    op._grasp.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_pose", "invalid_pose", "far", "invalid_ik", "motion"])
def test_geometry_grasp_fails_closed(failure):
    op, servo, mask = fixture([0.1, 0, 0])
    if failure == "missing_pose":
        servo.ee_pose = None
    elif failure == "invalid_pose":
        servo.ee_pose[:] = np.nan
    elif failure == "far":
        servo.ee_pose[0, 3] = 2
    elif failure == "invalid_ik":
        op.robot_model.manip_ik_for_grasp_frame.return_value = (None, None, None, False, None)
    else:
        op.robot.arm_to.return_value = False
    assert op.geometry_servo_step(servo, mask) is False
    op._grasp.assert_not_called()


def test_depth_edge_outlier_does_not_move_object_center():
    op, servo, mask = fixture([0, 0, 0])
    servo.get_ee_xyz_in_world_frame()[0, 0, :] = [0, 0, 10]
    assert op.geometry_servo_step(servo, mask) is True


def test_contact_calibration_is_applied_in_grasp_frame_not_world():
    op, servo, mask = fixture([-0.015, 0, 0])
    servo.ee_pose[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    op.contact_offset_m = np.array([0, 0.015, 0])
    assert op.geometry_servo_step(servo, mask) is True
    op._grasp.assert_called_once()


def test_nominal_link_center_does_not_close_before_contact_point_is_aligned():
    op, servo, mask = fixture([0, 0, 0])
    op.contact_offset_m = np.array([0, 0.015, 0])
    assert op.geometry_servo_step(servo, mask) is None
    np.testing.assert_allclose(op.robot_model.manip_ik_for_grasp_frame.call_args.args[0], [0, -0.015, 0])
    op._grasp.assert_not_called()


@pytest.mark.parametrize("offset", [[0, 0], [0, float("nan"), 0], [0, 0.06, 0]])
def test_invalid_contact_calibration_is_rejected(offset):
    op, _, _ = fixture([0, 0, 0])
    op.parameters = {"grasp": {"geometry_servo": True, "contact_offset_m": offset}}
    with pytest.raises(ValueError, match="contact offset"):
        op.configure(grounded_target=Mock(), servo_to_grasp=True, try_open_loop=False)


def test_world_delta_is_independent_of_episode_odometry_origin():
    ee_world = np.eye(4)
    ee_world[:3, :3] = Rotation.from_euler("z", np.pi / 2).as_matrix()
    ee_base = Rotation.from_euler("x", 0.3).as_quat()
    result = world_delta_to_model_base(np.array([0, 0.1, 0]), ee_world, ee_base)
    np.testing.assert_allclose(result, [0.1, 0, 0], atol=1e-10)


def test_repeated_no_motion_stops_instead_of_spending_the_servo_budget():
    op, servo, mask = fixture([0.1, 0, 0])
    assert [op.geometry_servo_step(servo, mask) for _ in range(4)] == [None, None, None, False]
    assert op.robot.arm_to.call_count == 3
    op._grasp.assert_not_called()


def test_reference_corrects_persistent_tracking_bias_instead_of_reissuing_same_goal():
    op, servo, mask = fixture([0.04, 0, 0])
    op.geometry_servo_tolerance_m = 0.005
    assert op.geometry_servo_step(servo, mask) is None
    first = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    np.testing.assert_allclose(first, [0.04, 0, 0])
    # A 12 mm steady tracking error survives a completed arm command.
    measured = first - [0.012, 0, 0]
    servo.ee_pose[:3, 3] = measured
    op.robot_model.manip_fk.return_value = (measured, np.array([0, 0, 0, 1]))
    assert op.geometry_servo_step(servo, mask) is None
    second = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    np.testing.assert_allclose(second, [0.052, 0, 0])
    servo.ee_pose[:3, 3] = second - [0.012, 0, 0]
    assert op.geometry_servo_step(servo, mask) is True
    op._grasp.assert_called_once()


def test_reference_saturates_within_measured_step_budget_and_preserves_orientation():
    op, servo, mask = fixture([0.2, 0, 0])
    assert op.geometry_servo_step(servo, mask) is None
    initial_rot = op.robot_model.manip_ik_for_grasp_frame.call_args.args[1].copy()
    measured = np.array([0.025, 0, 0])
    servo.ee_pose[:3, 3] = measured
    sagged = Rotation.from_euler("y", 0.03)
    servo.ee_pose[:3, :3] = sagged.as_matrix()
    op.robot_model.manip_fk.return_value = (measured, sagged.as_quat())
    assert op.geometry_servo_step(servo, mask) is None
    goal, rotation = op.robot_model.manip_ik_for_grasp_frame.call_args.args
    assert np.linalg.norm(goal - measured) <= 0.05 + 1e-10
    np.testing.assert_allclose(rotation, initial_rot)
    op._grasp.assert_not_called()


def test_reset_discards_grasp_reference_and_stall_history():
    op, servo, mask = fixture([0.2, 0, 0])
    op.observations = Mock()
    op.geometry_servo_step(servo, mask)
    assert op._geometry_reference_pos is not None
    op.reset()
    assert op._geometry_reference_pos is None and op._geometry_reference_rot is None
    assert op._geometry_previous_pose is None and op._geometry_stalled_steps == 0
