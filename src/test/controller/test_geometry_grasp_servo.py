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


def test_offset_geometry_moves_at_most_five_cm_and_does_not_close():
    op, servo, mask = fixture([0.1, 0, 0.02])
    assert op.geometry_servo_step(servo, mask) is None
    target = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    assert np.linalg.norm(target) == pytest.approx(0.05)
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
