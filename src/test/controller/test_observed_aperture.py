# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from emet.utils.gripper import measure_observed_aperture


def observation():
    servo = SimpleNamespace(
        ee_rgb=np.zeros((100, 100, 3), dtype=np.uint8),
        ee_depth=np.full((100, 100), 0.15),
        ee_camera_K=np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
        ee_camera_pose=np.eye(4),
    )
    detector = Mock()
    detector.detect_aruco_centers.return_value = (np.array([[[10, 50]], [[90, 50]]]), np.array([200, 201]))
    points = np.array([[x, y, 0.35] for x in [-0.02, 0.02] for y in [-0.01, 0.01]])
    return servo, detector, points


def test_span_uses_calibrated_depth_not_pixel_width_or_command_mapping():
    servo, detector, points = observation()
    span, width = measure_observed_aperture(servo, points, detector)
    assert span == pytest.approx(0.12)
    assert width == pytest.approx(0.04)


def test_width_uses_finger_axis_and_world_camera_transform():
    servo, detector, points = observation()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    servo.ee_camera_pose[:3, :3] = rotation
    servo.ee_camera_pose[:3, 3] = [1, 2, 3]
    points = points @ rotation.T + [1, 2, 3]
    assert measure_observed_aperture(servo, points, detector) == pytest.approx((0.12, 0.04))


@pytest.mark.parametrize("rotation", [np.eye(3), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])])
def test_aperture_clearance_uses_full_geometry_for_a_high_counter(rotation):
    servo, detector, points = observation()
    # Observed room failure: target 59 cm above the grasp in camera coordinates
    # but only 3.7 cm farther in optical depth. It is not inside the closing jaws.
    points[:, 1] -= 0.59
    points[:, 2] = 0.187
    servo.ee_camera_pose[:3, :3] = rotation
    servo.ee_camera_pose[:3, 3] = [1, 2, 3]
    points = points @ rotation.T + [1, 2, 3]
    assert measure_observed_aperture(servo, points, detector) == pytest.approx((0.12, 0.04))


@pytest.mark.parametrize("x", [0.0, 0.06, 0.075])
def test_aperture_rejects_geometry_near_any_part_of_the_closing_segment(x):
    servo, detector, points = observation()
    points[:, 0] += x
    points[:, 2] = 0.17
    with pytest.raises(ValueError, match="too close"):
        measure_observed_aperture(servo, points, detector)


@pytest.mark.parametrize("invalid", ["missing", "wrong_ids", "duplicate", "zero_depth", "nan_depth", "near_target"])
def test_uncertain_finger_or_target_geometry_cannot_authorize_narrowing(invalid):
    servo, detector, points = observation()
    centers, ids = detector.detect_aruco_centers.return_value
    if invalid == "missing":
        detector.detect_aruco_centers.return_value = (centers[:1], ids[:1])
    elif invalid == "wrong_ids":
        detector.detect_aruco_centers.return_value = (centers, np.array([5, 6]))
    elif invalid == "duplicate":
        detector.detect_aruco_centers.return_value = (centers, np.array([200, 200]))
    elif invalid in {"zero_depth", "nan_depth"}:
        servo.ee_depth[:, :] = 0 if invalid == "zero_depth" else np.nan
    else:
        points[:, 2] = 0.17
    with pytest.raises(ValueError):
        measure_observed_aperture(servo, points, detector)


def aperture_operation():
    from emet.controller.operations.grasp_object import GraspObjectOperation
    from emet.motion import HelloStretchIdx

    op = object.__new__(GraspObjectOperation)
    op.info = op.error = Mock()
    op.robot = Mock()
    op.robot_model = SimpleNamespace(GRIPPER_CLOSED=-0.3)
    op.gripper_aruco_detector = Mock()
    op.grounded_target = SimpleNamespace(points=np.ones((10, 3)))
    op.observed_aperture_margin_m = 0.06
    op._aperture_trace = []
    joints = np.zeros(11)
    joints[HelloStretchIdx.GRIPPER] = 0.5
    op.robot.get_joint_positions.return_value = joints
    op.robot.gripper_to.side_effect = lambda target, **kw: joints.__setitem__(HelloStretchIdx.GRIPPER, target)
    op.robot.get_servo_observation.side_effect = lambda: SimpleNamespace()
    return op


def test_aperture_uses_closed_loop_measurements_without_arm_motion():
    op = aperture_operation()
    with patch("emet.utils.gripper.measure_observed_aperture", side_effect=[(0.16, 0.05), (0.13, 0.05), (0.11, 0.05)]):
        assert op.prepare_observed_aperture()
    assert [c.args[0] for c in op.robot.gripper_to.call_args_list] == pytest.approx([0.475, 0.45])
    op.robot.arm_to.assert_not_called()
    assert len(op._aperture_trace) == 3


def test_invalid_aperture_geometry_never_commands_a_closure():
    op = aperture_operation()
    with patch("emet.utils.gripper.measure_observed_aperture", side_effect=ValueError("missing markers")):
        assert op.prepare_observed_aperture() is False
    op.robot.gripper_to.assert_not_called()
    op.robot.arm_to.assert_not_called()


def test_aperture_preset_preserves_geometry_only_control():
    from emet.config.loader import load_config

    control = load_config("configs/emet/query_geometry_manipulation_pilot.yaml").mapping_dict
    candidate = load_config("configs/emet/query_geometry_aperture_pilot.yaml").mapping_dict
    assert control["grasp"] == {"geometry_servo": True}
    assert candidate["grasp"] == {"geometry_servo": True, "observed_aperture_margin_m": 0.06}
    assert candidate["query_memory"] == control["query_memory"]


def test_narrow_tracking_row_changes_only_the_declared_aperture_margin():
    from emet.core.parameters import get_parameters

    control = get_parameters("configs/emet/query_geometry_tracked_pilot.yaml")
    candidate = get_parameters("configs/emet/query_geometry_tracked_narrow_pilot.yaml")
    assert control.data["grasp"]["observed_aperture_margin_m"] == 0.06
    assert candidate.data["grasp"]["observed_aperture_margin_m"] == 0.04
    candidate.data["grasp"]["observed_aperture_margin_m"] = 0.06
    assert candidate.data == control.data


def test_combined_clutter_preset_changes_only_aperture_from_contact_control():
    from emet.config.loader import load_config

    control = load_config("configs/emet/query_geometry_contact_pilot.yaml").mapping_dict
    candidate = load_config("configs/emet/query_geometry_contact_aperture_pilot.yaml").mapping_dict
    assert candidate["grasp"].pop("observed_aperture_margin_m") == 0.06
    assert candidate == control
