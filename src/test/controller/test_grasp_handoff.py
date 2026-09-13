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


def test_grounded_grasp_turns_arm_toward_target_then_reacquires(monkeypatch):
    op = operation()
    target = SimpleNamespace(xyz=np.array([0.09, -0.52, 0.52]))
    op.agent.prepare_query_target.return_value = target
    events = []
    op.robot.head_to.side_effect = lambda *args, **kwargs: events.append("head")
    monkeypatch.setattr("emet.controller.dynamem.look.wait_post_motion_obs", lambda *a, **kw: events.append("fresh"))
    op.agent.prepare_query_target.side_effect = lambda query: (events.append("ground"), target)[1]
    op.align_grounded_target_for_grasp()
    pose = op.robot.move_base_to.call_args.args[0]
    np.testing.assert_allclose(pose[:2], [0, 0])
    assert pose[2] == pytest.approx(np.arctan2(-0.53, 0.08) + np.pi / 2)
    op.agent.prepare_query_target.assert_called_once_with("red cylinder")
    assert op.grounded_target is target
    np.testing.assert_allclose(op.get_object_xyz(), target.xyz)
    assert events == ["head", "fresh", "ground"]


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


@pytest.mark.parametrize("world_shape", [(8, 8, 3), (4, 4, 3)])
def test_servo_validates_wrist_mask_without_head_semantics(world_shape):
    op = operation()
    op.intro = Mock()
    op.warn = Mock()
    op.gripper_aruco_detector = Mock()
    op.pregrasp_open_loop = Mock(return_value=True)
    op.grounded_target = SimpleNamespace(geometry_source="vlm_selected_depth_surface")
    op.track_image_center = True
    op.open_loop = False
    mask = np.ones((8, 8), dtype=bool)
    op.get_target_mask = Mock(return_value=mask)
    op._compute_center_depth = Mock(return_value=0.3)
    op.observations = Mock()
    op.observations.get_latest_centroid.return_value = np.array([4, 4])
    op.robot.get_joint_positions.return_value = np.zeros(11)
    servo = SimpleNamespace(
        ee_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        semantic=None,
        get_ee_xyz_in_world_frame=lambda: np.ones(world_shape),
    )
    op.robot.get_servo_observation.return_value = servo
    op.show_point_cloud = True
    # Stop at the first consumer of the validated 3D point, before any motion.
    op._debug_show_point_cloud = Mock(side_effect=RuntimeError("validated wrist point"))
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        if world_shape[:2] == mask.shape:
            with pytest.raises(RuntimeError, match="validated wrist point"):
                op.visual_servo_to_object(None)
            np.testing.assert_array_equal(op._debug_show_point_cloud.call_args.args[1], np.ones(3))
        else:
            with pytest.raises(ValueError, match="target mask shape"):
                op.visual_servo_to_object(None)
            op._debug_show_point_cloud.assert_not_called()
    op.agent.semantic_sensor.predict.assert_not_called()
    op.robot.arm_to.assert_not_called()


@pytest.mark.parametrize("arrived", [True, False])
def test_pregrasp_propagates_arm_motion_result(arrived):
    op = operation()
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.zeros(3), np.array([0, 0, 0, 1]))
    op.robot_model.manip_ik_for_grasp_frame.return_value = (np.zeros(11), None, None, True, None)
    op.robot.arm_to.return_value = arrived
    assert op.pregrasp_open_loop(op.get_object_xyz()) is arrived


def test_wrist_tracking_failure_saves_calibrated_evidence_without_accepting(tmp_path, monkeypatch):
    import json

    from emet.memory.grounded_target import GroundedTarget

    monkeypatch.setenv("EMET_EQA_EPISODE_DIR", str(tmp_path))
    op = operation()
    op.grounded_target = GroundedTarget(1, 2, 3, np.ones((16, 3)), "vlm_selected_depth_surface")
    servo = SimpleNamespace(
        ee_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        ee_depth=np.ones((8, 8)),
        ee_camera_K=np.eye(3),
        ee_camera_pose=np.eye(4),
        get_ee_xyz_in_world_frame=lambda: np.zeros((8, 8, 3)),
    )
    op.agent.ground_vlm_frame.return_value = (SimpleNamespace(instance=np.full((8, 8), -1)), [], [], {"valid": False})
    with pytest.raises(ValueError, match="absent or ambiguous"):
        op.get_target_mask(servo, center=(4, 4))
    records = list((tmp_path / "wrist_tracking").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["verification"]["valid"] is False
    assert record["metadata"]["camera_pose"] == np.eye(4).tolist()
    assert (records[0].parent / record["rgb_file"]).is_file()
