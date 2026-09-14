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
    op.agent.robot.get_base_pose_world.return_value = np.array([0.0, 0.0, -1.4])
    op.agent.robot.move_base_to.return_value = True
    return op


@pytest.mark.parametrize("target_z", [0.45, 1.2])
def test_grasp_orientation_preserves_signed_target_height(target_z):
    op = operation()
    op.intro = op.reset = Mock()
    op.show_object_to_grasp = op.servo_to_grasp = False
    op.reset_observation = op.delete_object_after_grasp = op.talk = False
    op._object_xyz = np.array([2.0, -1.7, target_z])
    op.robot.get_base_pose_world.return_value = np.array([2.0, -1.0, 0.0])
    joints = np.zeros(11)
    op.robot.get_joint_positions.return_value = joints
    op.robot.get_observation.return_value = SimpleNamespace(joint=joints)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (
        np.array([0.0, -0.3, 1.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    op.run()
    pitch = op.robot.arm_to.call_args.args[0][HelloStretchIdx.WRIST_PITCH]
    # Aim from the pitch pivot, not an offset along the grasp frame's Z axis.
    expected = op.offset_from_vertical + np.arctan2(0.4, 1.0 - target_z)
    assert pitch == pytest.approx(expected)
    assert (pitch > 0) == (target_z > 1.0)
    assert op.robot.get_robot_model.return_value.manip_fk.call_args.kwargs["node"] == "link_wrist_pitch"


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
    assert op.robot.move_base_to.call_args.kwargs["navigation_policy"] == "precision"
    assert op.robot.move_base_to.call_args.kwargs["world_frame"] is True
    np.testing.assert_allclose(pose[:2], [0, 0])
    assert pose[2] == pytest.approx(np.arctan2(-0.53, 0.08) + np.pi / 2)
    op.agent.prepare_query_target.assert_called_once_with("red cylinder")
    assert op.grounded_target is target
    np.testing.assert_allclose(op.get_object_xyz(), target.xyz)
    assert events == ["head", "fresh", "ground"]


@pytest.mark.parametrize("distance", [0.58, 0.75])
def test_grounded_grasp_plans_only_when_view_is_inside_pregrasp_workspace(distance):
    op = operation()
    op.aim_grasp_joints = Mock(return_value=np.zeros(11))
    op.solve_pregrasp = Mock(return_value=None)
    op._object_xyz = np.array([4, -2, 0.96])
    start = np.array([4, -2 - distance, 0])
    op.robot.get_base_pose_world.side_effect = [start, np.array([4, -2.75, 0])]
    joints = np.ones(11) * 0.1
    op.robot.get_joint_positions.return_value = joints
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.array([0, -0.41, 0.9]), None)
    op.agent.manipulation_radius = 0.55
    op.agent.navigate_to_target_pose.return_value = True
    op.ensure_grounded_grasp_workspace()
    assert op.agent.navigate_to_target_pose.call_count == (1 if distance == 0.58 else 0)
    if distance == 0.58:
        call = op.agent.navigate_to_target_pose.call_args
        assert call.kwargs["distance_range"] == pytest.approx((0.71, 0.85))
        np.testing.assert_array_equal(call.args[0], op._object_xyz)
        np.testing.assert_array_equal(call.args[1], start)
    np.testing.assert_array_equal(joints, np.ones(11) * 0.1)
    op.robot.arm_to.assert_not_called()
    op.robot.move_base_to.assert_not_called()


@pytest.mark.parametrize("failure", ["plan", "no_progress", "no_range", "invalid_geometry"])
def test_grasp_workspace_does_not_bypass_failed_navigation_or_bad_geometry(failure):
    op = operation()
    op.aim_grasp_joints = Mock(return_value=np.zeros(11))
    op.solve_pregrasp = Mock(return_value=None)
    op._object_xyz = np.array([0, -0.58, 0.96])
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.array([0, -0.41, 0.9]), None)
    op.agent.manipulation_radius = 0.55 if failure != "no_range" else 0.3
    op.agent.navigate_to_target_pose.return_value = failure != "plan"
    if failure == "invalid_geometry":
        op._object_xyz[0] = np.nan
    with pytest.raises((ValueError, RuntimeError), match="geometry|workspace"):
        op.ensure_grounded_grasp_workspace()
    op.robot.arm_to.assert_not_called()
    op.robot.move_base_to.assert_not_called()


def test_close_high_target_becomes_pregrasp_reachable_after_workspace_relocation():
    from emet.motion.kinematics import HelloStretchKinematics

    op = operation()
    model = HelloStretchKinematics()
    op.robot_model = model
    op.robot.get_robot_model.return_value = model
    op._object_xyz = np.array([0.0144, -0.58, 0.9662])
    joints = np.zeros(11)
    joints[HelloStretchIdx.LIFT] = 0.597
    joints[HelloStretchIdx.ARM] = 0.0113
    joints[HelloStretchIdx.WRIST_PITCH] = 0.487
    op.robot.get_joint_positions.return_value = joints
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    from emet.config.loader import load_config

    params = load_config("configs/emet/query_geometry_tracked_narrow_pilot.yaml").mapping_dict
    op.agent.manipulation_radius = params["motion_planner"]["goals"]["manipulation_radius"]
    assert not op.pregrasp_open_loop(op._object_xyz, distance_from_object=0.3)
    op.robot.arm_to.assert_not_called()

    def navigate(target, start, **kwargs):
        assert kwargs["distance_range"][0] > 0.7
        op.robot.get_base_pose_world.return_value = np.array([0, 0.16, 0])
        return True

    op.agent.navigate_to_target_pose.side_effect = navigate
    op.ensure_grounded_grasp_workspace()
    assert op.pregrasp_open_loop(op._object_xyz, distance_from_object=0.3)
    q = op.robot.arm_to.call_args.args[0]
    assert q[HelloStretchIdx.ARM] >= 0
    assert 0 < q[HelloStretchIdx.LIFT] < 1.0


def test_low_target_does_not_relocate_when_actual_pregrasp_is_reachable():
    from emet.motion.kinematics import HelloStretchKinematics

    op = operation()
    model = HelloStretchKinematics()
    op.robot_model = model
    op.robot.get_robot_model.return_value = model
    op._object_xyz = np.array([0.0, -0.56, 0.534])
    joints = np.zeros(11)
    joints[HelloStretchIdx.LIFT] = 0.6
    joints[HelloStretchIdx.ARM] = 0.01
    joints[HelloStretchIdx.WRIST_PITCH] = -1.5
    op.robot.get_joint_positions.return_value = joints
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    op.ensure_grounded_grasp_workspace()
    op.agent.navigate_to_target_pose.assert_not_called()
    op.robot.arm_to.assert_not_called()
    np.testing.assert_array_equal(op.robot.get_joint_positions(), joints)
    aimed = op.aim_grasp_joints(joints, op._object_xyz)
    q = op.solve_pregrasp(op._object_xyz, aimed, 0.3)
    assert q is not None and q[HelloStretchIdx.ARM] >= 0


def test_failed_alignment_does_not_reacquire_or_move_arm():
    op = operation()
    op.robot.move_base_to.return_value = False
    with pytest.raises(RuntimeError, match="orientation"):
        op.align_grounded_target_for_grasp()
    op.agent.prepare_query_target.assert_not_called()
    op.robot.arm_to.assert_not_called()


def test_world_target_alignment_does_not_use_episode_relative_odometry(monkeypatch):
    op = operation()
    op._object_xyz = np.array([4.0, -2.5, 0.52])
    op.robot.get_base_pose_world.return_value = np.array([4.0, -2.0, 1.0])
    op.robot.get_base_pose.return_value = np.zeros(3)
    op.agent.prepare_query_target.return_value = SimpleNamespace(xyz=op._object_xyz)
    monkeypatch.setattr("emet.controller.dynamem.look.wait_post_motion_obs", lambda *a, **k: None)
    op.align_grounded_target_for_grasp()
    np.testing.assert_allclose(op.robot.move_base_to.call_args.args[0], [4, -2, 0], atol=1e-10)
    assert op.robot.move_base_to.call_args.kwargs["world_frame"] is True
    op.robot.get_base_pose.assert_not_called()


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


def test_pregrasp_selects_reachable_standoff_without_clamping_invalid_ik():
    op = operation()
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.zeros(3), np.array([0, 0, 0, 1]))
    bad, good = np.zeros(11), np.zeros(11)
    bad[HelloStretchIdx.ARM] = -0.058
    good[HelloStretchIdx.ARM] = 0.02
    op.robot_model.manip_ik_for_grasp_frame.side_effect = [
        (bad, None, None, True, None),
        (good, None, None, True, None),
    ]
    assert op.pregrasp_open_loop(op.get_object_xyz(), distance_from_object=0.3)
    assert bad[HelloStretchIdx.ARM] == -0.058
    np.testing.assert_array_equal(op.robot.arm_to.call_args.args[0], good)
    assert op.robot_model.manip_ik_for_grasp_frame.call_count == 2


def test_pregrasp_angle_clipping_preserves_metric_standoff():
    op = operation()
    op.robot.get_base_pose.return_value = np.zeros(3)
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_robot_model.return_value.manip_fk.return_value = (np.array([0, 0, 1]), np.array([0, 0, 0, 1]))
    op.robot_model.manip_ik_for_grasp_frame.return_value = (np.zeros(11), None, None, True, None)
    target = np.array([0, -0.2, 0])
    assert op.pregrasp_open_loop(target, distance_from_object=0.3)
    requested = op.robot_model.manip_ik_for_grasp_frame.call_args.args[0]
    assert np.linalg.norm(requested - target) == pytest.approx(0.3)


def test_high_target_pregrasp_does_not_approach_from_below_its_support():
    from scipy.spatial.transform import Rotation

    op = operation()
    op.robot.get_base_pose_world.return_value = np.zeros(3)
    op.robot.get_joint_positions.return_value = np.zeros(11)
    # Looking up finds the object, but inserting upward from below its height
    # occludes the wrist behind the counter and approaches the support face.
    op.robot.get_robot_model.return_value.manip_fk.return_value = (
        np.array([0.0, -0.3, 0.6]),
        Rotation.from_euler("y", -0.37).as_quat(),
    )
    op.robot_model.manip_ik_for_grasp_frame.return_value = (np.zeros(11), None, None, True, None)
    target = np.array([0.0, -0.7, 1.0])
    assert op.pregrasp_open_loop(target, distance_from_object=0.3)
    position, quaternion = op.robot_model.manip_ik_for_grasp_frame.call_args.args
    assert position[2] >= target[2]
    assert np.linalg.norm(position - target) == pytest.approx(0.3)
    assert Rotation.from_quat(quaternion).as_euler("xyz")[1] == pytest.approx(0.0)


@pytest.mark.parametrize("center_depth", [0.0, 0.3])
def test_servo_stops_on_failed_motion_without_grasping(center_depth):
    op = operation()
    op.intro = op.warn = op.error = Mock()
    op.gripper_aruco_detector = Mock()
    op.pregrasp_open_loop = Mock(return_value=True)
    op.grounded_target = SimpleNamespace(geometry_source="vlm_selected_depth_surface")
    op.track_image_center = True
    op.open_loop = False
    mask = np.ones((8, 8), dtype=bool)
    op.get_target_mask = Mock(return_value=mask)
    op._compute_center_depth = Mock(return_value=center_depth)
    op.observations = Mock()
    op.observations.get_latest_centroid.return_value = np.array([4, 4])
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.get_servo_observation.return_value = SimpleNamespace(
        ee_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        ee_depth=np.full((8, 8), 0.3),
        get_ee_xyz_in_world_frame=lambda: np.ones((8, 8, 3)),
    )
    op.robot_model.manip_fk.return_value = (np.array([0, 0, 0.5]), None)
    op.robot.arm_to.return_value = False
    op._grasp = Mock()
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        assert op.visual_servo_to_object(None) is False
    op.robot.arm_to.assert_called_once()
    if center_depth == 0:
        # No forward arm/lift approach from missing center support.
        np.testing.assert_array_equal(op.robot.arm_to.call_args.args[0][:3], np.zeros(3))
    op._grasp.assert_not_called()


@pytest.mark.parametrize("motions", [[False], [True, False], [True, True]])
def test_grasp_propagates_approach_and_lift_failure(motions):
    op = operation()
    op.cheer = op.error = Mock()
    op.talk = False
    op.open_loop = False
    op.robot.get_joint_positions.return_value = np.zeros(11)
    op.robot.arm_to.side_effect = motions
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        assert op._grasp() is all(motions)
    if not motions[0]:
        op.robot.close_gripper.assert_not_called()


def test_geometry_grasp_does_not_overwrite_verified_pose_before_closure():
    op = operation()
    op.cheer = Mock()
    op.talk = False
    op.use_geometry_servo = True
    op.open_loop = False
    measured = np.arange(11, dtype=float) / 100
    op.robot.get_joint_positions.return_value = measured
    events = []
    op.robot.close_gripper.side_effect = lambda **kw: events.append("close")
    op.robot.arm_to.side_effect = lambda *a, **kw: (events.append("lift"), True)[1]
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        assert op._grasp()
    assert events == ["close", "lift"]
    expected = measured.copy()
    expected[HelloStretchIdx.LIFT] += 0.3
    np.testing.assert_array_equal(op.robot.arm_to.call_args.args[0], expected)


def test_center_depth_excludes_nonfinite_sensor_values():
    op = operation()
    servo = SimpleNamespace(ee_depth=np.array([[np.inf, np.nan], [0.0, 0.2]]))
    mask = np.ones((2, 2), dtype=bool)
    assert op._compute_center_depth(servo, mask, 1, 1) == pytest.approx(0.2)


@pytest.mark.parametrize("lift, goal", [(0.4, 0.7), (0.845, 1.0), (0.9, 1.0)])
def test_pickup_lift_respects_existing_stretch_height_limit(lift, goal):
    op = operation()
    op.cheer = op.error = Mock()
    op.talk = False
    op.use_geometry_servo = True
    joints = np.zeros(11)
    joints[HelloStretchIdx.LIFT] = lift
    op.robot.get_joint_positions.return_value = joints
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        assert op._grasp()
    assert op.robot.arm_to.call_args.args[0][HelloStretchIdx.LIFT] == pytest.approx(goal)
    assert joints[HelloStretchIdx.LIFT] == lift


@pytest.mark.parametrize("lift", [0.95, 1.0, 1.1, np.nan, np.inf])
def test_insufficient_or_unknown_lift_travel_stops_before_closure(lift):
    op = operation()
    op.cheer = op.error = Mock()
    op.talk = False
    op.use_geometry_servo = True
    joints = np.zeros(11)
    joints[HelloStretchIdx.LIFT] = lift
    op.robot.get_joint_positions.return_value = joints
    assert op._grasp() is False
    op.robot.close_gripper.assert_not_called()
    op.robot.arm_to.assert_not_called()


def test_changed_lift_feedback_after_closure_never_commands_lowering():
    op = operation()
    op.cheer = op.error = Mock()
    op.talk = False
    op.use_geometry_servo = True
    before, after = np.zeros(11), np.zeros(11)
    before[HelloStretchIdx.LIFT], after[HelloStretchIdx.LIFT] = 0.4, 0.8
    op.robot.get_joint_positions.side_effect = [before, before, after]
    with patch("emet.controller.operations.grasp_object.time.sleep"):
        assert op._grasp() is False
    op.robot.close_gripper.assert_called_once()
    op.robot.arm_to.assert_not_called()


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


@pytest.mark.parametrize("accepted", [False, True])
def test_wrist_tracking_saves_calibrated_evidence(tmp_path, monkeypatch, accepted):
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
        get_ee_xyz_in_world_frame=lambda: np.ones((8, 8, 3)),
    )
    op.agent.ground_vlm_frame.return_value = (
        SimpleNamespace(instance=np.full((8, 8), 0 if accepted else -1)),
        [],
        [0] if accepted else [],
        {"valid": accepted},
    )
    if accepted:
        assert op.get_target_mask(servo, center=(4, 4)).all()
    else:
        with pytest.raises(ValueError, match="absent or ambiguous"):
            op.get_target_mask(servo, center=(4, 4))
    records = list((tmp_path / "wrist_tracking").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["verification"]["valid"] is accepted
    assert record["verification"]["semantic_valid"] is accepted
    assert record["verification"]["association_valid"] is accepted
    assert record["metadata"]["camera_pose"] == np.eye(4).tolist()
    assert (records[0].parent / record["rgb_file"]).is_file()
