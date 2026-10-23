# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from emet.memory.graph_eqa import AgenticEQAExecutor
from emet.memory.graph_eqa.agentic.views import CapturedView, captured_view, target_in_view


def test_graphless_capture_verifies_exact_retained_frame_without_object_nodes():
    agent = MagicMock()
    agent.graph_memory = None
    agent.parameters = {}
    agent.voxel_map = SimpleNamespace(observations=[], encoder=None)
    frame = SimpleNamespace(rgb=np.full((4, 4, 3), 42, dtype=np.uint8), camera_pose=np.eye(4))
    agent.update.side_effect = lambda: agent.voxel_map.observations.append(frame)
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", max_rounds=4, router=False)
    ex._dense_max_sim_for_rgb = lambda *args: None
    ex._detector_for_verify = lambda: None
    ex._run_vlm_view_assess = MagicMock(return_value={"answerable": False})
    cap = ex._tool_capture_and_update()
    assert cap["ok"] and cap["status"] == "NEW_VIEW"
    oid = cap["obs_id"]
    assert ex._action_target_for_obs(oid).kind == "view"
    assert ex._view_identity_for_obs(oid) == (1, cap["view_id"])
    frame.rgb[:] = 99
    agent.robot.get_observation.reset_mock()
    ex._tool_verify_siglip("bowl", oid)
    agent.robot.get_observation.assert_not_called()
    np.testing.assert_array_equal(ex._run_vlm_view_assess.call_args.kwargs["rgb"], 42)
    assert captured_view(ex, oid).source_obs_id == 1
    assert agent.graph_memory is None
    agent.update.side_effect = None
    assert not ex._tool_capture_and_update()["ok"]


def test_voxel_presence_requires_matching_captured_frame():
    agent = MagicMock()
    agent.graph_memory = None
    agent.parameters = {}
    agent.voxel_map = SimpleNamespace(
        observations=[],
        encoder=None,
        find_alignment_over_model=lambda phrase: torch.tensor([0.9, 0.1]),
        semantic_memory=SimpleNamespace(_obs_counts=torch.tensor([99, 1])),
    )
    frame = SimpleNamespace(rgb=np.ones((4, 4, 3), dtype=np.uint8), camera_pose=np.eye(4))
    agent.update.side_effect = lambda: agent.voxel_map.observations.append(frame)
    ex = AgenticEQAExecutor(agent, "Where is the bowl?", max_rounds=4, router=False)
    oid = ex._tool_capture_and_update()["obs_id"]
    score, channel = ex._voxel_max_sim_for_obs("bowl", oid)
    assert abs(score - 0.1) < 1e-6 and channel == "voxel_obs"
    assert ex._voxel_max_sim_for_obs("bowl", 99) is None


def test_target_projection_uses_camera_world_pose_and_intrinsics():
    pose = np.eye(4)
    pose[:3, 3] = [3, -2, 1]
    view = CapturedView(1, 1, np.zeros((100, 100, 3)), pose, np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]]))
    assert target_in_view(view, [3, -2, 3])["target_in_frame"]
    assert target_in_view(view, [3, -2, 0])["status"] == "behind_camera"
    assert target_in_view(view, [7, -2, 3])["status"] == "outside_frame"


def test_molmo_missing_floor_patch_is_below_captured_view():
    """Replay calibration, not a claim of head reachability or free floor.

    Molmo seed 0, navigation-recovery-contract-20260930, floor capture
    1790825723337744477: blocked grid cell [529, 525] is world XY [1.7, 1.3].
    No simulator/assets needed. Z=0 is a diagnostic floor-height assumption;
    its +/-5 cm sensitivity must not change the out-of-frame diagnosis.
    """
    pose = np.array(
        [
            [0.8908319569, 0.3804366637, -0.2483674084, 1.4130874241],
            [0.4543327356, -0.7454153284, 0.4877886361, 1.6112846629],
            [0.0004358080, -0.5473791493, -0.8368846258, 1.2781680349],
            [0, 0, 0, 1],
        ]
    )
    intrinsics = np.array([[302.8082988, 0, 119.5020747], [0, 303.5241412, 211.5011765], [0, 0, 1]])
    view = CapturedView(1, 1, np.zeros((424, 240, 3), dtype=np.uint8), pose, intrinsics)
    target = np.array([1.7, 1.3, 0.0])
    for floor_z in (-0.05, 0.0, 0.05):
        result = target_in_view(view, [*target[:2], floor_z])
        assert result["status"] == "outside_frame"
        assert result["target_in_frame"] is False
        assert 0 < result["target_pixel_xy"][0] < 240
        assert result["target_pixel_xy"][1] > 424
    np.testing.assert_allclose(target_in_view(view, target)["target_pixel_xy"], [160.14, 584.70], atol=0.15)

    # A geometrically aimed camera puts the patch at the principal point.
    # Actual head limits/occlusion/depth validity still need a live check.
    forward = target - pose[:3, 3]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    aimed = pose.copy()
    aimed[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    aimed_view = CapturedView(2, 2, view.rgb, aimed, intrinsics)
    result = target_in_view(aimed_view, target)
    assert result["target_in_frame"] is True
    np.testing.assert_allclose(result["target_pixel_xy"], intrinsics[:2, 2])
