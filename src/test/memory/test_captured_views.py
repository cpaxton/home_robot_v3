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
