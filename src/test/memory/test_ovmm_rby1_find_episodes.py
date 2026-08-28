# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""OVMM find-phase rby1 fast-gate episodes."""

from __future__ import annotations

from emet.eval.ovmm_find_phase import load_find_phase_episodes


def test_rby1_fast_gate_episodes_present():
    eps = load_find_phase_episodes("configs/ovmm/find_phase_episodes.yaml")
    by_id = {e.id: e for e in eps}
    assert "default_table_rby1_s0_distinct_recep" in by_id
    assert "robocasa_rby1_pp_s1" in by_id
    assert by_id["default_table_rby1_s0_distinct_recep"].sim.endswith("default_table_rby1.yaml")
    assert by_id["robocasa_rby1_pp_s1"].sim.endswith("robocasa_pick_place_rby1.yaml")
    assert by_id["robocasa_rby1_pp_s1"].object_gt_body == "obj_main"


def test_run_mapping_protocol_rotates_default_table_rby1():
    from unittest.mock import MagicMock

    from emet.eval.ovmm_find_phase import run_mapping_protocol

    agent = MagicMock()
    agent.robot.get_emet_session.return_value = {
        "emet_robot_id": "rby1",
        "environment": {"kind": "default_table"},
    }
    agent.rotate_in_place = MagicMock()

    n = run_mapping_protocol(agent, explore_steps=0, not_rotate=False, mapping_rotate_steps=4)

    assert n == 1
    agent.rotate_in_place.assert_called_once_with(n_steps=4)


def test_prepare_default_table_rby1_mapping_view_moves_and_looks():
    from unittest.mock import MagicMock

    import numpy as np

    from emet.eval.ovmm_find_phase import _prepare_default_table_rby1_mapping_view

    robot = MagicMock()
    robot.get_emet_session.return_value = {
        "emet_robot_id": "rby1",
        "environment": {"kind": "default_table"},
    }
    agent = MagicMock()
    agent.robot = robot
    agent._find_phase_nav_timeout = lambda: 12.0

    assert _prepare_default_table_rby1_mapping_view(agent) is True

    robot.move_base_to.assert_called_once()
    goal = robot.move_base_to.call_args[0][0]
    np.testing.assert_allclose(goal, [0.0, 1.5, np.pi], rtol=0, atol=1e-6)
    assert robot.move_base_to.call_args.kwargs["timeout"] == 12.0
    robot.look_front.assert_called_once()


def test_run_mapping_protocol_agentic_explore_for_non_s0():
    """Mapping with explore_steps>0 uses AgenticEQAExecutor mode=explore, not execute_action."""
    from unittest.mock import MagicMock, patch

    from emet.eval.ovmm_find_phase import run_mapping_protocol

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.voxel_map = MagicMock()
    agent.ground_truth_mode = False
    agent._seed_local_radius_explored = MagicMock()
    agent.execute_action = MagicMock()
    agent.rotate_in_place = MagicMock()

    with patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result") as mock_run:
        mock_result = MagicMock()
        mock_result.n_explore = 2
        mock_result.n_nav = 1
        mock_run.return_value = mock_result

        n = run_mapping_protocol(agent, explore_steps=3, not_rotate=False, trace_meta={"ovmm_phase": "mapping"})

        assert n == 3
        agent.execute_action.assert_not_called()
        # Kitchen mapping must not spin in place; S0 rotate-only is explore_steps==0.
        agent.rotate_in_place.assert_not_called()
        mock_run.assert_called_once()
        # coverage only: question None, no object toward
        assert mock_run.call_args[0][1] is None
        assert mock_run.call_args.kwargs["goal"] == "explore and map the environment"
        assert mock_run.call_args.kwargs["max_nav_steps"] == 3
        assert mock_run.call_args.kwargs["max_rounds"] == 4


def test_run_mapping_protocol_falls_back_when_agentic_maps_zero_steps():
    """Empty agentic explore (0 nav + 0 explore) must not skip rotate/execute_action."""
    from unittest.mock import MagicMock, patch

    from emet.eval.ovmm_find_phase import run_mapping_protocol

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.voxel_map = MagicMock()
    agent.ground_truth_mode = False
    agent._seed_local_radius_explored = MagicMock()
    agent.execute_action = MagicMock()
    agent.rotate_in_place = MagicMock()

    with patch("emet.memory.graph_eqa.agentic_eqa.run_agentic_eqa_result") as mock_run:
        mock_result = MagicMock()
        mock_result.n_explore = 0
        mock_result.n_nav = 0
        mock_run.return_value = mock_result

        n = run_mapping_protocol(agent, explore_steps=3, not_rotate=False, trace_meta={"ovmm_phase": "mapping"})

    assert n == 4  # 1 rotate + 3 execute_action
    agent.rotate_in_place.assert_called_once()
    assert agent.execute_action.call_count == 3
    assert agent._ovmm_mapping_result is mock_result


def test_explore_frontier_faces_frontier_and_look_ahead():
    """After nav, explore faces frontier, look_ahead (tilt 0), then capture/update."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.voxel_map = MagicMock()
    agent.planner = MagicMock()
    robot = MagicMock()
    robot.look_ahead = MagicMock()
    robot.wait_for_obs = MagicMock()
    agent.robot = robot
    agent.navigate_to_target_pose = MagicMock(return_value=MagicMock(__bool__=lambda s: True, finished=True, ok=True))
    agent._last_nav_attempt = MagicMock(success=True, finished=True, dist_m=1.0, note="ok")

    ex = AgenticEQAExecutor(agent, question=None, max_rounds=5, max_nav_steps=5, router=False)
    ex._robot_xyt_world = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))
    ex.graph_memory.graph_room_at_robot = MagicMock(return_value="unknown")
    ex.graph_memory.frontier_id_near_xy = MagicMock(return_value="")
    ex.graph_memory.world_evidence = MagicMock()
    ex.graph_memory.world_evidence.frontiers = {}
    ex._tool_capture_and_update = MagicMock(return_value={"ok": True, "obs_id": 42})
    ex._begin_policy_approach = MagicMock(return_value="hypo1")
    ex._policy_approached = MagicMock()
    ex._retire_visited_frontier = MagicMock()
    ex._save_frontier_pick_panel = MagicMock(return_value=None)
    ex._attach_gt = MagicMock()
    ex._append_trace = MagicMock()
    ex._refresh_room_after_motion = MagicMock()

    agent._vlm_frontier_choice = MagicMock(return_value=None)
    agent._siglip_guided_frontier = MagicMock(return_value=None)
    agent._best_frontier_point_from_graph = MagicMock(return_value=None)
    with patch(
        "emet.controller.habitat_nav.pick_uncovered_explore_target",
        return_value=np.array([5.0, 0.0, 0.0]),
    ):
        out = ex._tool_explore_frontier(toward="", frontier_id="")

    assert out.get("ok") is True
    # target_theta must face frontier (atan2 0)
    assert agent.navigate_to_target_pose.called
    theta = agent.navigate_to_target_pose.call_args[0][2]
    assert theta is not None
    import math

    assert math.isclose(theta, 0.0, abs_tol=1e-6)
    # arrival look_ahead then capture
    robot.look_ahead.assert_called_once()
    ex._tool_capture_and_update.assert_called_once()


def test_voxel_proposal_beats_camera_pose_view():
    """Unused voxel handle is investigated before a nearby camera-pose graph card."""
    from unittest.mock import MagicMock

    import numpy as np

    from emet.mapping.voxel_localize import voxel_proposal_id
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.graph_memory = MagicMock()
    agent.voxel_map = MagicMock()
    agent.planner = MagicMock()
    agent.robot = MagicMock()
    agent.robot.get_base_pose = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))
    agent.navigate_to_target_pose = MagicMock(return_value=MagicMock(finished=True, ok=True, __bool__=lambda s: True))

    ex = AgenticEQAExecutor(
        agent, question="Where is the jar on the counter?", max_rounds=8, max_nav_steps=8, router=False
    )
    ex._robot_xyt_world = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))
    voxel_xyz = np.array([1.0, 0.0, 0.5])
    voxel_h = NavHypothesis(phrase="jar", obs_id=voxel_proposal_id(0), xyz=voxel_xyz, score=400, source="voxel")
    cam_h = NavHypothesis(phrase="jar", obs_id=1, xyz=np.array([0.05, 0.05, 1.2]), score=10, source="graph")
    ex._hypotheses = [voxel_h, cam_h]
    # unused detection should beat camera pose
    best = ex._unused_detection_hypothesis()
    assert best is not None and int(best.obs_id) == int(voxel_h.obs_id)
    ex._dist_to_anchor_m = MagicMock(return_value=0.05)
    assert ex._hypothesis_is_camera_pose_place(cam_h) is True
    # nearby search should not pick the camera-pose card when detection exists
    ex._dist_to_anchor_m = MagicMock(return_value=1.0)
    # rewire helper to avoid grounded filter
    ex._grounded_visible_place_obs = MagicMock(return_value=None)
    ex._obs_already_verified = MagicMock(return_value=False)
    ex._hypothesis_nav_blocked = MagicMock(return_value=False)
    ex._place_approaches_exhausted = MagicMock(return_value=False)
    ex._nav_to_obs_counts = {}
    ex._tried = {}
    ex._place_inspect = {}
    ex._close_map_attempts = {}
    ex._unreachable_obs_ids = set()
    ex.action_progress_mode = "off"
    ex.decision_policy = "legacy"
    nearby = ex._nearby_untried_investigate_hyp(max_dist_m=3.5)
    assert nearby is not None
    assert int(nearby.obs_id) == int(voxel_h.obs_id)
