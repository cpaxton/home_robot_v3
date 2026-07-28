# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Tests for VLM frontier scoring helpers (controller_graph_eqa + agentic explore)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from emet.controller.controller_graph_eqa import GraphEQAController, _parse_image_pick


def test_parse_image_pick():
    assert _parse_image_pick("2", 4) == 1
    assert _parse_image_pick("Image 3 looks most promising.", 6) == 2
    assert _parse_image_pick("I would pick image 1.", 3) == 0
    assert _parse_image_pick("7", 6) is None  # out of range
    assert _parse_image_pick("0", 6) is None  # 1-based
    assert _parse_image_pick("none of these", 4) is None
    assert _parse_image_pick("", 4) is None


def test_vlm_frontier_choice_ranks_reachable_rgb_pool():
    """Sample utility-ranked frontier RGBs; VLM image pick selects the waypoint."""
    agent = GraphEQAController.__new__(GraphEQAController)
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    # Non-Habitat path: skip navmesh filtering.
    agent.robot.__class__.__name__ = "FakeRobot"

    far = SimpleNamespace(
        is_frontier=True,
        obs_id=2,
        xyz=np.array([5.0, 0.0, 1.0]),
        labels=["area"],
        frontier_cell_count=40,
        frontier_keyword_score=0.0,
        nav_failures=0,
    )
    near = SimpleNamespace(
        is_frontier=True,
        obs_id=1,
        xyz=np.array([1.0, 0.0, 1.0]),
        labels=["area"],
        frontier_cell_count=4,
        frontier_keyword_score=0.0,
        nav_failures=0,
    )
    gm = MagicMock()
    gm.eqa_client = MagicMock(return_value="Image 1")
    gm.get_nodes.return_value = [far, near]
    gm._observation_by_id.side_effect = lambda oid: SimpleNamespace(
        rgb=np.full((8, 8, 3), int(oid) * 40, dtype=np.uint8)
    )
    agent.graph_memory = gm
    agent.voxel_map = None

    # Patch habitat check to False so we don't need a real Habitat client.
    import emet.controller.controller_graph_eqa as mod

    old = mod.is_habitat_robot_client
    mod.is_habitat_robot_client = lambda _r: False
    try:
        pt = agent._vlm_frontier_choice("Where is the fruit bowl?")
    finally:
        mod.is_habitat_robot_client = old

    assert pt is not None
    # Far has higher area utility → listed first as Image 1 → VLM picks it.
    assert abs(float(pt[0]) - 5.0) < 1e-6
    assert gm.eqa_client.called
    args = gm.eqa_client.call_args[0][0]
    assert isinstance(args, list) and len(args) == 3  # directive + 2 images


def test_agentic_explore_prefers_vlm_frontier_candidate():
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor

    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_max_nav_steps": 8}}
    agent.graph_memory = MagicMock()
    agent.graph_memory.get_nodes.return_value = []
    vlm_xy = np.array([3.0, 4.0, 1.0], dtype=float)
    agent._vlm_frontier_choice = MagicMock(return_value=vlm_xy)
    agent._siglip_guided_frontier = MagicMock(return_value=None)
    agent._best_frontier_point_from_graph = MagicMock(return_value=None)
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent.navigate_to_target_pose = MagicMock(return_value=True)

    class _FakeRobot:
        def get_base_pose(self):
            return np.array([0.0, 0.0, 0.0])

    agent.robot = _FakeRobot()

    ex = AgenticEQAExecutor(agent, "Where is the clock?", router=False, collect_trace=True)
    ex._tool_capture_and_update = MagicMock(return_value={"ok": True, "obs_id": 9})
    ex._verify_after_motion = MagicMock(return_value={"ok": True})
    ex._save_frontier_pick_panel = MagicMock(return_value=None)
    ex._robot_xyt = MagicMock(return_value=np.array([0.0, 0.0, 0.0]))
    ex._begin_policy_approach = MagicMock(return_value="h1")
    ex._policy_approached = MagicMock()
    ex._attach_gt = MagicMock()
    ex._escape_min_travel_m = MagicMock(return_value=0.0)

    out = ex._tool_explore_frontier(toward="wall clock")
    assert out["ok"] is True
    agent._vlm_frontier_choice.assert_called()
    nav_goal = agent.navigate_to_target_pose.call_args[0][0]
    assert abs(float(nav_goal[0]) - 3.0) < 1e-6
    assert abs(float(nav_goal[1]) - 4.0) < 1e-6
    row = next(r for r in reversed(ex._trace_rows) if r.get("tool") == "explore_frontier")
    assert row.get("source") == "vlm_frontier"
