# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""GraphEQA answer-only mode (post-explore question bank) must skip frontier nav."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image


def _make_agent():
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = MagicMock(spec=GraphEQAController)
    agent._realtime_updates = False
    agent._fast_explore_lookaround = True
    agent._eqa_explore_when_uncovered = True
    agent._vlm_frontier_scoring = False
    agent._habitat_blocked_goals = set()
    agent._habitat_recent_goals = []
    agent.parameters = {"eqa_stall_patience": 0}
    agent.graph_memory = MagicMock()
    agent.graph_memory.query_answer.return_value = (
        "because",
        "near the counter",
        False,
        "still exploring",
        np.array([1.0, 2.0, 0.0]),
        [Image.new("RGB", (8, 8), color=(1, 2, 3))],
    )
    agent.graph_memory.get_nodes.return_value = []
    agent.graph_memory.last_eqa_action_obs_id = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.planner = MagicMock()
    agent._planning_base_xyt = lambda xyt: xyt
    agent.navigate_to_target_pose = MagicMock(return_value=True)
    agent.rerun_visualizer = MagicMock()
    agent.space = MagicMock()
    agent.voxel_map = MagicMock()
    return agent


def test_run_eqa_one_iter_skips_nav_when_disallowed():
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    with patch.object(GraphEQAController, "_sync_graph_frontier_nodes", lambda self: None):
        with patch.object(GraphEQAController, "_rerun_refresh_monologue_panel", lambda self: None):
            out = GraphEQAController.run_eqa_one_iter(
                agent,
                "Where is the sink?",
                allow_navigation=False,
            )
    _answer, discord_text, _imgs, confidence = out
    assert confidence is False
    assert "near the counter" in discord_text
    agent.navigate_to_target_pose.assert_not_called()
    agent.look_around.assert_not_called()
    agent.robot.look_front.assert_called()


def test_run_eqa_answer_only_passes_allow_navigation_false():
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    one_iter = MagicMock(
        return_value=(
            "near the counter",
            "near the counter\nI also provide relevant images here.",
            [],
            False,
        )
    )
    agent.run_eqa_one_iter = one_iter
    discord_text, _imgs = GraphEQAController.run_eqa(
        agent,
        "Where is the sink?",
        max_planning_steps=1,
        allow_navigation=False,
    )
    assert "near the counter" in discord_text
    one_iter.assert_called_once()
    assert one_iter.call_args.kwargs.get("allow_navigation") is False
