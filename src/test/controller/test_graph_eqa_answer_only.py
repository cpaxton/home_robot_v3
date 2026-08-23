# Copyright (c) Chris Paxton 2026

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


def test_agentic_verify_defaults_off_for_discord_herman():
    """Discord / Herman must keep classic run_eqa unless operator opts in."""
    from emet.memory.graph_eqa.agentic_eqa import agentic_verify_enabled

    agent = _make_agent()
    agent.parameters = {}
    assert agentic_verify_enabled(agent) is False
    agent.parameters = {"eqa": {"agentic_verify": False}}
    assert agentic_verify_enabled(agent) is False


def test_run_eqa_uses_classic_path_when_agentic_off(monkeypatch):
    """GraphEQAController.run_eqa must not enter agentic loop by default."""
    import emet.memory.graph_eqa.agentic_eqa as agentic_eqa
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    agent.parameters = {"eqa": {"agentic_verify": False}}
    called = {"agentic": 0}

    def _boom(*_a, **_k):
        called["agentic"] += 1
        raise AssertionError("agentic path must not run when agentic_verify is off")

    monkeypatch.setattr(agentic_eqa, "run_agentic_eqa", _boom)
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
    assert called["agentic"] == 0
    assert "near the counter" in discord_text
    one_iter.assert_called_once()


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


def test_run_eqa_single_step_allows_nav_when_enabled():
    """max_planning_steps=1 + allow_navigation=True still passes nav into one_iter."""
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    one_iter = MagicMock(return_value=("near the counter", "near the counter", [], True))
    agent.run_eqa_one_iter = one_iter
    GraphEQAController.run_eqa(
        agent,
        "Where is the sink?",
        max_planning_steps=1,
        allow_navigation=True,
    )
    assert one_iter.call_args.kwargs.get("allow_navigation") is True


def test_run_eqa_multi_step_skips_nav_on_final_step():
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    nav_flags: list[bool] = []

    def _one_iter(*_a, allow_navigation=True, **_kw):
        nav_flags.append(bool(allow_navigation))
        return ("a", "a", [], False)

    agent.run_eqa_one_iter = _one_iter
    agent.update = MagicMock()
    GraphEQAController.run_eqa(
        agent,
        "Where is the sink?",
        max_planning_steps=3,
        allow_navigation=True,
    )
    assert nav_flags == [True, True, False]
