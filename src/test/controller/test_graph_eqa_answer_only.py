# Copyright (c) Chris Paxton 2026

"""GraphEQA answer-only mode (post-explore question bank) must skip frontier nav."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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


def test_run_eqa_one_iter_stays_when_find_view_already_attached():
    """q43: do not frontier-chase when Image 1 is already the clock/FIND view."""
    from emet.controller.controller_graph_eqa import GraphEQAController

    agent = _make_agent()
    agent.graph_memory.query_answer.return_value = (
        "clock reads 2-4pm",
        "2-4pm",
        False,
        "hard to read",
        None,
        [Image.new("RGB", (8, 8), color=(1, 2, 3))],
    )
    agent.graph_memory.eqa_stay_on_attached_view.return_value = True
    agent.graph_memory.eqa_attached_target_obs_id.return_value = 7
    agent.graph_memory.last_eqa_look_obs_id = None
    with patch.object(GraphEQAController, "_sync_graph_frontier_nodes", lambda self: None):
        with patch.object(GraphEQAController, "_rerun_refresh_monologue_panel", lambda self: None):
            GraphEQAController.run_eqa_one_iter(agent, "What time is it now?", allow_navigation=True)
    agent.navigate_to_target_pose.assert_not_called()
    agent.space.sample_frontier.assert_not_called()
    assert agent.graph_memory.last_eqa_look_obs_id == 7


def test_siglip_visual_find_maps_voxel_frames_and_ranks_by_score():
    """find_all_images voxel ids map to graph obs; score order beats chronological sort."""
    from types import SimpleNamespace

    import torch

    from emet.controller.controller_graph_eqa import GraphEQAController
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory

    rgb_a = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_b = np.ones((8, 8, 3), dtype=np.uint8)
    mem = GraphEQAMemory(defer_llm_clients=True)
    mem.add_observation(rgb_a, np.array([0.0, 0.0, 0.5]), ["railing"], viewer_xyz=np.array([0.0, 0.0, 1.0]))
    mem.add_observation(rgb_b, np.array([8.0, 8.0, 0.5]), ["island"], viewer_xyz=np.array([8.0, 8.0, 1.0]))

    def _frame(rgb, cam):
        pose = np.eye(4)
        pose[:3, 3] = cam
        return SimpleNamespace(rgb=rgb, camera_pose=pose)

    agent = _make_agent()
    agent.graph_memory = mem
    agent.voxel_map = MagicMock()
    agent.voxel_map.observations = [
        _frame(rgb_a, [0.0, 0.0, 1.0]),
        _frame(rgb_b, [8.0, 8.0, 1.0]),
    ]
    agent.voxel_map.find_all_images.return_value = (
        torch.tensor([1, 2]),
        torch.tensor([[0.0, 0.0, 0.5], [8.0, 8.0, 0.5]]),
        torch.tensor([0.22, 0.41]),
    )
    ranked = GraphEQAController._siglip_visual_find(agent, "stool", 4)
    assert [oid for _s, oid in ranked] == [2, 1]
    assert ranked[0][0] == pytest.approx(0.41, abs=1e-5)


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
