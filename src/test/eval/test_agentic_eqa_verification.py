# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Hang / reliability claims for agentic EQA (H*).

Run the whole agentic gate (CPU, no GPU)::

    uv run emet test src/test/eval/test_agentic_*.py -v

Sibling files:
  test_agentic_loop.py              A* loop + S1
  test_agentic_router_tools.py      T* router / schemas
  test_agentic_multimodal_cache.py  C* cache + D* traces
  test_agentic_room_and_nav.py      room, frontier, investigate, nav
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from emet.controller.habitat_nav import NavOutcome

# ---------------------------------------------------------------------------
# Hang / reliability — must pass on current branch
# ---------------------------------------------------------------------------


def test_H1_release_zmq_ports_listeners_only():
    """H1: release_zmq_ports must pass listeners_only=True to avoid killing clients."""
    from emet.utils.port_utils import release_zmq_ports

    with patch("emet.utils.port_utils.kill_processes_on_port", return_value=True) as mock_kill:
        freed = release_zmq_ports(0)
    assert freed == [4401, 4402, 4403, 4404]
    assert mock_kill.call_count == 4
    for call in mock_kill.call_args_list:
        assert call.kwargs.get("listeners_only") is True


def test_H2_vram_prep_warms_then_releases_siglip():
    """H2: CONFIRMED_MEMORY features cached; encoders None before VLM."""
    from emet.eval.dynagraph_vram import prepare_dynagraph_vram_for_eqa
    from emet.memory.graph_eqa import GraphEQAMemory

    class Enc:
        def encode_image(self, rgb):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

        def encode_text(self, text):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    enc = Enc()
    agent = MagicMock()
    agent.encoder = enc
    agent.voxel_map = MagicMock()
    agent.voxel_map.encoder = enc
    gm = GraphEQAMemory(defer_llm_clients=True)
    gm.memory_summary_enabled = True
    gm.add_observation(np.zeros((8, 8, 3), dtype=np.uint8), np.array([1.0, 2.0, 0.5]), ["plant"])
    gm._relevant_phrases = ["woven basket"]
    agent.graph_memory = gm

    prepare_dynagraph_vram_for_eqa(agent)
    assert agent.encoder is None
    assert agent.voxel_map.encoder is None
    assert gm._confirmed_memory_siglip_encoder is None
    assert gm._obs_siglip_features


def test_agentic_executor_consumes_manifest_budget_environment(monkeypatch):
    from emet.memory.graph_eqa.agentic_eqa import build_agentic_eqa_executor

    monkeypatch.setenv("EMET_EQA_AGENTIC_MAX_TOOL_ROUNDS", "5")
    monkeypatch.setenv("EMET_EQA_AGENTIC_MAX_NAV_STEPS", "4")
    agent = MagicMock()
    agent.parameters = {"eqa": {"agentic_max_tool_rounds": 8, "agentic_max_nav_steps": 8}}
    with patch("emet.eval.dynagraph_vram.warm_siglip_confirmed_memory"):
        executor = build_agentic_eqa_executor(agent, "Where is the chair?")
    assert executor.max_rounds == 5
    assert executor.max_nav_steps == 4


def test_H3_answer_only_skips_nav():
    """H3: allow_navigation=False → no navigate_to_target_pose / look_around."""
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
        [Image.new("RGB", (8, 8))],
    )
    agent.graph_memory.get_nodes.return_value = []
    agent.graph_memory.last_eqa_action_obs_id = None
    agent.robot = MagicMock()
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])
    agent.planner = MagicMock()
    agent._planning_base_xyt = lambda xyt: xyt
    agent.navigate_to_target_pose = MagicMock(return_value=NavOutcome.REACHED)
    agent.rerun_visualizer = MagicMock()
    agent.space = MagicMock()
    agent.voxel_map = MagicMock()

    with patch.object(GraphEQAController, "_sync_graph_frontier_nodes", lambda self: None):
        with patch.object(GraphEQAController, "_rerun_refresh_monologue_panel", lambda self: None):
            GraphEQAController.run_eqa_one_iter(agent, "Where is the sink?", allow_navigation=False)

    agent.navigate_to_target_pose.assert_not_called()
    agent.look_around.assert_not_called()
