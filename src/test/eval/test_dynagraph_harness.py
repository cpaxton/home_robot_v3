# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from unittest.mock import MagicMock, patch

from emet.core.parameters import get_parameters
from emet.eval.benchmark_dynagraph import (
    apply_dynagraph_harness,
    apply_dynagraph_harness_overrides,
    apply_habitat_eqa_method_parameters,
    dynagraph_harness_flags,
    harness_controller_options,
    resolve_harness_profile,
)


def test_habitat_eqa_harness_profile_and_flags():
    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    assert resolve_harness_profile("habitat_eqa") == "unified_eqa"
    assert params.get("dynagraph_merge_xy_m") == 0.45
    assert params.get("dynagraph_staleness_horizon") == 256
    flags = dynagraph_harness_flags(params)
    assert flags["memory_summary"] is True
    assert flags["mcq_debias"] is False
    assert flags["explore_when_uncovered"] == "conservative"
    assert flags["siglip_grounding"] is True


def test_enrich_episode_metrics_harness_fingerprint_merge_on():
    """Episode JSONL fingerprint must record merge-on unified_eqa defaults."""
    from types import SimpleNamespace

    from emet.habitat.episode_debug import enrich_episode_metrics
    from emet.habitat.metrics import EpisodeMetrics

    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    agent = SimpleNamespace(parameters=params, graph_memory=None)
    metrics = EpisodeMetrics(
        dataset="hmeqa",
        method="dynagraph",
        question_id=17,
        scene="s",
        floor=0,
        question="q",
        gold_answer_letter="D",
        predicted_answer="D",
        correct=True,
        confident=True,
        planning_steps=1,
        success=True,
    )
    enrich_episode_metrics(metrics, agent=agent, choices=["a", "b", "c", "d"])
    assert float(metrics.harness.get("dynagraph_merge_xy_m")) == 0.45
    assert float(metrics.harness.get("fallback_spatial_merge_xy_m")) == 0.45
    assert metrics.harness.get("profile") == "unified_eqa"
    assert metrics.harness.get("explore_when_uncovered") == "conservative"


def test_habitat_eqa_overrides_layered():
    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    apply_dynagraph_harness_overrides(params, mcq_debias=True, explore_when_uncovered="off")
    flags = dynagraph_harness_flags(params)
    assert flags["mcq_debias"] is True
    assert flags["explore_when_uncovered"] == "off"


def test_sqa3d_harness_disables_eqa_extras():
    params = apply_dynagraph_harness(get_parameters("dynav_config.yaml"), "sqa3d", "dynagraph")
    flags = dynagraph_harness_flags(params)
    assert flags["memory_summary"] is False
    assert flags["mcq_debias"] is False
    assert flags["explore_when_uncovered"] == "off"


def test_habitat_ovmm_find_harness():
    opts = harness_controller_options("habitat_ovmm_find", "dynagraph")
    assert opts.get("eqa") is False
    assert opts.get("use_instance_graph") is True


@patch("emet.controller.controller_dynagraph.GraphEQAController.__init__", return_value=None)
def test_dynagraph_controller_reads_harness_flags(mock_init: MagicMock):
    from emet.controller.controller_dynagraph import DynagraphController

    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    agent = DynagraphController.__new__(DynagraphController)
    agent.ground_truth_mode = False
    agent.visualize_ground_truth = False
    agent._gt_graph_loaded = False
    agent._skip_graph_perception_updates = False
    agent.graph_memory = MagicMock()
    agent.rerun_visualizer = MagicMock(enabled=False)

    flags = dynagraph_harness_flags(params)
    explore_mode = str(flags.get("explore_when_uncovered", "off"))
    agent._eqa_explore_when_uncovered = explore_mode in ("on", "conservative")
    agent._eqa_explore_uncovered_habitat_frontier = explore_mode in ("on", "conservative")
    agent.graph_memory.memory_summary_enabled = bool(flags.get("memory_summary", False))
    agent.graph_memory.mcq_debias_enabled = bool(flags.get("mcq_debias", False))

    assert agent._eqa_explore_when_uncovered is True
    assert agent._eqa_explore_uncovered_habitat_frontier is True
    assert agent.graph_memory.memory_summary_enabled is True
    assert agent.graph_memory.mcq_debias_enabled is False
