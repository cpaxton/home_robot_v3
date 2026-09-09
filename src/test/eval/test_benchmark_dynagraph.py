# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

from emet.core.parameters import get_parameters
from emet.eval.benchmark_dynagraph import (
    apply_habitat_eqa_method_parameters,
    apply_ovmm_backend_dynagraph,
    apply_sqa3d_dynagraph,
    dynagraph_harness_flags,
    harness_controller_options,
    profile_settings,
    resolve_harness_profile,
    resolve_ovmm_dynagraph_profile,
    resolve_sqa3d_dynagraph_profile,
)
from emet.eval.ovmm_find_phase import apply_backend_parameters


def test_dynav_config_matches_interactive_profile():
    params = get_parameters("dynav_config.yaml")
    interactive = profile_settings("interactive")
    assert params.get("dynagraph_merge_xy_m") == interactive["dynagraph_merge_xy_m"]
    assert params.get("dynagraph_staleness_horizon") == interactive["dynagraph_staleness_horizon"]


def test_ovmm_and_shared_module_agree():
    base = get_parameters("dynav_config.yaml")
    for backend in ("static_graph", "graph_eqa", "dynagraph", "ground_truth"):
        via_shared = apply_ovmm_backend_dynagraph(base, backend)
        via_legacy = apply_backend_parameters(get_parameters("dynav_config.yaml"), backend)  # type: ignore[arg-type]
        assert via_shared.get("dynagraph_merge_xy_m") == via_legacy.get("dynagraph_merge_xy_m")
        assert via_shared.get("dynagraph_staleness_horizon") == via_legacy.get("dynagraph_staleness_horizon")


def test_ovmm_find_phase_profile_tighter_than_interactive():
    params = apply_ovmm_backend_dynagraph(get_parameters("dynav_config.yaml"), "dynagraph")
    interactive = profile_settings("interactive")
    assert params.get("dynagraph_merge_xy_m") < interactive["dynagraph_merge_xy_m"]
    assert params.get("dynagraph_staleness_horizon") == interactive["dynagraph_staleness_horizon"]


def test_s0_parity_oneshot_uses_interactive_merge_agentic_keeps_find_phase():
    interactive = profile_settings("interactive")
    find_phase = apply_ovmm_backend_dynagraph(get_parameters("dynav_config.yaml"), "dynagraph")
    oneshot = apply_backend_parameters(
        get_parameters("dynav_config.yaml"), "dynagraph", s0_parity=True, use_agentic=False
    )
    agentic = apply_backend_parameters(
        get_parameters("dynav_config.yaml"), "dynagraph", s0_parity=True, use_agentic=True
    )
    assert oneshot.get("dynagraph_merge_xy_m") == interactive["dynagraph_merge_xy_m"]
    assert agentic.get("dynagraph_merge_xy_m") == find_phase.get("dynagraph_merge_xy_m")
    assert agentic.get("dynagraph_merge_xy_m") < interactive["dynagraph_merge_xy_m"]


def test_sqa3d_tuned_uses_interactive_merge():
    params = apply_sqa3d_dynagraph(get_parameters("dynav_config.yaml"), method="dynagraph", profile="tuned")
    interactive = profile_settings("interactive")
    assert params.get("dynagraph_merge_xy_m") == interactive["dynagraph_merge_xy_m"]
    eqa = profile_settings("eqa")
    assert (
        params.get("graph_eqa_extract", {}).get("navigation_samples_max")
        == eqa["graph_eqa_extract"]["navigation_samples_max"]
    )


def test_sqa3d_smoke_disables_merge():
    params = apply_sqa3d_dynagraph(get_parameters("dynav_config.yaml"), method="dynagraph", profile="smoke")
    smoke = profile_settings("smoke")
    assert params.get("dynagraph_merge_xy_m") == smoke["dynagraph_merge_xy_m"]
    assert params.get("dynagraph_staleness_horizon") == smoke["dynagraph_staleness_horizon"]


def test_profile_resolution():
    assert resolve_ovmm_dynagraph_profile("static_graph") == "static_graph"
    assert resolve_ovmm_dynagraph_profile("graph_eqa") == "static_graph"  # legacy alias
    assert resolve_ovmm_dynagraph_profile("dynagraph") == "find_phase"
    assert resolve_sqa3d_dynagraph_profile("dynagraph", profile="tuned") == "eqa"
    assert resolve_sqa3d_dynagraph_profile("dynamem", profile="tuned") is None


def test_ovmm_static_graph_uses_baseline_profile():
    for backend in ("static_graph", "graph_eqa"):
        params = apply_ovmm_backend_dynagraph(get_parameters("dynav_config.yaml"), backend)
        assert params.get("dynagraph_merge_xy_m") == 0.0
        assert params.get("dynagraph_staleness_horizon") == 0


def test_harness_controller_docs_present():
    ovmm = harness_controller_options("ovmm_find_phase", "dynagraph")
    sqa3d = harness_controller_options("sqa3d", "dynagraph")
    habitat = harness_controller_options("habitat_eqa", "dynagraph")
    assert ovmm.get("use_instance_graph") is True
    assert sqa3d.get("use_instance_graph") is False
    assert sqa3d.get("prompt_variant") == "sqa3d"
    assert habitat.get("mcq_debias") is False
    assert habitat.get("explore_when_uncovered") == "conservative"


def test_dynamic_explore_static_graph_baseline_merge():
    from emet.eval.benchmark_dynagraph import apply_dynamic_explore_backend

    for backend in ("static_graph", "graph_eqa"):
        params = apply_dynamic_explore_backend(get_parameters("dynav_config.yaml"), backend)
        assert params.get("dynagraph_merge_xy_m") == 0.0
        flags = dynagraph_harness_flags(params)
        assert flags["explore_when_uncovered"] == "off"


def test_habitat_eqa_method_parameters_use_unified_eqa_profile():
    params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), "dynagraph")
    unified = profile_settings("unified_eqa")
    assert resolve_harness_profile("habitat_eqa") == "unified_eqa"
    assert params.get("dynagraph_merge_xy_m") == unified["dynagraph_merge_xy_m"] == 0.45
    assert params.get("dynagraph_staleness_horizon") == unified["dynagraph_staleness_horizon"] == 256
    assert (params.get("dynagraph_harness") or {}).get("profile") == "unified_eqa"
    flags = dynagraph_harness_flags(params)
    assert flags["mcq_debias"] is False
    assert flags["explore_when_uncovered"] == "conservative"
    fusion = params.get("graph_object_fusion") or {}
    assert float(fusion.get("fallback_spatial_merge_xy_m", -1)) == 0.45


def test_habitat_eqa_static_graph_uses_baseline_zero_merge():
    """HM-EQA ``static_graph`` (alias ``graph_eqa``) is the GraphEQA-inspired paper row."""
    for method in ("static_graph", "graph_eqa"):
        params = apply_habitat_eqa_method_parameters(get_parameters("dynav_config.yaml"), method)
        baseline = profile_settings("static_graph")
        assert params.get("dynagraph_merge_xy_m") == baseline["dynagraph_merge_xy_m"] == 0.0
        assert params.get("dynagraph_staleness_horizon") == baseline["dynagraph_staleness_horizon"] == 0
        block = params.get("dynagraph_harness") or {}
        assert block.get("profile") == "static_graph"
        assert block.get("memory_summary") is False
        assert block.get("mcq_debias") is False
        assert block.get("explore_when_uncovered") == "off"
        assert block.get("siglip_grounding") is False
        flags = dynagraph_harness_flags(params)
        assert flags["memory_summary"] is False
        assert flags["mcq_debias"] is False
        assert flags["explore_when_uncovered"] == "off"
        assert flags["siglip_grounding"] is False
        fusion = params.get("graph_object_fusion") or {}
        assert float(fusion.get("fallback_spatial_merge_xy_m", -1)) == 0.0


def test_zero_merge_profiles_disable_fusion_fallback():
    from emet.eval.benchmark_dynagraph import apply_dynagraph_profile

    for name in ("smoke", "static_graph", "graph_eqa_baseline"):
        params = apply_dynagraph_profile(get_parameters("dynav_config.yaml"), name)
        assert params.get("dynagraph_merge_xy_m") == 0.0
        fusion = params.get("graph_object_fusion") or {}
        assert float(fusion.get("fallback_spatial_merge_xy_m", -1)) == 0.0, name


def test_unified_eqa_enables_merge_like_interactive():
    from emet.eval.benchmark_dynagraph import apply_dynagraph_profile

    interactive = profile_settings("interactive")
    params = apply_dynagraph_profile(get_parameters("dynav_config.yaml"), "unified_eqa")
    assert params.get("dynagraph_merge_xy_m") == interactive["dynagraph_merge_xy_m"]
    assert params.get("dynagraph_staleness_horizon") == interactive["dynagraph_staleness_horizon"]
    fusion = params.get("graph_object_fusion") or {}
    assert float(fusion.get("fallback_spatial_merge_xy_m")) == float(interactive["dynagraph_merge_xy_m"])
    assert (
        params.get("graph_eqa_extract", {}).get("navigation_samples_max")
        == profile_settings("unified_eqa")["graph_eqa_extract"]["navigation_samples_max"]
    )
