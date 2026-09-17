# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.core.parameters import Parameters
from emet.eval.benchmark_dynagraph import enable_query_driven_memory


@pytest.mark.parametrize(
    "preset,mask_backend",
    [("query_detector_segmented_pilot", "yoloe_sam2"), ("query_segmented_support_pilot", "sam2")],
)
def test_habitat_profile_retains_shared_preset(monkeypatch, preset, mask_backend):
    from emet.core.parameters import get_parameters
    from emet.eval.benchmark_dynagraph import apply_habitat_ovmm_find_parameters

    monkeypatch.setenv("EMET_CONFIG", f"configs/emet/{preset}.yaml")
    params = apply_habitat_ovmm_find_parameters(get_parameters("dynav_config.yaml"), "lazy_graph")
    enable_query_driven_memory(params, "lazy_graph")
    assert params.get("query_memory")["mask_backend"] == mask_backend
    assert params.get("query_memory")["surface_presentation"] == "support_only"
    assert params.get("eqa")["vl_hf_model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert params.get("eqa")["vl_quantization"] == "int4"


def test_policy_enables_shared_loop_without_changing_fusion_or_budgets():
    params = Parameters(graph_object_fusion={"enabled": True, "use_instance_nodes": True}, eqa={"max_steps": 7})
    before = dict(params.get("graph_object_fusion"))
    enable_query_driven_memory(params, "lazy_graph")
    assert params.get("query_driven_memory")
    assert params.get("eqa")["agentic_verify"]
    assert params.get("eqa")["max_steps"] == 7
    assert params.get("graph_object_fusion") == before


@pytest.mark.parametrize("backend,fusion", [("dynamem", {"enabled": True}), ("lazy_graph", {"enabled": False})])
def test_incompatible_policy_fails_explicitly(backend, fusion):
    with pytest.raises(ValueError):
        enable_query_driven_memory(Parameters(graph_object_fusion=fusion), backend)


def test_mujoco_find_cli_exposes_query_policy():
    from click.testing import CliRunner

    from emet.app.eval_ovmm import ovmm_group

    result = CliRunner().invoke(ovmm_group, ["find", "--help"])
    assert result.exit_code == 0, result.output
    assert "--query-driven-memory" in result.output


def test_mujoco_query_policy_rejects_wrong_backend_before_sim_start():
    from emet.eval.ovmm_find_phase import FindPhaseRunConfig, run_episode_find_phase

    with pytest.raises(ValueError, match="lazy_graph"):
        run_episode_find_phase(None, FindPhaseRunConfig(backend="dynamem", query_driven_memory=True))
