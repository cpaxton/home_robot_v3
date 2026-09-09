# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import pytest

from emet.core.parameters import Parameters
from emet.eval.benchmark_dynagraph import enable_query_driven_memory


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
