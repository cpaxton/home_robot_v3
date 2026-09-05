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
