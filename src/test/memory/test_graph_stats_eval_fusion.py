# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from types import SimpleNamespace

from emet.eval.benchmark_dynagraph import apply_eval_graph_fusion_parameters
from emet.memory.graph_eqa.graph_stats import format_graph_node_breakdown, graph_node_breakdown


class _FakeNode:
    def __init__(self, *, viewpoint=False, frontier=False):
        self.is_viewpoint = viewpoint
        self.is_frontier = frontier


def test_graph_node_breakdown_counts():
    gm = SimpleNamespace(
        get_nodes=lambda: [
            _FakeNode(),
            _FakeNode(viewpoint=True),
            _FakeNode(frontier=True),
        ]
    )
    b = graph_node_breakdown(gm)
    assert b == {"total": 3, "object": 1, "viewpoint": 1, "frontier": 1}
    assert "1 obj" in format_graph_node_breakdown(gm)


def test_apply_eval_graph_fusion_parameters_enables_fallback():
    from emet.core.parameters import Parameters

    params = Parameters()
    params["dynagraph_merge_xy_m"] = 0.45
    out = apply_eval_graph_fusion_parameters(params)
    fusion = out.get("graph_object_fusion")
    assert fusion["enabled"] is True
    assert fusion["fallback_spatial_merge_xy_m"] == 0.45


def test_apply_eval_graph_fusion_parameters_zeros_fallback_when_merge_disabled():
    from emet.core.parameters import Parameters

    params = Parameters()
    params["dynagraph_merge_xy_m"] = 0.0
    out = apply_eval_graph_fusion_parameters(params, merge_xy_m=0.0)
    fusion = out.get("graph_object_fusion")
    assert fusion["enabled"] is True
    assert fusion["fallback_spatial_merge_xy_m"] == 0.0
