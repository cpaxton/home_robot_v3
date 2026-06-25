# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from __future__ import annotations

from emet.app.stream_agent_factory import format_stream_stats


def test_format_stream_stats_graph_breakdown():
    text = format_stream_stats(
        {
            "n_voxel_observations": 3,
            "n_voxel_explored_cells": 341,
            "n_graph_nodes": 24,
            "n_graph_object_nodes": 11,
            "n_graph_viewpoint_nodes": 12,
            "n_graph_frontier_nodes": 1,
        }
    )
    assert "11 obj" in text
    assert "12 vp" in text
    assert "1 fr" in text
    assert "24 total" in text


def test_format_stream_stats_graph_legacy_total_only():
    text = format_stream_stats(
        {
            "n_voxel_observations": 1,
            "n_voxel_explored_cells": 100,
            "n_graph_nodes": 5,
            "n_graph_object_nodes": 5,
            "n_graph_frontier_nodes": 0,
            "n_graph_viewpoint_nodes": 0,
        }
    )
    assert "5 graph nodes" in text
