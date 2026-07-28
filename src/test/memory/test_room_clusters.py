# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for graph-derived room clusters (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from emet.memory.graph_eqa.room_clusters import (
    cluster_object_nodes,
    estimate_room_at_xy,
    format_rooms_compact,
    merge_room_estimates,
    name_cluster_from_labels,
)


def _node(
    nid: int,
    x: float,
    y: float,
    *,
    labels: list[str] | None = None,
    is_frontier: bool = False,
    is_viewpoint: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=nid,
        xyz=(x, y, 0.5),
        labels=list(labels or ["object"]),
        obs_id=nid,
        is_frontier=is_frontier,
        is_viewpoint=is_viewpoint,
    )


def test_two_xy_blobs_two_clusters():
    nodes = [
        _node(1, 0.0, 0.0, labels=["stove", "kitchen"]),
        _node(2, 0.5, 0.2, labels=["fridge"]),
        _node(3, 10.0, 10.0, labels=["sofa", "living room"]),
        _node(4, 10.4, 9.8, labels=["coffee table"]),
    ]
    clusters = cluster_object_nodes(nodes, link_radius_m=2.0)
    assert len(clusters) == 2
    names = {c.room_name for c in clusters}
    assert "kitchen" in names
    assert "living_room" in names


def test_stamp_room_at_xy_updates_nearest():
    from emet.memory.graph_eqa.room_clusters import stamp_room_at_xy

    nodes = [
        _node(1, 0.0, 0.0, labels=["chair"]),
        _node(2, 0.3, 0.0, labels=["table"]),
        _node(3, 8.0, 0.0, labels=["chair"]),
    ]
    clusters = cluster_object_nodes(nodes, link_radius_m=2.0)
    assert all(c.room_name == "unknown" for c in clusters)
    stamped = stamp_room_at_xy(clusters, (0.1, 0.0), "outdoor")
    near = min(stamped, key=lambda c: (c.centroid_xy[0] - 0.1) ** 2 + c.centroid_xy[1] ** 2)
    assert near.room_name == "outdoor"


def test_near_edge_merges_far_nodes():
    nodes = [
        _node(1, 0.0, 0.0, labels=["chair"]),
        _node(2, 5.0, 0.0, labels=["table"]),
    ]
    # Planar distance 5 m > link radius → separate without edge.
    alone = cluster_object_nodes(nodes, edges=[], link_radius_m=2.0)
    assert len(alone) == 2
    merged = cluster_object_nodes(
        nodes,
        edges=[(1, 2, "near")],
        link_radius_m=2.0,
    )
    assert len(merged) == 1


def test_patio_labels_name_outdoor_cluster():
    assert name_cluster_from_labels(["brick patio", "chair"]) == "patio"
    nodes = [
        _node(1, 0.0, 0.0, labels=["brick patio"]),
        _node(2, 0.4, 0.1, labels=["outdoor chair"]),
    ]
    clusters = cluster_object_nodes(nodes, link_radius_m=2.0)
    assert len(clusters) == 1
    assert clusters[0].room_name == "patio"


def test_object_hints_hypothesize_kitchen():
    from emet.memory.graph_eqa.room_clusters import hypothesize_room_name

    # Explicit room words only — furniture classes do not invent rooms.
    assert hypothesize_room_name(["kitchen island"]) == "kitchen"
    assert hypothesize_room_name(["living room sofa"]) == "living_room"
    assert hypothesize_room_name(["stove", "fridge"]) == "unknown"


def test_lawn_furniture_not_dining_room():
    """Chairs/tables never invent dining; outdoor labels stay outdoor/patio via explicit words."""
    from emet.memory.graph_eqa.room_clusters import hypothesize_room_name, merge_room_estimates

    assert hypothesize_room_name(["chair", "table", "grass"]) == "unknown"
    assert hypothesize_room_name(["chair", "table"]) == "unknown"
    assert hypothesize_room_name(["dining table", "chair"]) == "dining_room"
    assert hypothesize_room_name(["brick patio"]) == "patio"
    assert hypothesize_room_name(["outdoor chair"]) == "outdoor"
    # VLM-first merge: VLM outdoor beats graph dining.
    assert merge_room_estimates("outdoor", "dining_room") == "outdoor"
    assert merge_room_estimates("unknown", "patio") == "patio"


def test_room_mismatch_vs_location_mcq():
    from emet.memory.graph_eqa.room_clusters import (
        question_target_rooms,
        room_mismatches_question,
    )

    q = "\n".join(
        [
            "Where is the wall clock?",
            "A) dining area",
            "B) kitchen",
            "C) sunroom",
            "D) living area near the fireplace",
        ]
    )
    targets = question_target_rooms(q)
    assert "kitchen" in targets
    assert "living_room" in targets or "dining_room" in targets
    assert room_mismatches_question("patio", q) is True
    assert room_mismatches_question("kitchen", q) is False
    assert room_mismatches_question("unknown", q) is False


def test_estimate_room_at_xy_nearest_blob():
    nodes = [
        _node(1, 0.0, 0.0, labels=["kitchen"]),
        _node(2, 0.3, 0.0, labels=["sink"]),
        _node(3, 8.0, 0.0, labels=["patio"]),
        _node(4, 8.2, 0.1, labels=["grill"]),
    ]
    clusters = cluster_object_nodes(nodes, link_radius_m=2.0)
    assert estimate_room_at_xy(clusters, (0.1, 0.0)) == "kitchen"
    assert estimate_room_at_xy(clusters, (8.0, 0.0)) == "patio"
    assert estimate_room_at_xy(clusters, (100.0, 100.0), max_dist_m=3.0) == "unknown"


def test_paint_room_labels_on_export_rgb():
    from emet.memory.graph_eqa.room_clusters import RoomCluster, paint_room_labels

    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:] = (40, 40, 50)
    clusters = [
        RoomCluster(
            cluster_id=1,
            node_ids=(1,),
            labels=("stove",),
            centroid_xy=(1.0, 1.0),
            room_name="kitchen",
        ),
        RoomCluster(
            cluster_id=2,
            node_ids=(2,),
            labels=("sofa",),
            centroid_xy=(3.0, 1.0),
            room_name="living_room",
        ),
    ]
    # Full grid 40x40, crop is full, export is 256x256.
    out = paint_room_labels(
        rgb,
        clusters,
        grid_origin_xy=np.array([0.0, 0.0]),
        grid_resolution=0.1,
        full_shape_hw=(40, 40),
        crop_offset_ij=(0, 0),
        crop_shape_hw=(40, 40),
        font_size=20,
    )
    # Yellow label fill should appear.
    yellow = ((out[:, :, 0] > 230) & (out[:, :, 1] > 200) & (out[:, :, 2] < 120)).sum()
    assert yellow > 50
    assert not np.array_equal(out, rgb)

    nodes = [
        _node(1, 0.0, 0.0, labels=["kitchen"]),
        _node(2, 0.2, 0.0, labels=["stove"]),
    ]
    clusters = cluster_object_nodes(nodes, link_radius_m=2.0)
    line = format_rooms_compact(clusters)
    assert line.startswith("Rooms:")
    assert "kitchen" in line
    assert merge_room_estimates("unknown", "patio") == "patio"
    assert merge_room_estimates("kitchen", "unknown") == "kitchen"
    assert merge_room_estimates("living_room", "kitchen") == "living_room"
    assert merge_room_estimates("outdoor", "dining_room") == "outdoor"


def test_graph_memory_room_at_robot():
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode

    gm = GraphEQAMemory(parameters={}, defer_llm_clients=True)
    gm._nodes = [
        GraphNode(
            node_id=1,
            labels=["kitchen island"],
            xyz=np.array([0.0, 0.0, 0.5]),
            obs_id=1,
        ),
        GraphNode(
            node_id=2,
            labels=["fridge"],
            xyz=np.array([0.4, 0.1, 0.5]),
            obs_id=2,
        ),
        GraphNode(
            node_id=3,
            labels=["brick patio"],
            xyz=np.array([12.0, 0.0, 0.5]),
            obs_id=3,
        ),
    ]
    gm._update_edges()
    assert len(gm._room_clusters) >= 1
    assert gm.graph_room_at_robot((0.1, 0.0)) == "kitchen"
    assert gm.graph_room_at_robot((12.0, 0.0)) == "patio"
    assert "Rooms:" in gm.format_rooms_line()


def test_router_graph_patio_sets_prefer_explore_when_vlm_unknown():
    pytest.importorskip("emet.memory.graph_eqa.agentic_eqa")
    from emet.memory.graph_eqa.agentic_eqa import AgenticEQAExecutor
    from emet.memory.graph_eqa.agentic_tools import build_state_message
    from emet.memory.graph_eqa.graph_memory import NavHypothesis

    agent = MagicMock()
    agent.parameters = {}
    gm = agent.graph_memory
    gm.memory_summary_enabled = False
    gm._nodes = [MagicMock(is_frontier=True, obs_id=99)]
    gm.graph_room_at_robot = MagicMock(return_value="patio")
    gm.format_rooms_line = MagicMock(return_value="Rooms: patio(3)")
    reply = '{"current_room": "unknown", "tool_calls": [{"name": "explore_frontier", "arguments": {}}], "message": ""}'
    gm.eqa_client = MagicMock(return_value=reply)
    agent.robot.get_base_pose.return_value = np.array([0.0, 0.0, 0.0])

    q = "\n".join(
        [
            "Where is the wall clock?",
            "A) dining area",
            "B) kitchen",
            "C) sunroom",
            "D) living area near the fireplace",
        ]
    )
    ex = AgenticEQAExecutor(
        agent,
        q,
        max_rounds=3,
        max_nav_steps=4,
    )
    ex._hypotheses = [
        NavHypothesis(
            phrase="unexplored frontier",
            obs_id=99,
            xyz=np.array([-15.0, 0.0, 0.0]),
            score=0.2,
            source="frontier",
        ),
    ]
    calls, picked_by, meta = ex._route_tool_calls()
    assert picked_by == "vlm"
    assert calls == [("explore_frontier", {})]
    assert meta.get("current_room_graph") == "patio"
    assert meta.get("current_room_vlm") == "unknown"
    assert meta.get("current_room") == "patio"
    assert meta.get("prefer_explore_room_mismatch") is True
    assert ex._prefer_explore is True
    assert ex._prefer_explore_reason == "room_mismatch"
    msg = build_state_message(ex)
    assert "Current room (graph): patio" in msg
    assert "Rooms: patio(3)" in msg
    assert "does not match rooms named" in msg
