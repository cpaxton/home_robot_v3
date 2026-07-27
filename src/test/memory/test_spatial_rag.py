# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for spatial RAG prompt retrieval (no GPU)."""

from __future__ import annotations

from types import SimpleNamespace

from emet.memory.graph_eqa.spatial_rag import (
    expand_neighbors,
    format_regions_for_prompt,
    select_seed_nodes,
    select_spatial_regions,
)


def _node(
    nid: int,
    x: float,
    y: float,
    *,
    labels: list[str] | None = None,
    obs_id: int | None = None,
    is_frontier: bool = False,
    support: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=nid,
        xyz=(x, y, 0.0),
        labels=list(labels or ["object"]),
        obs_id=obs_id if obs_id is not None else nid,
        is_frontier=is_frontier,
        is_viewpoint=False,
        support_count=support,
        frontier_cell_count=0,
    )


def test_seeds_prefer_keyword_and_obs():
    nodes = [
        _node(1, 0, 0, labels=["cabinet"]),
        _node(2, 1, 0, labels=["red pillow", "sofa"], obs_id=20),
        _node(3, 10, 10, labels=["toilet"]),
    ]
    seeds = select_seed_nodes(nodes, keywords=["pillow"], prefer_obs_ids=[20])
    assert [n.node_id for n in seeds][0] == 2


def test_expand_keeps_spatial_neighbors_drops_far():
    nodes = [
        _node(1, 0, 0, labels=["sofa"]),
        _node(2, 1.0, 0, labels=["pillow"]),
        _node(3, 20, 0, labels=["car"]),
    ]
    seeds = [nodes[0]]
    expanded = expand_neighbors(nodes, seeds, radius_m=2.5)
    ids = {n.node_id for n in expanded}
    assert 1 in ids and 2 in ids
    assert 3 not in ids


def test_select_spatial_regions_clusters_adjacent():
    nodes = [
        _node(1, 0, 0, labels=["sofa"], obs_id=10),
        _node(2, 0.5, 0, labels=["red pillow"], obs_id=10),
        _node(3, 0.8, 0.2, labels=["lamp"], obs_id=11),
        _node(4, 15, 0, labels=["stove"], obs_id=40),
        _node(5, 15.2, 0.1, labels=["fridge"], obs_id=41),
        _node(99, 0, 0, labels=["frontier yard"], is_frontier=True),
    ]
    result = select_spatial_regions(
        nodes,
        keywords=["pillow", "sofa"],
        prefer_obs_ids=[10],
        radius_m=2.5,
        max_regions=6,
    )
    assert result.regions
    text = format_regions_for_prompt(result)
    assert "REGION 1" in text
    assert "sofa" in text.lower() or "pillow" in text.lower()
    assert "anchor" in text
    # Far kitchen should be a separate region or dropped if not seeded; sofa neighborhood present
    kept_labels = " ".join(lab for r in result.regions for lab in r.labels).lower()
    assert "sofa" in kept_labels or "pillow" in kept_labels
    assert result.frontier_nodes


def test_distant_keyword_nodes_can_form_second_region():
    nodes = [
        _node(1, 0, 0, labels=["sofa", "pillow"]),
        _node(2, 0.4, 0, labels=["side table"]),
        _node(3, 12, 0, labels=["fan"]),
        _node(4, 12.3, 0.2, labels=["bed"]),
    ]
    result = select_spatial_regions(
        nodes,
        keywords=["pillow", "fan"],
        radius_m=2.5,
        max_regions=6,
    )
    assert len(result.regions) >= 2
    text = format_regions_for_prompt(result)
    assert "fan" in text.lower() or "bed" in text.lower()


def test_graph_memory_to_string_spatial_rag_regions(monkeypatch):
    """Exercise to_string REGION path without constructing a full GraphEQAMemory (no VLM)."""
    import numpy as np

    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory, GraphNode

    # Bypass heavy __init__ side effects (VL client / config resolution).
    mem = object.__new__(GraphEQAMemory)
    mem.parameters = {"eqa": {"spatial_rag": True, "spatial_rag_radius_m": 2.5}}
    mem._nodes = [
        GraphNode(
            node_id=1,
            labels=["sofa", "pillow"],
            xyz=np.array([0.0, 0.0, 0.0]),
            obs_id=10,
        ),
        GraphNode(
            node_id=2,
            labels=["lamp"],
            xyz=np.array([0.5, 0.1, 0.0]),
            obs_id=11,
        ),
        GraphNode(
            node_id=3,
            labels=["distant car"],
            xyz=np.array([30.0, 0.0, 0.0]),
            obs_id=99,
        ),
    ]
    mem._edges = []
    mem._relevant_objects = ["pillow"]
    mem.last_eqa_obs_ids = [10]
    mem.last_eqa_prompt_node_count = 0
    mem.last_eqa_prompt_regions = 0
    mem.last_eqa_spatial_rag = None

    text = mem.to_string(
        max_object_nodes=48,
        question_keywords=["pillow"],
        prefer_obs_ids=[10],
        record_prompt_count=True,
    )
    assert "REGION" in text
    assert "sofa" in text.lower() or "pillow" in text.lower()
    assert mem.last_eqa_prompt_regions >= 1
    assert mem.last_eqa_spatial_rag is not None
