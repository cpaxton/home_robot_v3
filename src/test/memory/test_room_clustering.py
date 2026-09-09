# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""CPU tests for ``room_clustering.partition`` (proximity only)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from emet.memory.graph_eqa.room_clustering import partition
from emet.memory.graph_eqa.room_clusters import cluster_object_nodes


def _node(nid: int, x: float, y: float, *, labels: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=nid,
        xyz=(x, y, 0.1),
        labels=list(labels or []),
        is_frontier=False,
        is_viewpoint=False,
    )


def test_partition_proximity_matches_cluster_object_nodes():
    nodes = [
        _node(1, 0.0, 0.0, labels=["kitchen island"]),
        _node(2, 0.5, 0.1, labels=["sink"]),
        _node(3, 10.0, 10.0, labels=["bed"]),
    ]
    via_partition = partition(nodes, backend="proximity", link_radius_m=2.0)
    via_cluster = cluster_object_nodes(nodes, link_radius_m=2.0)
    assert [(c.node_ids, c.room_name, c.centroid_xy) for c in via_partition] == [
        (c.node_ids, c.room_name, c.centroid_xy) for c in via_cluster
    ]
    assert len(via_partition) == 2


def test_partition_unknown_backend_errors():
    with pytest.raises(ValueError, match="unknown room clustering backend"):
        partition([], backend="not_a_backend")


def test_partition_unimplemented_backend_errors():
    with pytest.raises(ValueError, match="not implemented"):
        partition([], backend="occupancy_cc")
    with pytest.raises(ValueError, match="not implemented"):
        partition([], backend="portal")


def test_ovmm_find_phase_does_not_pin_episode_phrases():
    from emet.eval import ovmm_find_phase

    src = inspect.getsource(ovmm_find_phase)
    assert "pin_phrases_after_mapping" not in src
