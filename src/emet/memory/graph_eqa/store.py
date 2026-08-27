# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Owned scene-graph state for graph memory.

``GraphEQAMemory`` holds a :class:`GraphStore` plus a ``WorldEvidenceStore`` and
delegates mutate / query methods onto the facade. Implementation functions still
read ``self._nodes`` (property aliases onto this store).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from emet.memory.graph_eqa.attempt_ledger import AttemptRecord
from emet.memory.graph_eqa.types import GraphNavigationSample, GraphNode, GraphObservation, RelationBelief


class GraphStore:
    """Nodes, observations, edges, beliefs, frontiers, and the attempt ledger."""

    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self.edges: list[tuple[int, int, str]] = []
        self.observations: list[GraphObservation] = []
        self.next_obs_id: int = 1
        self.room_clusters: list[Any] = []
        self.room_connectivity_fn: Callable[[tuple[float, float], tuple[float, float]], bool] | None = None
        self.relation_beliefs: dict[tuple[int, int, str], RelationBelief] = {}
        self.change_events: list[dict[str, Any]] = []
        self.nav_samples: list[GraphNavigationSample] = []
        self.viewpoint_by_obs_id: dict[int, int] = {}
        self.graph_timestep: int = 0
        self.fallback_timestep: int = 0
        self.spatial_merge_m: float = 0.0
        self.staleness_horizon: int = 0
        self.viewpoint_merge_m: float = 0.15
        self.instance_ingest_stats: dict[str, int] = {
            "proposed": 0,
            "rejected_confidence": 0,
            "rejected_support": 0,
            "admitted": 0,
            "merged": 0,
            "created": 0,
        }
        self.frontier_nodes_enabled: bool = True
        self.frontier_max_nodes: int = 12
        self.frontier_min_cluster_cells: int = 3
        self.frontier_keyword_score_weight: float = 1.0
        self.image_nav_min_approach_m: float = 0.35
        self.retracted_nav_claims: set[tuple[int, str]] = set()
        self.retraction_evidence_views: set[tuple[int, str]] = set()
        self.attempt_records: list[AttemptRecord] = []
        self.attempt_ledger_max: int = 512
        self.attempt_ledger_question_id: str | None = None
        self.persist_absent_claims: bool = False
        self.room_events: list[dict[str, Any]] = []
        self.room_events_max: int = 64
        self.obs_revisions: dict[int, int] = {}
        self.last_obs_content_update_id: int | None = None
        self.obs_siglip_features: dict[int, np.ndarray] = {}
        self.record_navigation: bool = True
        self.nav_max: int = 256
        self.obs_nav_dists: dict[int, list[float]] = {}


# Facade attribute name → GraphStore field. Private aliases keep existing method bodies.
_STORE_ATTRS: dict[str, str] = {
    "_nodes": "nodes",
    "_edges": "edges",
    "_observations": "observations",
    "_next_obs_id": "next_obs_id",
    "_room_clusters": "room_clusters",
    "_room_connectivity_fn": "room_connectivity_fn",
    "_relation_beliefs": "relation_beliefs",
    "_change_events": "change_events",
    "_nav_samples": "nav_samples",
    "_viewpoint_by_obs_id": "viewpoint_by_obs_id",
    "_graph_timestep": "graph_timestep",
    "_fallback_timestep": "fallback_timestep",
    "spatial_merge_m": "spatial_merge_m",
    "staleness_horizon": "staleness_horizon",
    "viewpoint_merge_m": "viewpoint_merge_m",
    "instance_ingest_stats": "instance_ingest_stats",
    "frontier_nodes_enabled": "frontier_nodes_enabled",
    "_frontier_max_nodes": "frontier_max_nodes",
    "_frontier_min_cluster_cells": "frontier_min_cluster_cells",
    "_frontier_keyword_score_weight": "frontier_keyword_score_weight",
    "image_nav_min_approach_m": "image_nav_min_approach_m",
    "_retracted_nav_claims": "retracted_nav_claims",
    "_retraction_evidence_views": "retraction_evidence_views",
    "_attempt_records": "attempt_records",
    "_attempt_ledger_max": "attempt_ledger_max",
    "_attempt_ledger_question_id": "attempt_ledger_question_id",
    "persist_absent_claims": "persist_absent_claims",
    "_room_events": "room_events",
    "_room_events_max": "room_events_max",
    "_obs_revisions": "obs_revisions",
    "_last_obs_content_update_id": "last_obs_content_update_id",
    "_obs_siglip_features": "obs_siglip_features",
    "_record_navigation": "record_navigation",
    "_nav_max": "nav_max",
    "_obs_nav_dists": "obs_nav_dists",
}


def _ensure_store(self: Any) -> GraphStore:
    store = getattr(self, "store", None)
    if store is None:
        store = GraphStore()
        self.store = store
    return store


def _store_property(field: str) -> property:
    def getter(self: Any) -> Any:
        return getattr(_ensure_store(self), field)

    def setter(self: Any, value: Any) -> None:
        setattr(_ensure_store(self), field, value)

    return property(getter, setter)


def attach_store_accessors(cls: type) -> None:
    """Forward graph-owned attributes on ``GraphEQAMemory`` to ``self.store``."""
    for alias, field in _STORE_ATTRS.items():
        setattr(cls, alias, _store_property(field))
