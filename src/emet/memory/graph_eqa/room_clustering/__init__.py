# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Room partition tool for graph updates.

Default backend is naive ``proximity`` (``near`` + XY radius). Occupancy /
portal backends are config stubs for a later sweep.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from emet.memory.graph_eqa.room_clustering.config import (
    BACKEND_OCCUPANCY_CC,
    BACKEND_PORTAL,
    BACKEND_PROXIMITY,
    DEFAULT_BACKEND,
    DEFAULT_LINK_RADIUS_M,
    IMPLEMENTED_BACKENDS,
    KNOWN_BACKENDS,
    resolve_backend,
)
from emet.memory.graph_eqa.room_clustering.proximity import RoomCluster, cluster_object_nodes

__all__ = [
    "BACKEND_OCCUPANCY_CC",
    "BACKEND_PORTAL",
    "BACKEND_PROXIMITY",
    "DEFAULT_BACKEND",
    "DEFAULT_LINK_RADIUS_M",
    "IMPLEMENTED_BACKENDS",
    "KNOWN_BACKENDS",
    "RoomCluster",
    "cluster_object_nodes",
    "partition",
    "resolve_backend",
]


def partition(
    nodes: Sequence[Any],
    edges: Sequence[tuple[int, int, str]] | None = None,
    *,
    backend: str | None = None,
    link_radius_m: float | None = None,
    connectivity_fn: Any | None = None,
    voxel_map: Any | None = None,
) -> list[RoomCluster]:
    """Assign instance nodes to room clusters. ``voxel_map`` is for future backends."""
    del voxel_map  # occupancy_cc / portal
    chosen = resolve_backend(backend)
    if chosen not in IMPLEMENTED_BACKENDS:
        raise ValueError(
            f"room clustering backend {chosen!r} is not implemented yet; use one of {sorted(IMPLEMENTED_BACKENDS)}"
        )
    radius = DEFAULT_LINK_RADIUS_M if link_radius_m is None else float(link_radius_m)
    return cluster_object_nodes(
        nodes,
        edges,
        link_radius_m=radius,
        connectivity_fn=connectivity_fn,
    )
