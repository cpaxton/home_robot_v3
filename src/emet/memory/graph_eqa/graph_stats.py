# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Graph node counts for eval sweeps and stream diagnostics."""

from __future__ import annotations

from typing import Any


def graph_node_breakdown(graph_memory: Any | None) -> dict[str, int]:
    """Return object / viewpoint / frontier / total node counts."""
    if graph_memory is None or not hasattr(graph_memory, "get_nodes"):
        return {"total": 0, "object": 0, "viewpoint": 0, "frontier": 0}
    nodes = graph_memory.get_nodes()
    frontier = sum(1 for n in nodes if getattr(n, "is_frontier", False))
    viewpoint = sum(1 for n in nodes if getattr(n, "is_viewpoint", False))
    obj = len(nodes) - frontier - viewpoint
    return {
        "total": len(nodes),
        "object": max(0, obj),
        "viewpoint": viewpoint,
        "frontier": frontier,
    }


def format_graph_node_breakdown(graph_memory: Any | None) -> str:
    """Compact status line: ``graph 11 obj / 12 vp / 1 fr (24 total)``."""
    b = graph_node_breakdown(graph_memory)
    return (
        f"graph {b['object']} obj / {b['viewpoint']} vp / {b['frontier']} fr "
        f"({b['total']} total)"
    )
