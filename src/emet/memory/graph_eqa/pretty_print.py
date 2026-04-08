# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Human-readable formatting for GraphEQAMemory / graph blobs (terminal, logs, PRs).

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def format_scene_graph_pretty(
    memory: "GraphEQAMemory",
    *,
    title: str = "Scene graph",
    max_nodes: Optional[int] = None,
) -> str:
    """Multi-line formatted scene graph (nodes, then edges)."""
    lines: List[str] = []
    sep = "─" * 56
    lines.append(sep)
    lines.append(f" {title}")
    lines.append(sep)

    nodes = memory.get_nodes()
    edges = memory.get_edges()
    if max_nodes is not None:
        nodes = nodes[:max_nodes]

    lines.append(f" Nodes ({len(memory.get_nodes())})")
    for n in nodes:
        lbl = ", ".join(n.labels) if n.labels else "(no labels)"
        lines.append(
            f"   [{n.node_id:3d}]  {lbl}"
            f"    xyz=({n.xyz[0]:7.3f}, {n.xyz[1]:7.3f}, {n.xyz[2]:7.3f})  obs={n.obs_id}"
        )
    if max_nodes is not None and len(memory.get_nodes()) > max_nodes:
        lines.append(f"   ... ({len(memory.get_nodes()) - max_nodes} more nodes omitted)")

    lines.append(f" Edges ({len(edges)})")
    for a, b, rel in edges:
        b_str = "floor" if b == -1 else str(b)
        lines.append(f"   {rel}({a}, {b_str})")

    lines.append(sep)
    return "\n".join(lines)


def format_graph_edges_only(
    edges: Sequence[Tuple[int, int, str]], title: str = "Edges"
) -> str:
    lines = [f"{title} ({len(edges)})"]
    for a, b, rel in edges:
        b_str = "floor" if b == -1 else str(b)
        lines.append(f"  {rel}({a}, {b_str})")
    return "\n".join(lines)
