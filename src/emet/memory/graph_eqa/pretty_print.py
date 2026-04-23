# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Human-readable formatting for GraphEQAMemory / graph blobs (terminal, logs, PRs).

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emet.memory.graph_eqa.graph_memory import GraphEQAMemory


def _format_node_labels_line(labels: list[str] | None, *, max_total: int = 120, max_shown: int = 3) -> str:
    """Readable one-line summary: first few labels, cap width, ``(+N more)`` if needed."""
    if not labels:
        return "(no labels)"
    shown = labels[:max_shown]
    rest = len(labels) - len(shown)
    s = ", ".join(shown)
    if rest > 0:
        s = f"{s} (+{rest} more)"
    if len(s) <= max_total:
        return s
    return s[: max_total - 3] + "..."


def format_scene_graph_pretty(
    memory: GraphEQAMemory,
    *,
    title: str = "Scene graph",
    max_nodes: int | None = None,
) -> str:
    """Multi-line formatted scene graph (nodes, then edges)."""
    lines: list[str] = []
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
        lbl = _format_node_labels_line(n.labels)
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

    nav = memory.get_navigation_samples()
    if nav:
        lines.append(f" Navigation samples ({len(nav)}) — camera views without a semantic graph node")
        tail = nav[-4:]
        for nv in tail:
            if nv.base_xyz is not None:
                lines.append(
                    f"   base=({nv.base_xyz[0]:7.3f}, {nv.base_xyz[1]:7.3f}, {nv.base_xyz[2]:7.3f})  "
                    f"anchor=({nv.xyz[0]:7.3f}, {nv.xyz[1]:7.3f}, {nv.xyz[2]:7.3f})"
                )
            else:
                lines.append(
                    f"   anchor=({nv.xyz[0]:7.3f}, {nv.xyz[1]:7.3f}, {nv.xyz[2]:7.3f})"
                )
        if len(nav) > len(tail):
            lines.append(f"   ... ({len(nav) - len(tail)} earlier samples)")

    lines.append(sep)
    return "\n".join(lines)


def format_graph_edges_only(
    edges: Sequence[tuple[int, int, str]], title: str = "Edges"
) -> str:
    lines = [f"{title} ({len(edges)})"]
    for a, b, rel in edges:
        b_str = "floor" if b == -1 else str(b)
        lines.append(f"  {rel}({a}, {b_str})")
    return "\n".join(lines)
