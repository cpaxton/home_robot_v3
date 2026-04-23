# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Dev/test: compare perception-built graph nodes to MuJoCo / Robocasa placement
# metadata (ground truth). Not used in the live agent loop.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphNode


def _norm_label(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def nearest_gt_for_node(
    node: GraphNode,
    placements: Dict[str, Any],
    max_dist_xy: float = 1.2,
) -> Optional[Tuple[str, float, np.ndarray]]:
    """
    Find best-matching placement entry by label substring and xy distance.

    ``placements`` matches robocasa ``object_placements_info``:
    ``body_name -> {cat, pos, quat}`` with ``pos`` shape (3,).
    """
    best: Optional[Tuple[str, float, np.ndarray]] = None
    node_text = " ".join(_norm_label(l) for l in node.labels)
    nxy = np.asarray(node.xyz[:2], dtype=np.float64)

    for body_name, info in placements.items():
        cat = info.get("cat", "")
        pos = np.asarray(info.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64).ravel()
        if pos.size < 3:
            continue
        cat_n = _norm_label(str(cat))
        if cat_n and not any(cat_n in _norm_label(l) for l in node.labels) and cat_n not in node_text:
            # weak match: also try body token
            if _norm_label(body_name) not in node_text:
                continue
        d = float(np.linalg.norm(nxy - pos[:2]))
        if d > max_dist_xy:
            continue
        if best is None or d < best[1]:
            best = (body_name, d, pos[:3].copy())
    return best


def compare_graph_to_placements_report(
    nodes: List[GraphNode],
    object_placements_info: Dict[str, Any],
    max_dist_xy: float = 1.2,
) -> str:
    """Printable report: each graph node vs nearest GT placement (if any)."""
    lines = [
        "Graph vs MuJoCo/Robocasa placements (dev check)",
        "─" * 60,
    ]
    if not object_placements_info:
        lines.append("  (no GT placements dict provided)")
        return "\n".join(lines)

    for n in nodes:
        lbl = ", ".join(n.labels) if n.labels else "?"
        match = nearest_gt_for_node(n, object_placements_info, max_dist_xy=max_dist_xy)
        if match is None:
            lines.append(
                f"  node {n.node_id} [{lbl}]  xyz=({n.xyz[0]:.2f},{n.xyz[1]:.2f},{n.xyz[2]:.2f})"
                f"  ->  NO GT match within {max_dist_xy}m xy"
            )
        else:
            body, d, gpos = match
            lines.append(
                f"  node {n.node_id} [{lbl}]  xyz=({n.xyz[0]:.2f},{n.xyz[1]:.2f},{n.xyz[2]:.2f})"
                f"  ~  GT {body} ({d:.2f}m xy)  pos=({gpos[0]:.2f},{gpos[1]:.2f},{gpos[2]:.2f})"
            )
    lines.append("─" * 60)
    return "\n".join(lines)
