# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# Dev/test: compare perception-built graph nodes to MuJoCo / Robocasa placement
# metadata (ground truth). Not used in the live agent loop.

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphNode


def _norm_label(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def nearest_gt_for_node(
    node: GraphNode,
    placements: dict[str, Any],
    max_dist_xy: float = 1.2,
) -> tuple[str, float, np.ndarray] | None:
    """
    Find best-matching placement entry by label substring and xy distance.

    ``placements`` matches robocasa ``object_placements_info``:
    ``body_name -> {cat, pos, quat}`` with ``pos`` shape (3,).
    """
    best: tuple[str, float, np.ndarray] | None = None
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
    nodes: list[GraphNode],
    object_placements_info: dict[str, Any],
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


def _label_matches(node_label: str, gt_label: str) -> bool:
    nl = _norm_label(node_label)
    gl = _norm_label(gt_label)
    if not gl:
        return True
    return gl in nl or nl in gl or gl == nl


def score_nodes_vs_gt(
    nodes: list[GraphNode],
    gt_objects: list[Any],
    *,
    match_xy_m: float = 0.55,
    require_label_match: bool = True,
) -> dict[str, float]:
    """
    Score fused graph object nodes against GT export ``objects[]``.

    Returns gt_recall, node_precision, duplication_penalty, node_count.
    When ``require_label_match`` is false, only XY distance gates a hit (label diagnostic omitted).
    """
    if not gt_objects:
        return {
            "gt_recall": 0.0,
            "node_precision": 0.0,
            "duplication_penalty": 0.0,
            "node_count": float(len(nodes)),
        }

    gt_matched = 0
    dup_penalty = 0.0
    node_hits = 0
    node_exact = 0

    for gt in gt_objects:
        if not isinstance(gt, dict):
            continue
        gpos = np.asarray(gt.get("pos_world", gt.get("pos", [0, 0, 0])), dtype=np.float64).ravel()[:3]
        glabel = str(gt.get("label", ""))
        matches = []
        for n in nodes:
            if require_label_match and not _label_matches(n.labels[0] if n.labels else "", glabel):
                continue
            nxy = np.asarray(n.xyz, dtype=np.float64).reshape(3)
            d = float(np.linalg.norm(nxy[:2] - gpos[:2]))
            if d <= match_xy_m:
                matches.append(n)
        if matches:
            gt_matched += 1
            dup_penalty += max(0, len(matches) - 1)

    for n in nodes:
        nlabel = n.labels[0] if n.labels else ""
        gpos_n = np.asarray(n.xyz, dtype=np.float64).reshape(3)
        gt_hits = []
        for gt in gt_objects:
            if not isinstance(gt, dict):
                continue
            glabel = str(gt.get("label", ""))
            if require_label_match and not _label_matches(nlabel, glabel):
                continue
            gpos = np.asarray(gt.get("pos_world", gt.get("pos", [0, 0, 0])), dtype=np.float64).ravel()[:3]
            if float(np.linalg.norm(gpos_n[:2] - gpos[:2])) <= match_xy_m:
                gt_hits.append(gt)
        if gt_hits:
            node_hits += 1
            if len(gt_hits) == 1:
                node_exact += 1

    n_gt = max(1, len([g for g in gt_objects if isinstance(g, dict)]))
    n_nodes = len(nodes)
    return {
        "gt_recall": float(gt_matched) / float(n_gt),
        "node_precision": float(node_exact) / float(max(1, node_hits)),
        "duplication_penalty": float(dup_penalty),
        "node_count": float(n_nodes),
    }
