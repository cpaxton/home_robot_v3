# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Frontier clustering and question-guided exploration helpers for GraphEQAMemory."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.ndimage import label

FRONTIER_DESC_PREFIX = "frontier:"


def exploration_keywords_from_text(
    text: str | None,
    *,
    extra: list[str] | None = None,
) -> list[str]:
    """Tokenize an EQA question / enrich hints into frontier-scoring keywords."""
    from emet.memory.graph_eqa.graph_memory import heuristic_relevant_objects

    merged: list[str] = []
    for token in list(extra or []) + heuristic_relevant_objects(text or ""):
        key = token.strip().lower()
        if key and key not in merged:
            merged.append(key)
    if text:
        try:
            from emet.habitat.hmeqa_enrich_labels import parse_enrich_label_text

            for token in parse_enrich_label_text(text):
                if token not in merged:
                    merged.append(token)
        except ImportError:
            pass
    return merged[:8]


def _as_bool_numpy(mask: Any) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool)
    return np.asarray(mask, dtype=bool)


def reachable_waypoint_for_cluster(
    cells: np.ndarray,
    reachable: np.ndarray,
) -> tuple[int, int] | None:
    """Pick a navigable cell adjacent to a frontier cluster (not the arc centroid).

    Frontier clusters are rings/arcs around reachable space. Their mean cell often
    lands in already-explored floor (or inside obstacles). Navigation must target a
    cell that is **reachable** and **8-adjacent** to at least one frontier cell so
    the robot actually approaches new coverage.
    """
    if cells.size == 0:
        return None
    reachable_b = np.asarray(reachable, dtype=bool)
    h, w = reachable_b.shape
    adjacent: set[tuple[int, int]] = set()
    for r, c in cells:
        ri, ci = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = ri + dr, ci + dc
                if 0 <= nr < h and 0 <= nc < w and reachable_b[nr, nc]:
                    adjacent.add((nr, nc))
    if not adjacent:
        return None
    cy = float(cells[:, 0].mean())
    cx = float(cells[:, 1].mean())
    return min(adjacent, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2)


def cluster_frontier_mask(
    unexplored: np.ndarray,
    *,
    min_cells: int = 3,
    reachable: np.ndarray | None = None,
) -> list[tuple[str, tuple[int, int], int]]:
    """
    Connected components on an unexplored-frontier bool grid.

    When ``reachable`` is provided, each cluster's goal cell is the nearest
    reachable neighbor of the cluster (see :func:`reachable_waypoint_for_cluster`),
    not the geometric centroid — centroids of frontier arcs sit mid-floor.

    Returns:
        List of ``(cluster_id, (row, col) goal grid cell, cell_count)``.
    """
    if not unexplored.any():
        return []
    labeled, n_comp = label(unexplored)
    reachable_b = _as_bool_numpy(reachable) if reachable is not None else None
    out: list[tuple[str, tuple[int, int], int]] = []
    for cid in range(1, int(n_comp) + 1):
        cells = np.argwhere(labeled == cid)
        if cells.shape[0] < min_cells:
            continue
        centroid = tuple(int(round(float(cells[:, d].mean()))) for d in range(2))
        goal = centroid
        if reachable_b is not None:
            snapped = reachable_waypoint_for_cluster(cells, reachable_b)
            if snapped is not None:
                goal = snapped
        out.append((f"c{cid}", goal, int(cells.shape[0])))
    return out


def hint_labels_near_grid(
    grid_ij: tuple[int, int],
    image_descriptions: list[tuple[list[str], Any]] | None,
    *,
    max_dist_cells: int = 12,
    max_labels: int = 4,
) -> list[str]:
    """Labels from the nearest voxel ``image_descriptions`` cluster to a frontier cell."""
    if not image_descriptions:
        return []
    gi, gj = grid_ij
    best: tuple[float, list[str]] | None = None
    for cluster, coord in image_descriptions:
        if coord is None:
            continue
        c = np.asarray(coord, dtype=float).reshape(-1)
        if c.size < 2:
            continue
        dist = float(np.hypot(float(c[0]) - gi, float(c[1]) - gj))
        if dist > max_dist_cells:
            continue
        labels = [str(x).strip().lower() for x in cluster if str(x).strip()]
        if not labels:
            continue
        if best is None or dist < best[0]:
            best = (dist, labels)
    if best is None:
        return []
    return best[1][:max_labels]


def keyword_overlap_score(labels: list[str], keywords: list[str]) -> float:
    if not labels or not keywords:
        return 0.0
    blob = " ".join(labels).lower()
    hits = sum(1 for kw in keywords if kw.lower() in blob)
    return float(hits) / float(len(keywords))


def keyword_score_map(
    frontier_mask: np.ndarray,
    image_descriptions: list[tuple[list[str], Any]] | None,
    keywords: list[str],
    *,
    radius_cells: int = 10,
) -> np.ndarray:
    """Per-cell keyword affinity on the frontier (0 outside frontier)."""
    scores = np.zeros(frontier_mask.shape, dtype=np.float32)
    if not keywords or not frontier_mask.any():
        return scores
    for i, j in zip(*np.where(frontier_mask), strict=False):
        hints = hint_labels_near_grid(
            (int(i), int(j)),
            image_descriptions,
            max_dist_cells=radius_cells,
        )
        scores[i, j] = keyword_overlap_score(hints, keywords)
    return scores
