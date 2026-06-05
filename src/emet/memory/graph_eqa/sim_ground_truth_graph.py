# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Build GraphEQA / Dynagraph scene graphs from sim ground-truth placements (no VLM)."""

from __future__ import annotations

from typing import Any

import numpy as np

from emet.memory.graph_eqa.graph_memory import GraphEQAMemory
from emet.memory.graph_eqa.mujoco_align import compare_graph_to_placements_report

_EMET_INTERNAL_KEYS = frozenset({"_emet_spawn_hint_xyt"})


def read_sim_object_placements(session: dict[str, Any] | None) -> dict[str, dict[str, Any]] | None:
    """Return ``sim_object_placements`` from ``robot.get_emet_session()``, or ``None``."""
    if not session:
        return None
    raw = session.get("sim_object_placements")
    if not isinstance(raw, dict) or not raw:
        return None
    out: dict[str, dict[str, Any]] = {}
    for body_name, info in raw.items():
        if body_name in _EMET_INTERNAL_KEYS or not isinstance(info, dict):
            continue
        pos = info.get("pos")
        if not pos:
            continue
        out[str(body_name)] = {
            "cat": str(info.get("cat") or body_name),
            "pos": np.asarray(pos, dtype=np.float64).reshape(3),
            "quat": np.asarray(info.get("quat") or [1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(4),
        }
    return out or None


def populate_graph_memory_from_placements(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    placements: dict[str, dict[str, Any]],
    *,
    max_objects: int = 48,
) -> int:
    """
    Add one graph node per GT placement (labels from ``cat``, xyz from ``pos``).

    Returns the number of nodes added.
    """
    added = 0
    for body_name, info in placements.items():
        if body_name in _EMET_INTERNAL_KEYS:
            continue
        if added >= max_objects:
            break
        cat = str(info.get("cat") or body_name).strip()
        pos = np.asarray(info.get("pos"), dtype=np.float64).reshape(-1)
        if pos.size < 3 or not cat:
            continue
        graph_memory.add_observation(
            rgb,
            pos[:3],
            [cat],
            description=f"ground_truth:{body_name}",
        )
        added += 1
    return added


def build_ground_truth_graph_from_session(
    graph_memory: GraphEQAMemory,
    rgb: np.ndarray,
    session: dict[str, Any] | None,
) -> tuple[int, dict[str, dict[str, Any]] | None]:
    """Populate ``graph_memory`` from session GT; return (nodes_added, placements_dict)."""
    placements = read_sim_object_placements(session)
    if not placements:
        return 0, None
    n = populate_graph_memory_from_placements(graph_memory, rgb, placements)
    return n, placements


def ground_truth_alignment_report(
    graph_memory: GraphEQAMemory,
    placements: dict[str, dict[str, Any]] | None,
    *,
    max_dist_xy: float = 1.2,
) -> str:
    """Printable dev report: built graph vs GT placements (identity check in GT mode)."""
    if not placements:
        return "Graph vs GT: (no sim_object_placements in emet_session)"
    return compare_graph_to_placements_report(
        graph_memory.get_nodes(),
        placements,
        max_dist_xy=max_dist_xy,
    )
