# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""
Habitat OVMM find-phase adapter (Phase 2, optional).

Reuses :mod:`emet.eval.ovmm_find_phase` metric definitions on a small Habitat-OVMM
minival split. Habitat-Sim runs in ``.venv-habitat`` (see ``docs/habitat/install.md``);
this module stays importable from the main ``emet`` package without ``habitat_sim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emet.eval.ovmm_find_phase import compute_find_phase_metrics


def load_habitat_ovmm_episodes(split: str = "minival") -> list[dict[str, Any]]:
    """
    Load Habitat OVMM episode metadata for find-phase evaluation.

    Returns empty list until OVMM minival JSON is wired (Habitat EQA harness is
    separate: ``emet run graph-eqa-habitat`` / ``docs/habitat_eqa.md``).
    """
    _ = split
    return []


def score_habitat_find_phase(
    *,
    obj_pred_xyz,
    recep_pred_xyz,
    object_gt_xyz,
    recep_gt_xyz_list: list,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    radius_m: float = 0.75,
) -> dict[str, Any]:
    """
    Score one Habitat find-phase step using the same oracle as emet sim.

    ``object_gt_xyz`` / ``recep_gt_xyz_list`` should be world-frame positions from Habitat GT.
    """
    placements: dict[str, dict[str, Any]] = {}
    if object_gt_xyz is not None:
        placements["habitat_object"] = {"cat": object_query, "pos": list(object_gt_xyz)}
    for i, xyz in enumerate(recep_gt_xyz_list or []):
        placements[f"habitat_recep_{i}"] = {"cat": goal_recep, "pos": list(xyz)}
    return compute_find_phase_metrics(
        obj_pred_xyz=obj_pred_xyz,
        recep_pred_xyz=recep_pred_xyz,
        placements=placements or None,
        object_query=object_query,
        start_recep=start_recep,
        goal_recep=goal_recep,
        radius_m=radius_m,
    )


def run_habitat_minival_batch(
    output_dir: str | Path,
    *,
    backend: str = "dynagraph",
) -> list[dict[str, Any]]:
    """
    Batch runner entry point for Habitat minival (no-op when episodes unavailable).

    Full implementation will mirror ``scripts/eval_ovmm_find_phases.py`` with a Habitat env bridge.
    """
    episodes = load_habitat_ovmm_episodes("minival")
    if not episodes:
        return []
    _ = backend
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return []
