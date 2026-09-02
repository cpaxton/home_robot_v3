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
Habitat OVMM find-phase adapter (Phase 2).

Episode registry and metric helpers live here (importable from main ``emet``).
Habitat-Sim execution runs in ``.venv-habitat`` via ``emet-habitat`` CLI or
``scripts/eval_habitat_ovmm_find_phases.py``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from emet.eval.ovmm_find_phase import score_ovmm_find_query
from emet.utils.config import resolve_config_yaml_path

DEFAULT_HABITAT_EPISODES = "configs/ovmm/habitat_find_phase_episodes.yaml"


def load_habitat_ovmm_episodes(
    path: str | Path | None = None,
    *,
    split: str = "minival",
) -> list[dict[str, Any]]:
    """
    Load Habitat find-phase episode metadata.

    ``split`` is reserved for future OVMM-HSSD minival JSON; HM3D proxy episodes
    are listed in ``configs/ovmm/habitat_find_phase_episodes.yaml``.
    """
    _ = split
    full = Path(resolve_config_yaml_path(str(path or DEFAULT_HABITAT_EPISODES)))
    with full.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    rows = raw.get("episodes") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"expected list under 'episodes' in {full}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def score_habitat_find_phase(
    *,
    obj_pred_xyz,
    recep_pred_xyz,
    placements: dict[str, dict[str, Any]] | None,
    object_query: str,
    start_recep: str,
    goal_recep: str,
    radius_m: float = 0.75,
    object_gt_body: str | None = None,
) -> dict[str, Any]:
    """Score one Habitat find-phase step (XZ horizontal plane, bounds-aware)."""
    return score_ovmm_find_query(
        SimpleNamespace(obj_xyz=obj_pred_xyz, recep_xyz=recep_pred_xyz),
        placements=placements,
        object_query=object_query,
        start_recep=start_recep,
        goal_recep=goal_recep,
        radius_m=radius_m,
        object_gt_body=object_gt_body,
        frame="habitat_xz",
    )


__all__ = [
    "DEFAULT_HABITAT_EPISODES",
    "load_habitat_ovmm_episodes",
    "score_habitat_find_phase",
]
