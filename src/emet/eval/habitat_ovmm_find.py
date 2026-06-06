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
from typing import Any

import yaml

from emet.eval.ovmm_find_phase import (
    FindPhaseRunConfig,
    compute_find_phase_metrics,
    distance_to_placement_xy,
    horizontal_coords,
)
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
    return compute_find_phase_metrics(
        obj_pred_xyz=obj_pred_xyz,
        recep_pred_xyz=recep_pred_xyz,
        placements=placements,
        object_query=object_query,
        start_recep=start_recep,
        goal_recep=goal_recep,
        radius_m=radius_m,
        object_gt_body=object_gt_body,
        frame="habitat_xz",
    )


def run_habitat_minival_batch(
    output_dir: str | Path,
    *,
    backend: str = "dynagraph",
    episodes_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Batch runner entry point (delegates to ``emet_habitat`` when available).

    From repo root with Habitat venv::

        .venv-habitat/bin/emet-habitat run-ovmm-find-batch --output-dir runs/ovmm_habitat
    """
    episodes = load_habitat_ovmm_episodes(episodes_path)
    if not episodes:
        return []
    try:
        from emet_habitat.ovmm_find_runner import (
            load_habitat_find_phase_episodes,
            run_habitat_find_phase_episode,
        )
    except ImportError as exc:
        raise RuntimeError("Habitat find-phase batch requires .venv-habitat (./scripts/install_habitat.sh)") from exc

    loaded = load_habitat_find_phase_episodes(episodes_path or resolve_config_yaml_path(DEFAULT_HABITAT_EPISODES))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_cfg = FindPhaseRunConfig(backend=backend)  # type: ignore[arg-type]
    results: list[dict[str, Any]] = []
    for ep in loaded:
        metrics = run_habitat_find_phase_episode(ep, run_cfg)
        (out / f"{ep.id}_{backend}.json").write_text(
            __import__("json").dumps(metrics, indent=2),
            encoding="utf-8",
        )
        results.append(metrics)
    return results


__all__ = [
    "DEFAULT_HABITAT_EPISODES",
    "FindPhaseRunConfig",
    "distance_to_placement_xy",
    "horizontal_coords",
    "load_habitat_ovmm_episodes",
    "run_habitat_minival_batch",
    "score_habitat_find_phase",
]
