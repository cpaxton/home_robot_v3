# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Named eval output profiles: what to keep on disk vs what to log.

HM-EQA episode bundles default to dumping every RGB frame, overlay map, and MP4
(``full``). Iterative count/clock debugging does not need that — it fills the
disk and floods job logs that Cursor then tries to ingest.

Profiles (``EMET_EVAL_OUTPUT_PROFILE`` or ``eval.profile``):

- ``full`` — current paper/H2H dumps (frames, videos, overlays, VL token prints).
- ``lean`` — score + traces + one top-down map + trajectory. No per-step RGB, MP4,
  overlay timelapse, object-crop mosaics, evidence-view PNGs, or VL decode spam.
  Keep the 30s VL generate heartbeat so STALE_KILL still sees log mtime.
- ``metrics`` — JSONL / ``eqa_history.json`` / ``metrics.json`` only. No maps.

Per-flag env vars (``EMET_EVAL_EXPORT_*``) still override the profile.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig

PROFILE_FULL = "full"
PROFILE_LEAN = "lean"
PROFILE_METRICS = "metrics"
EVAL_OUTPUT_PROFILES = (PROFILE_FULL, PROFILE_LEAN, PROFILE_METRICS)

# Diagnostic bools that ``lean`` / ``metrics`` turn off. Kept flags are the
# cheap ones we actually use when debugging a miss: one map, trajectory, history.
_LEAN_DIAGNOSTICS: dict[str, bool] = {
    "export_map": True,
    "export_obstacle_grids": False,
    "export_trajectory": True,
    "export_rgb_frames": False,
    "export_video": False,
    "export_object_crops": False,
    "export_full_graph": False,
    "export_compact_memory": False,
    "export_world_evidence_rgb": False,
    "export_voxel_history": False,
    "export_voxel_pickle": False,
    "export_gt_navmesh_map": False,
    "export_map_overlay": False,
    "export_map_video": False,
    "export_video_substeps": False,
    "export_frontier_picks": False,
}

_METRICS_DIAGNOSTICS: dict[str, bool] = {
    **_LEAN_DIAGNOSTICS,
    "export_map": False,
    "export_trajectory": False,
}

_FULL_DIAGNOSTICS: dict[str, bool] = {
    "export_map": True,
    "export_obstacle_grids": True,
    "export_trajectory": True,
    "export_rgb_frames": True,
    "export_video": True,
    "export_object_crops": True,
    "export_full_graph": False,
    "export_compact_memory": False,
    "export_world_evidence_rgb": True,
    "export_voxel_history": True,
    "export_voxel_pickle": False,
    "export_gt_navmesh_map": True,
    "export_map_overlay": True,
    "export_map_video": True,
    "export_video_substeps": True,
    "export_frontier_picks": True,
}

_PROFILE_DIAGNOSTICS: dict[str, dict[str, bool]] = {
    PROFILE_FULL: _FULL_DIAGNOSTICS,
    PROFILE_LEAN: _LEAN_DIAGNOSTICS,
    PROFILE_METRICS: _METRICS_DIAGNOSTICS,
}


def _env_truthy_or_none(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def normalize_eval_output_profile(name: str | None) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return PROFILE_FULL
    if text in ("compact", "score"):
        return PROFILE_LEAN
    if text in ("json", "jsonl", "min", "minimal"):
        return PROFILE_METRICS
    if text not in EVAL_OUTPUT_PROFILES:
        raise ValueError(
            f"Unknown eval output profile {name!r}; expected one of {EVAL_OUTPUT_PROFILES}"
        )
    return text


def resolve_eval_output_profile(
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """``EMET_EVAL_OUTPUT_PROFILE`` beats ``eval.profile`` in YAML."""
    env = os.environ.get("EMET_EVAL_OUTPUT_PROFILE", "").strip()
    if env:
        return normalize_eval_output_profile(env)
    subset = None
    if parameters is not None:
        subset = parameters.get("eval") if hasattr(parameters, "get") else None
        if subset is None and isinstance(parameters, dict):
            subset = parameters.get("eval")
    if isinstance(subset, dict) and subset.get("profile"):
        return normalize_eval_output_profile(str(subset.get("profile")))
    return PROFILE_FULL


def profile_diagnostic_overrides(profile: str) -> dict[str, bool]:
    return dict(_PROFILE_DIAGNOSTICS[normalize_eval_output_profile(profile)])


def eval_log_vl_progress(parameters: Mapping[str, Any] | None = None) -> bool:
    """Per-generate / per-token VL stdout. Heartbeats stay on regardless."""
    env = _env_truthy_or_none("EMET_EVAL_LOG_VL")
    if env is not None:
        return env
    env = _env_truthy_or_none("EMET_VL_DECODE_PROGRESS")
    if env is not None:
        return env
    return resolve_eval_output_profile(parameters) == PROFILE_FULL


def eval_log_eqa_prep(parameters: Mapping[str, Any] | None = None) -> bool:
    """``query_answer: ensure_llm_clients…`` style INFO lines."""
    env = _env_truthy_or_none("EMET_EVAL_LOG_EQA_PREP")
    if env is not None:
        return env
    return resolve_eval_output_profile(parameters) == PROFILE_FULL


def habitat_voxel_history_default() -> bool:
    """Habitat runners historically force voxel history unless the profile is lean."""
    env = _env_truthy_or_none("EMET_EVAL_EXPORT_VOXEL_HISTORY")
    if env is not None:
        return env
    if resolve_eval_output_profile() in (PROFILE_LEAN, PROFILE_METRICS):
        return False
    return True


@dataclass(frozen=True)
class EvalOutputConfig:
    """What an eval episode should write and print.

    ``diagnostics`` is the existing episode-bundle flag set. Log flags are separate
    because they affect stdout/job.log, not ``~/.cache/habitat_eqa/episodes``.
    """

    profile: str = PROFILE_FULL
    diagnostics: EpisodeDiagnosticsConfig | None = None
    log_vl_progress: bool = True
    log_eqa_prep: bool = True

    @classmethod
    def from_env(
        cls,
        parameters: Mapping[str, Any] | None = None,
        **diag_overrides: Any,
    ) -> EvalOutputConfig:
        from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig

        profile = resolve_eval_output_profile(parameters)
        diagnostics = EpisodeDiagnosticsConfig.from_env(parameters, **diag_overrides)
        return cls(
            profile=profile,
            diagnostics=diagnostics,
            log_vl_progress=eval_log_vl_progress(parameters),
            log_eqa_prep=eval_log_eqa_prep(parameters),
        )
