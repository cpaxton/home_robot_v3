# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Typed ``eval:`` block for episode diagnostics (maps, RGB frames, MP4)."""

from __future__ import annotations

from typing import Any

import draccus

from emet.core.parameters import Parameters
from emet.eval.episode_diagnostics import EpisodeDiagnosticsConfig


def load_eval_diagnostics_from_parameters(
    parameters: Parameters | dict[str, Any] | None,
) -> EpisodeDiagnosticsConfig:
    """Read ``eval:`` from a loaded emet config or :class:`Parameters` (``eval`` key)."""
    if parameters is None:
        return EpisodeDiagnosticsConfig()
    subset = parameters.get("eval") if hasattr(parameters, "get") else None
    if subset is None and isinstance(parameters, dict):
        subset = parameters.get("eval")
    if not isinstance(subset, dict):
        return EpisodeDiagnosticsConfig()
    return draccus.decode(EpisodeDiagnosticsConfig, subset)


def resolve_episode_diagnostics_config(
    parameters: Parameters | dict[str, Any] | None = None,
    **cli_overrides: Any,
) -> EpisodeDiagnosticsConfig:
    """Merge YAML ``eval:``, env vars, and explicit CLI/runner overrides.

    Precedence (highest first): ``cli_overrides`` > env (when set) > YAML > defaults.
    """
    from emet.eval.episode_diagnostics import (
        _env_int,
        _env_map_max_side,
        _env_map_min_side,
        _env_map_video_stride,
        _env_truthy_or_none,
    )

    yaml_cfg = load_eval_diagnostics_from_parameters(parameters)

    def _bool(field: str, env_name: str, default: bool) -> bool:
        if field in cli_overrides and cli_overrides[field] is not None:
            return bool(cli_overrides[field])
        env_v = _env_truthy_or_none(env_name)
        if env_v is not None:
            return env_v
        return bool(getattr(yaml_cfg, field, default))

    def _int(field: str, env_name: str, default: int, *, min_val: int | None = None) -> int:
        if field in cli_overrides and cli_overrides[field] is not None:
            val = int(cli_overrides[field])
        elif env_name == "EMET_EVAL_MAP_STRIDE":
            val = _env_int(env_name, int(getattr(yaml_cfg, field, default)))
        elif env_name == "EMET_EVAL_MAP_VIDEO_STRIDE":
            val = _env_map_video_stride(int(getattr(yaml_cfg, field, default)))
        elif env_name == "EMET_EVAL_MAP_MAX_SIDE":
            val = _env_map_max_side(int(getattr(yaml_cfg, field, default)))
        elif env_name == "EMET_EVAL_MAP_MIN_SIDE":
            val = _env_map_min_side(int(getattr(yaml_cfg, field, default)))
        else:
            val = int(getattr(yaml_cfg, field, default))
        if min_val is not None:
            return max(min_val, val)
        return val

    cfg = EpisodeDiagnosticsConfig(
        export_map=_bool("export_map", "EMET_EVAL_EXPORT_MAP", True),
        export_map_stride=_int("export_map_stride", "EMET_EVAL_MAP_STRIDE", 0),
        export_obstacle_grids=_bool("export_obstacle_grids", "EMET_EVAL_EXPORT_OBSTACLE_GRIDS", True),
        export_trajectory=_bool("export_trajectory", "EMET_EVAL_EXPORT_TRAJECTORY", True),
        export_rgb_frames=_bool("export_rgb_frames", "EMET_EVAL_EXPORT_FRAMES", True),
        export_video=_bool("export_video", "EMET_EVAL_EXPORT_VIDEO", True),
        export_object_crops=_bool("export_object_crops", "EMET_EVAL_EXPORT_OBJECT_CROPS", True),
        export_full_graph=_bool("export_full_graph", "EMET_EVAL_EXPORT_GRAPH", False),
        export_voxel_history=_bool("export_voxel_history", "EMET_EVAL_EXPORT_VOXEL_HISTORY", False),
        export_voxel_pickle=_bool("export_voxel_pickle", "EMET_EVAL_EXPORT_VOXEL_PICKLE", False),
        max_map_side=_int("max_map_side", "EMET_EVAL_MAP_MAX_SIDE", 1280, min_val=256),
        min_map_side=_int("min_map_side", "EMET_EVAL_MAP_MIN_SIDE", 1024, min_val=128),
        filter_map_islands=_bool("filter_map_islands", "EMET_EVAL_FILTER_MAP_ISLANDS", True),
        export_gt_navmesh_map=_bool("export_gt_navmesh_map", "EMET_EVAL_EXPORT_GT_MAP", True),
        export_map_overlay=_bool("export_map_overlay", "EMET_EVAL_EXPORT_MAP_OVERLAY", True),
        export_map_video=_bool("export_map_video", "EMET_EVAL_EXPORT_MAP_VIDEO", True),
        map_video_stride=_int("map_video_stride", "EMET_EVAL_MAP_VIDEO_STRIDE", 5, min_val=1),
        video_fps=float(
            cli_overrides["video_fps"]
            if cli_overrides.get("video_fps") is not None
            else getattr(yaml_cfg, "video_fps", 6.0)
        ),
        export_video_substeps=_bool("export_video_substeps", "EMET_EVAL_EXPORT_VIDEO_SUBSTEPS", True),
        video_motion_paced=_bool("video_motion_paced", "EMET_EVAL_VIDEO_MOTION_PACED", True),
        video_meters_per_frame=float(
            cli_overrides["video_meters_per_frame"]
            if cli_overrides.get("video_meters_per_frame") is not None
            else getattr(yaml_cfg, "video_meters_per_frame", 0.25)
        ),
        video_radians_per_frame=float(
            cli_overrides["video_radians_per_frame"]
            if cli_overrides.get("video_radians_per_frame") is not None
            else getattr(yaml_cfg, "video_radians_per_frame", 0.1745329252)
        ),
        video_crossfade_teleport_m=float(
            cli_overrides["video_crossfade_teleport_m"]
            if cli_overrides.get("video_crossfade_teleport_m") is not None
            else getattr(yaml_cfg, "video_crossfade_teleport_m", 1.5)
        ),
    )

    # Habitat runners pass export_voxel_history via helper when CLI unset.
    if cli_overrides.get("export_voxel_history") is not None:
        cfg.export_voxel_history = bool(cli_overrides["export_voxel_history"])

    return cfg
