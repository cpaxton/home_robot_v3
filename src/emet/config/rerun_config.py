# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Typed ``rerun:`` block for agent / dynav YAML (see ``agents/default_rerun.yaml``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import yaml

from emet.core.parameters import Parameters
from emet.utils.config import resolve_config_yaml_path
from emet.visualization.null_visualizer import NullVisualizer, visualizer_is_enabled


def _env_truthy(key: str) -> bool | None:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return None
    return raw in ("1", "true", "yes", "on")


def resolve_rerun_bool(
    *,
    cli: bool = False,
    yaml_value: bool | None = None,
    env_key: str | None = None,
    default: bool = False,
) -> bool:
    """Precedence: explicit CLI True, then YAML, then env, then *default*."""
    if cli:
        return True
    if yaml_value is not None:
        return bool(yaml_value)
    if env_key:
        env_v = _env_truthy(env_key)
        if env_v is not None:
            return env_v
    return default


@dataclass
class RerunDynagraphConfig:
    """Dynagraph-heavy Rerun channels (off by default for viewer stability)."""

    log_crops: bool = False
    log_edges: bool = False
    log_summary: bool = False
    log_gallery: bool = False


@dataclass
class RerunAgentConfig:
    """Live Rerun viewer options for ``emet run agent`` / dynamem / dynagraph."""

    headless: bool | None = None
    native_viewer: bool | None = None
    bind_all: bool | None = None
    show_panels: bool | None = None
    debug: bool | None = None
    mjcf_show_visual_mesh: bool = True
    mjcf_show_skeleton: bool = False
    server_memory_limit: str = "4GB"
    display_robot_mesh: bool | None = None
    show_camera_point_clouds: bool = False
    max_displayed_points_per_camera: int = 4096
    max_map_2d_points: int = 25000
    voxel_map_stride: int = 2
    dynagraph_stride: int = 2
    mjcf_mesh_stride: int = 3
    dynagraph: RerunDynagraphConfig = field(default_factory=RerunDynagraphConfig)


def load_rerun_config_from_parameters(parameters: Parameters | dict[str, Any] | None) -> RerunAgentConfig:
    """Read ``rerun:`` from a loaded dynav / agent parameters dict."""
    if parameters is None:
        return RerunAgentConfig()
    subset = parameters.get("rerun") if hasattr(parameters, "get") else None
    if subset is None:
        return RerunAgentConfig()
    if not isinstance(subset, dict):
        return RerunAgentConfig()
    return draccus.decode(RerunAgentConfig, subset)


def load_rerun_agent_overlay(config_path: str | None) -> RerunAgentConfig:
    """Load ``rerun:`` subtree from an agent-style YAML path."""
    if not config_path:
        return RerunAgentConfig()
    full_path = Path(resolve_config_yaml_path(config_path))
    with full_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    subset = raw.get("rerun")
    if subset is None:
        return RerunAgentConfig()
    if not isinstance(subset, dict):
        return RerunAgentConfig()
    return draccus.decode(RerunAgentConfig, subset)


def apply_rerun_environment_from_config(cfg: RerunAgentConfig) -> None:
    """Apply ``bind_all`` from YAML when ``RERUN_BIND_ALL`` is not already set."""
    if cfg.bind_all and _env_truthy("RERUN_BIND_ALL") is None:
        os.environ["RERUN_BIND_ALL"] = "1"


def build_rerun_visualizer_kwargs(
    parameters: Parameters | dict[str, Any] | None = None,
    *,
    output_path: Path | str | None = None,
    display_robot_mesh: bool = True,
    mjcf_robot: tuple[str, tuple[str, ...], int, str] | None = None,
    memory_view: bool = False,
    num_frames: int = 0,
    cli_headless: bool = False,
    cli_native_viewer: bool = False,
    cli_show_panels: bool = False,
) -> dict[str, Any]:
    """Merge agent YAML, env vars, and CLI flags into :class:`RerunVisualizer` kwargs.

    CLI flags only force options **on** (``--headless``, ``--rerun-native``, ``--rerun-show-panels``).
    When a CLI flag is omitted (False), YAML and environment variables apply.
    """
    cfg = load_rerun_config_from_parameters(parameters)
    apply_rerun_environment_from_config(cfg)

    headless = resolve_rerun_bool(
        cli=cli_headless,
        yaml_value=cfg.headless,
        env_key="RERUN_HEADLESS",
        default=False,
    )
    native_viewer = resolve_rerun_bool(
        cli=cli_native_viewer,
        yaml_value=cfg.native_viewer,
        env_key="RERUN_NATIVE_VIEWER",
        default=False,
    )
    show_panels = resolve_rerun_bool(
        cli=cli_show_panels,
        yaml_value=cfg.show_panels,
        default=False,
    )
    mesh = display_robot_mesh if cfg.display_robot_mesh is None else bool(cfg.display_robot_mesh)
    # Dataclass defaults make log_crops / log_edges bools, so resolve_rerun_bool never
    # falls through to env. Operators still need EMET_DYNAGRAPH_RERUN_* =1 to force ON
    # even when YAML is false (cannot force off).
    crops = bool(cfg.dynagraph.log_crops) or (_env_truthy("EMET_DYNAGRAPH_RERUN_CROPS") is True)
    edges = bool(cfg.dynagraph.log_edges) or (_env_truthy("EMET_DYNAGRAPH_RERUN_EDGES") is True)

    return {
        "output_path": Path(output_path) if output_path is not None else None,
        "display_robot_mesh": mesh,
        "mjcf_robot": mjcf_robot,
        "memory_view": memory_view,
        "num_frames": num_frames,
        "headless": headless,
        "rerun_native_viewer": native_viewer,
        "collapse_panels": not show_panels,
        "server_memory_limit": cfg.server_memory_limit or "4GB",
        "mjcf_show_visual_mesh": cfg.mjcf_show_visual_mesh,
        "mjcf_show_skeleton": cfg.mjcf_show_skeleton,
        "dynagraph_rerun_crops": crops,
        "dynagraph_rerun_edges": edges,
        "dynagraph_rerun_summary": cfg.dynagraph.log_summary,
        "dynagraph_rerun_gallery": cfg.dynagraph.log_gallery,
        "max_map_2d_points": max(1000, int(cfg.max_map_2d_points)),
        "show_camera_point_clouds": cfg.show_camera_point_clouds,
        "max_displayed_points_per_camera": int(cfg.max_displayed_points_per_camera),
        "voxel_map_stride": max(1, int(cfg.voxel_map_stride)),
        "dynagraph_stride": max(1, int(cfg.dynagraph_stride)),
        "mjcf_mesh_stride": max(1, int(cfg.mjcf_mesh_stride)),
    }


def open_live_rerun_visualizer(
    parameters: Parameters | dict[str, Any] | None,
    *,
    enabled: bool,
    output_path: Path | str | None = None,
    display_robot_mesh: bool = True,
    mjcf_robot: tuple[str, tuple[str, ...], int, str] | None = None,
    cli_headless: bool = False,
    cli_native_viewer: bool = False,
    cli_show_panels: bool = False,
) -> Any:
    """Construct ``RerunVisualizer`` (or ``NullVisualizer``) from YAML / env / CLI.

    Stretch passes ``display_robot_mesh=True`` (URDF). Generic ZMQ passes MJCF
    when the robot spec has a mesh file.
    """
    if not enabled:
        return NullVisualizer()

    # Deferred: RerunVisualizer imports rerun-sdk native extensions.
    from emet.visualization.rerun import RerunVisualizer

    out_p = Path(output_path) if output_path is not None else None
    if out_p is not None:
        out_p.mkdir(parents=True, exist_ok=True)
    kwargs = build_rerun_visualizer_kwargs(
        parameters,
        output_path=out_p,
        display_robot_mesh=display_robot_mesh,
        mjcf_robot=mjcf_robot,
        cli_headless=cli_headless,
        cli_native_viewer=cli_native_viewer,
        cli_show_panels=cli_show_panels,
    )
    return RerunVisualizer(**kwargs)


def eval_rerun_enabled() -> bool:
    """Live Rerun during OVMM / Habitat / other evals. Default off (GPU-safe).

    Set ``EMET_EVAL_RERUN=1`` (true/yes/on) or pass ``--rerun`` on ``emet ovmm find|full``
    / ``emet-habitat run-episode``. One viewer per box (ports 9090 / 9877).
    """
    return _env_truthy("EMET_EVAL_RERUN") is True


def maybe_attach_eval_rerun_visualizer(
    parameters: Parameters | dict[str, Any] | None,
    existing: Any | None = None,
    *,
    force: bool = False,
) -> Any:
    """Keep an already-live visualizer; otherwise open one when eval Rerun is requested.

    Habitat has no ZMQ ``_rerun``. OVMM / sim eval construct the robot with
    ``enable_rerun_server=eval_rerun_enabled()``. ``force`` is for Habitat ``--rerun``.
    """
    if visualizer_is_enabled(existing):
        return existing
    # Keep MagicMock / other test doubles; only attach over None or NullVisualizer.
    if existing is not None and not isinstance(existing, NullVisualizer):
        return existing
    if not (force or eval_rerun_enabled()):
        return existing if existing is not None else NullVisualizer()
    native = _env_truthy("RERUN_NATIVE_VIEWER") is True
    return open_live_rerun_visualizer(
        parameters,
        enabled=True,
        display_robot_mesh=False,
        cli_headless=not native,
        cli_native_viewer=native,
    )
