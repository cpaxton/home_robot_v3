#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Regenerate top-down map exports from an episode diagnostics bundle.

Writes (when inputs allow):
  - topdown_map.png — agent explored map + trajectory (islands pruned)
  - topdown_gt_navmesh.png — Habitat navmesh GT (requires --scene-id or bundle spawn)
  - topdown_map_overlay.png — GT + agent + trajectory composite

Example:
  uv run python scripts/render_episode_map_overlay.py \\
    ~/.cache/habitat_eqa/episodes/cli_episode_q0017/q0017_dynagraph \\
    --scene-id 00033-oPj9qMxrDEa --max-side 1920
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_bundle(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, dict, list[tuple[float, float, float]]]:
    explored = np.load(bundle_dir / "explored_2d.npy")
    obstacles = np.load(bundle_dir / "obstacles_2d.npy")
    meta = json.loads((bundle_dir / "grid_meta.json").read_text(encoding="utf-8"))
    traj: list[tuple[float, float, float]] = []
    traj_path = bundle_dir / "trajectory.jsonl"
    if traj_path.is_file():
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pose = row.get("pose_xyt")
            if pose and len(pose) >= 2:
                theta = float(pose[2]) if len(pose) >= 3 else 0.0
                traj.append((float(pose[0]), float(pose[1]), theta))
    return obstacles, explored, meta, traj


def _robot_xy_from_traj(traj: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    return traj[-1] if traj else None


def _load_pathfinder(scene_id: str):
    try:
        import habitat_sim
    except ImportError as err:
        raise SystemExit("habitat_sim required: use .venv-habitat or install habitat") from err
    from emet.habitat.config import hm3d_scene_navmesh_path

    nav = hm3d_scene_navmesh_path(scene_id)
    if not nav.is_file():
        raise FileNotFoundError(f"navmesh not found for scene {scene_id}: {nav}")
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(str(nav))
    return pf


def _floor_y_from_bundle(bundle_dir: Path) -> float:
    spawn_path = bundle_dir / "spawn_record.json"
    if spawn_path.is_file():
        spawn = json.loads(spawn_path.read_text(encoding="utf-8"))
        snapped = spawn.get("init_pose_snapped") or {}
        if isinstance(snapped, dict) and "y" in snapped:
            return float(snapped["y"])
    return 0.0


def render_bundle(
    bundle_dir: Path,
    *,
    scene_id: str | None,
    max_side: int,
    min_side: int,
    filter_islands: bool,
    write_overlay: bool,
    write_gt: bool,
) -> dict[str, str]:
    bundle_dir = bundle_dir.expanduser().resolve()
    obstacles, explored, meta, traj = _load_bundle(bundle_dir)
    go = np.asarray(meta.get("grid_origin", [0.0, 0.0]), dtype=np.float64).reshape(-1)[:2]
    res = float(meta.get("grid_resolution", 0.1) or 0.1)
    robot_xy = _robot_xy_from_traj(traj)

    from emet.visualization.map_snapshot import eval_topdown_map_rgb, snapshot_eval_overlay_from_voxel_map

    class _VM:
        grid_origin = go
        grid_resolution = res

        def get_2d_map(self):
            return obstacles, explored

    vm = _VM()
    out: dict[str, str] = {}

    agent_rgb = eval_topdown_map_rgb(
        obstacles,
        explored,
        go,
        res,
        robot_xy,
        max_side=max_side,
        min_map_side=min_side,
        trajectory_xyt=traj or None,
        filter_islands=filter_islands,
    )
    map_path = bundle_dir / "topdown_map.png"
    from emet.eval.episode_diagnostics import _save_rgb_png

    _save_rgb_png(map_path, agent_rgb)
    out["topdown_map"] = str(map_path)

    gt_nav = None
    if write_gt or write_overlay:
        if not scene_id:
            raise ValueError("--scene-id required for GT navmesh overlay")
        pf = _load_pathfinder(scene_id)
        floor_y = _floor_y_from_bundle(bundle_dir)
        from emet.habitat.navmesh_topdown import habitat_gt_topdown_cropped, rasterize_habitat_navmesh_grid

        gt_nav, gt_rgb = habitat_gt_topdown_cropped(
            pf,
            obstacles,
            explored,
            go,
            res,
            robot_xy,
            floor_y=floor_y,
            max_side=max_side,
            trajectory_xyt=traj or None,
            filter_islands=filter_islands,
        )
        if write_gt:
            gt_path = bundle_dir / "topdown_gt_navmesh.png"
            _save_rgb_png(gt_path, gt_rgb)
            out["topdown_gt_navmesh"] = str(gt_path)

    if write_overlay:
        if gt_nav is None and scene_id:
            pf = _load_pathfinder(scene_id)
            floor_y = _floor_y_from_bundle(bundle_dir)
            gt_nav = rasterize_habitat_navmesh_grid(pf, explored.shape, go, res, floor_y=floor_y)
        overlay = snapshot_eval_overlay_from_voxel_map(
            vm,
            robot_xy,
            max_side=max_side,
            trajectory_xyt=traj or None,
            gt_navigable=gt_nav,
            filter_islands=filter_islands,
        )
        if overlay is not None:
            overlay_path = bundle_dir / "topdown_map_overlay.png"
            _save_rgb_png(overlay_path, overlay)
            out["topdown_map_overlay"] = str(overlay_path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Episode diagnostics directory")
    parser.add_argument("--scene-id", default="", help="HM3D scene id for navmesh GT")
    parser.add_argument("--max-side", type=int, default=1280, help="Max output width/height in pixels")
    parser.add_argument("--min-side", type=int, default=1024, help="Upscale small crops to at least this size")
    parser.add_argument("--no-filter-islands", action="store_true", help="Keep remote explored islands")
    parser.add_argument("--no-gt", action="store_true", help="Skip topdown_gt_navmesh.png")
    parser.add_argument("--no-overlay", action="store_true", help="Skip topdown_map_overlay.png")
    parser.add_argument(
        "--write-map-video",
        action="store_true",
        help="Encode maps/overlay_step_*.png or maps/step_*.png to topdown_exploration.mp4",
    )
    parser.add_argument("--map-video-fps", type=float, default=6.0, help="FPS for --write-map-video")
    args = parser.parse_args()

    try:
        paths = render_bundle(
            args.bundle_dir,
            scene_id=args.scene_id.strip() or None,
            max_side=max(256, int(args.max_side)),
            min_side=max(128, int(args.min_side)),
            filter_islands=not args.no_filter_islands,
            write_gt=not args.no_gt,
            write_overlay=not args.no_overlay,
        )
    except (FileNotFoundError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    for key, path in paths.items():
        print(f"{key}: {path}")

    if args.write_map_video:
        from emet.eval.episode_diagnostics import _write_map_exploration_mp4

        mp4 = _write_map_exploration_mp4(
            args.bundle_dir.expanduser().resolve(),
            fps=float(args.map_video_fps),
            prefer_overlay=True,
        )
        if mp4:
            print(f"topdown_exploration: {mp4}")
        else:
            print(
                "warning: no map video written (need maps/overlay_step_*.png or maps/step_*.png, ≥2 frames)",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
