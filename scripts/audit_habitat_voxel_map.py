#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Analyze Habitat episode voxel bundles (observations history + final 2D grids)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "header":
            header = row
        elif row.get("type") == "observation":
            rows.append(row)
    if header is None:
        raise ValueError(f"no header row in {path}")
    return header, rows


def _connected_components(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, np.ndarray]] = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or seen[i, j]:
                continue
            q: deque[tuple[int, int]] = deque([(i, j)])
            seen[i, j] = True
            cells: list[tuple[int, int]] = []
            while q:
                ci, cj = q.popleft()
                cells.append((ci, cj))
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < h and 0 <= nj < w and mask[ni, nj] and not seen[ni, nj]:
                        seen[ni, nj] = True
                        q.append((ni, nj))
            components.append((len(cells), np.asarray(cells, dtype=np.int32)))
    components.sort(key=lambda x: x[0], reverse=True)
    return components


def _grid_center(cells: np.ndarray) -> tuple[float, float]:
    return float(cells[:, 0].mean()), float(cells[:, 1].mean())


def _nearest_obs_to_ij(
    obs_rows: list[dict[str, Any]],
    target_ij: tuple[float, float],
    *,
    field: str,
) -> tuple[int | None, float]:
    best_idx: int | None = None
    best_dist = float("inf")
    ti, tj = target_ij
    for row in obs_rows:
        ij = row.get(field)
        if not ij or len(ij) < 2:
            continue
        di = float(ij[0]) - ti
        dj = float(ij[1]) - tj
        dist = (di * di + dj * dj) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_idx = int(row["obs_idx"])
    return best_idx, best_dist


def audit_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    report: dict[str, Any] = {"bundle_dir": str(bundle_dir)}

    hist_path = bundle_dir / "observations_history.jsonl"
    if hist_path.is_file():
        header, obs_rows = _load_jsonl(hist_path)
        report["n_observations"] = header.get("n_observations", len(obs_rows))
        report["grid_resolution"] = header.get("grid_resolution")
        report["grid_origin_xy"] = header.get("grid_origin_xy")
        report["spawn_record"] = header.get("spawn_record")
        if (bundle_dir / "spawn_record.json").is_file() and not report["spawn_record"]:
            report["spawn_record"] = json.loads((bundle_dir / "spawn_record.json").read_text())

        mismatches = []
        for row in obs_rows:
            delta = row.get("gps_camera_grid_delta_ij")
            if delta and (abs(delta[0]) > 0 or abs(delta[1]) > 0):
                mismatches.append(
                    {
                        "obs_idx": row["obs_idx"],
                        "gps_grid_ij": row.get("gps_grid_ij"),
                        "camera_grid_ij": row.get("camera_grid_ij"),
                        "camera_grid_ij_xz": row.get("camera_grid_ij_xz"),
                        "delta_ij": delta,
                    }
                )
        report["gps_camera_grid_mismatches"] = mismatches

        pcd_offsets = []
        pcd_x_mismatches = []
        for row in obs_rows:
            base = row.get("base_pose_xyt")
            centroid = row.get("pcd_centroid_xz")
            if not base or not centroid:
                continue
            dx = float(centroid[0]) - float(base[0])
            dz = float(centroid[1]) - float(base[1])
            dist = (dx * dx + dz * dz) ** 0.5
            if dist > 0.5:
                pcd_offsets.append(
                    {
                        "obs_idx": row["obs_idx"],
                        "base_pose_xz": [base[0], base[1]],
                        "pcd_centroid_xz": centroid,
                        "offset_m": dist,
                        "offset_x_m": abs(dx),
                        "offset_z_m": abs(dz),
                    }
                )
            if abs(dx) > 0.5 or (
                float(centroid[0]) * float(base[0]) < 0
                and abs(float(base[0])) > 0.05
                and abs(float(centroid[0])) > 0.05
            ):
                pcd_x_mismatches.append(
                    {
                        "obs_idx": row["obs_idx"],
                        "base_x": float(base[0]),
                        "pcd_x": float(centroid[0]),
                        "offset_x_m": abs(dx),
                    }
                )
        report["large_pcd_centroid_offsets_m"] = pcd_offsets
        report["pcd_planar_x_mismatches"] = pcd_x_mismatches
    else:
        report["warning"] = "observations_history.jsonl missing; final-grid summary only"
        obs_rows = []

    explored_path = bundle_dir / "explored_2d.npy"
    obstacles_path = bundle_dir / "obstacles_2d.npy"
    if explored_path.is_file():
        explored = np.load(explored_path)
        obstacles = np.load(obstacles_path) if obstacles_path.is_file() else np.zeros_like(explored)
        n_exp = int(explored.sum())
        n_obs = int((explored & obstacles).sum())
        n_free = int((explored & ~obstacles).sum())
        report["explored_cells"] = n_exp
        report["explored_obstacle_frac"] = (n_obs / n_exp) if n_exp else None
        report["explored_free_frac"] = (n_free / n_exp) if n_exp else None

        components = _connected_components(explored)
        comp_summaries = []
        for count, cells in components[:5]:
            ci, cj = _grid_center(cells)
            nearest_gps, dist_gps = _nearest_obs_to_ij(obs_rows, (ci, cj), field="gps_grid_ij")
            nearest_cam, dist_cam = _nearest_obs_to_ij(obs_rows, (ci, cj), field="camera_grid_ij")
            comp_summaries.append(
                {
                    "cell_count": count,
                    "center_ij": [ci, cj],
                    "nearest_obs_gps_grid": nearest_gps,
                    "nearest_obs_gps_dist_cells": dist_gps,
                    "nearest_obs_camera_grid": nearest_cam,
                    "nearest_obs_camera_dist_cells": dist_cam,
                }
            )
        report["explored_components"] = comp_summaries
        if len(components) > 1 and obs_rows:
            satellite = components[1]
            sat_cells = satellite[1]
            sci, scj = _grid_center(sat_cells)
            best_idx, best_dist = _nearest_obs_to_ij(obs_rows, (sci, scj), field="camera_grid_ij")
            report["satellite_blob_hypothesis"] = {
                "cell_count": satellite[0],
                "center_ij": [sci, scj],
                "best_matching_obs_idx": best_idx,
                "best_matching_dist_cells": best_dist,
            }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Episode debug bundle directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report = audit_bundle(args.bundle_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"bundle: {report['bundle_dir']}")
        if report.get("warning"):
            print(f"warning: {report['warning']}")
        if "n_observations" in report:
            print(f"observations: {report['n_observations']}")
            print(f"gps/camera grid mismatches: {len(report.get('gps_camera_grid_mismatches', []))}")
            for mm in report.get("gps_camera_grid_mismatches", [])[:5]:
                print(
                    f"  obs {mm['obs_idx']}: gps={mm['gps_grid_ij']} cam={mm['camera_grid_ij']} delta={mm['delta_ij']}"
                )
        if "explored_cells" in report:
            print(
                f"explored cells: {report['explored_cells']} "
                f"(obstacle_frac={report.get('explored_obstacle_frac')}, free_frac={report.get('explored_free_frac')})"
            )
            for comp in report.get("explored_components", []):
                print(
                    f"  component {comp['cell_count']} cells center={comp['center_ij']} "
                    f"nearest_cam_obs={comp['nearest_obs_camera_grid']}"
                )
        if report.get("satellite_blob_hypothesis"):
            sat = report["satellite_blob_hypothesis"]
            print(
                f"satellite blob: {sat['cell_count']} cells at {sat['center_ij']} "
                f"best obs_idx={sat['best_matching_obs_idx']} (dist={sat['best_matching_dist_cells']:.1f} cells)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
