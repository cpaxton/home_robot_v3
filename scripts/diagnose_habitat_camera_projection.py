#!/usr/bin/env python3
# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Diagnose Habitat depth unprojection / obstacle-map alignment (bundle or live spin)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_obs_history(bundle_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = bundle_dir / "observations_history.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
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
        raise ValueError(f"no header in {path}")
    return header, rows


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return unit normal and mean residual for Nx3 points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0]), float("nan")
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    if normal[2] < 0:
        normal = -normal
    residual = float(np.median(np.abs((pts - centroid) @ normal)))
    return normal, residual


def _unproject_obs(rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    import torch

    from emet.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates

    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = d.shape
    depth_t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0)
    pose_t = torch.from_numpy(np.asarray(camera_pose, dtype=np.float64)).unsqueeze(0).float()
    k = np.asarray(intrinsics, dtype=np.float64)[:3, :3]
    inv_k = torch.linalg.inv(torch.from_numpy(k).unsqueeze(0).float())
    valid = torch.from_numpy(np.isfinite(d) & (d > 0.05) & (d < 10.0)).unsqueeze(0).unsqueeze(0)
    mask = ~valid
    xyz = unproject_masked_depth_to_xyz_coordinates(depth=depth_t, pose=pose_t, inv_intrinsics=inv_k, mask=mask)
    return xyz.detach().cpu().numpy()


def _analyze_points(
    world_xyz: np.ndarray,
    *,
    base_xy: tuple[float, float],
    floor_band: tuple[float, float] = (0.0, 0.35),
) -> dict[str, Any]:
    pts = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return {"n_points": 0}
    heights = pts[:, 2]
    floor_mask = (heights >= floor_band[0]) & (heights <= floor_band[1])
    floor_pts = pts[floor_mask]
    normal, residual = _fit_plane_svd(floor_pts) if floor_pts.shape[0] >= 50 else (np.array([0, 0, 1.0]), float("nan"))
    centroid_xy = pts[:, :2].mean(axis=0)
    base = np.array(base_xy, dtype=np.float64)
    offset_m = float(np.linalg.norm(centroid_xy - base))
    return {
        "n_points": int(pts.shape[0]),
        "n_floor_points": int(floor_pts.shape[0]),
        "floor_normal": [float(x) for x in normal],
        "floor_normal_tilt_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, abs(float(normal[2]))))))),
        "floor_fit_residual_m": residual,
        "median_height_m": float(np.median(heights)),
        "pcd_centroid_xy": [float(centroid_xy[0]), float(centroid_xy[1])],
        "base_to_centroid_m": offset_m,
        "base_to_centroid_xy": [float(centroid_xy[0] - base[0]), float(centroid_xy[1] - base[1])],
    }


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {pts.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        cols = None if colors is None else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        for i, p in enumerate(pts):
            if cols is None:
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            else:
                c = cols[i]
                fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def analyze_bundle(bundle_dir: Path) -> dict[str, Any]:
    header, rows = _load_obs_history(bundle_dir)
    report: dict[str, Any] = {
        "mode": "bundle",
        "bundle_dir": str(bundle_dir.resolve()),
        "n_observations": len(rows),
        "grid_resolution": header.get("grid_resolution"),
        "spawn_record": header.get("spawn_record"),
    }
    deltas = [r.get("gps_camera_grid_delta_ij") for r in rows if r.get("gps_camera_grid_delta_ij")]
    if deltas:
        arr = np.asarray(deltas, dtype=np.float64)
        report["gps_camera_grid_delta_mean_ij"] = arr.mean(axis=0).tolist()
        report["gps_camera_grid_delta_std_ij"] = arr.std(axis=0).tolist()
        report["gps_camera_grid_mismatch_count"] = int(np.sum(np.abs(arr).sum(axis=1) > 0))

    spin_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        base = row.get("base_pose_xyt")
        if not base:
            continue
        key = f"{round(base[0], 3)}_{round(base[1], 3)}"
        spin_groups.setdefault(key, []).append(row)

    spin_summaries = []
    for key, group in spin_groups.items():
        if len(group) < 3:
            continue
        centroids = np.array([g.get("pcd_centroid_xy") or [0, 0] for g in group], dtype=np.float64)
        spread = float(np.max(np.linalg.norm(centroids - centroids.mean(axis=0, keepdims=True), axis=1)))
        headings = [float(g["base_pose_xyt"][2]) for g in group if g.get("base_pose_xyt")]
        spin_summaries.append(
            {
                "base_xy_key": key,
                "n_obs": len(group),
                "heading_span_deg": float(math.degrees(max(headings) - min(headings))) if headings else 0.0,
                "pcd_centroid_spread_m": spread,
            }
        )
    spin_summaries.sort(key=lambda s: s["n_obs"], reverse=True)
    report["spin_in_place_groups"] = spin_summaries[:5]

    pcd_offsets = []
    for row in rows:
        base = row.get("base_pose_xyt")
        cen = row.get("pcd_centroid_xy")
        if not base or not cen:
            continue
        dist = float(math.hypot(cen[0] - base[0], cen[1] - base[1]))
        if dist > 0.5:
            pcd_offsets.append({"obs_idx": row["obs_idx"], "offset_m": dist, "base": base[:2], "centroid": cen})
    report["large_pcd_centroid_offsets"] = pcd_offsets[:12]
    report["interpretation"] = (
        "gps_camera_grid_delta_ij should be ~[0,0] when camera_pose planar translation matches gps. "
        "Large constant delta (~7 cells) indicates pre-fix body-pitch camera bug. "
        "High pcd_centroid_spread at fixed XY is expected during rotate-in-place (tilted camera); "
        "fusing those views into one 2D cell creates obstacle 'fans' on the top-down map."
    )
    return report


def run_live_spin(*, question_id: int, turns: int, out_dir: Path) -> dict[str, Any]:
    from emet_habitat.observations import habitat_rgb_depth_to_observations
    from emet_habitat.simulator import HabitatEQASimulator

    from emet.habitat.config import default_hm3d_scene_dir
    from emet.habitat.datasets import get_question, load_hmeqa_questions, load_scene_init_poses

    out_dir.mkdir(parents=True, exist_ok=True)
    q = get_question(load_hmeqa_questions(None), question_id=question_id)
    init_pose = load_scene_init_poses(None)[(q.scene, q.floor)]
    sim = HabitatEQASimulator.from_scene_id(
        q.scene,
        hm3d_root=default_hm3d_scene_dir(),
        use_hm3d_semantics=False,
    )
    sim.set_init_pose(init_pose)
    floor_y = sim.floor_y

    rows: list[dict[str, Any]] = []
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    def _capture(label: str) -> None:
        frame = sim.get_frame()
        obs = habitat_rgb_depth_to_observations(
            rgb=frame.rgb,
            depth=frame.depth,
            agent_state=frame.agent_state,
            intrinsics=frame.intrinsics,
            floor_y=floor_y,
            sensor_height=sim.sensor_height,
            camera_tilt_deg=sim.camera_tilt_deg,
        )
        cam_pose = np.asarray(obs.camera_pose, dtype=np.float64)
        xyz = _unproject_obs(obs.rgb, obs.depth, obs.camera_K, cam_pose)
        base_xy = (float(obs.gps[0]), float(obs.gps[1]))
        stats = _analyze_points(xyz, base_xy=base_xy)
        cam_t = cam_pose[:3, 3]
        gi = int(round((base_xy[0] + 512.0) / 0.1))  # illustrative only
        gj = int(round((base_xy[1] + 512.0) / 0.1))
        cam_gi = int(round((float(cam_t[0]) + 512.0) / 0.1))
        cam_gj = int(round((float(cam_t[1]) + 512.0) / 0.1))
        row = {
            "label": label,
            "base_xy": list(base_xy),
            "compass_deg": float(math.degrees(float(obs.compass[0]))),
            "camera_t_voxel": [float(cam_t[0]), float(cam_t[1]), float(cam_t[2])],
            "gps_camera_grid_delta_ij": [cam_gi - gi, cam_gj - gj],
            **stats,
        }
        rows.append(row)
        if xyz.shape[0] > 0:
            step = len(rows)
            hue = int((step * 40) % 255)
            cols = np.stack(
                [
                    np.full(xyz.shape[0], hue, dtype=np.uint8),
                    np.full(xyz.shape[0], 80, dtype=np.uint8),
                    np.full(xyz.shape[0], 255 - hue, dtype=np.uint8),
                ],
                axis=1,
            )
            all_pts.append(xyz)
            all_cols.append(cols)

    _capture("init")
    for i in range(turns):
        sim.step("turn_left")
        _capture(f"turn_{i + 1}")

    if all_pts:
        _write_ply(out_dir / "spin_pointcloud.ply", np.vstack(all_pts), np.vstack(all_cols))

    report = {
        "mode": "live",
        "question_id": question_id,
        "scene": q.scene,
        "floor_y": floor_y,
        "turns": turns,
        "out_dir": str(out_dir.resolve()),
        "steps": rows,
    }
    (out_dir / "spin_projection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Episode bundle with observations_history.jsonl")
    parser.add_argument("--live", action="store_true", help="Run Q17 in Habitat-Sim (needs .venv-habitat)")
    parser.add_argument("--question-id", type=int, default=17)
    parser.add_argument("--turns", type=int, default=12, help="turn_left steps for live spin")
    parser.add_argument("--out", type=Path, default=None, help="Output dir for live PLY/JSON")
    parser.add_argument("--json", action="store_true", help="Print JSON report to stdout")
    args = parser.parse_args()

    if args.live:
        out = args.out or Path(f"/tmp/habitat_cam_diag_q{args.question_id:04d}")
        report = run_live_spin(question_id=args.question_id, turns=args.turns, out_dir=out)
    elif args.bundle:
        report = analyze_bundle(args.bundle.expanduser().resolve())
    else:
        parser.error("Pass --bundle PATH or --live")

    if args.json or not args.live:
        print(json.dumps(report, indent=2))
    else:
        print(json.dumps({k: report[k] for k in report if k != "steps"}, indent=2))
        print(f"Wrote {report['out_dir']}/spin_projection_report.json")
        print(f"Wrote {report['out_dir']}/spin_pointcloud.ply")


if __name__ == "__main__":
    main()
