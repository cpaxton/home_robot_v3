# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Geometric smoke: recover a known start-pose fudge (no GPU / sim required).

Prints measured pre/post errors so we can claim local refine works without a perfect spawn.

  uv run python scripts/smoke_lifelong_pose_refine.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from emet.memory.lifelong import refine_start_pose, se2_matrix, transform_points_xyz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=0.30, help="Fudge translation X (m)")
    parser.add_argument("--dy", type=float, default=-0.15, help="Fudge translation Y (m)")
    parser.add_argument("--dyaw-deg", type=float, default=15.0, help="Fudge yaw (degrees)")
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for measured numbers",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Structured walls (ICP needs geometry; isotropic blobs fail from identity).
    xs = np.linspace(-1.5, 1.5, max(40, int(args.n_points) // 6))
    ys = np.linspace(-1.5, 1.5, max(40, int(args.n_points) // 6))
    wall_a = np.column_stack([np.full_like(ys, -1.5), ys, np.zeros_like(ys)])
    wall_b = np.column_stack([xs, np.full_like(xs, 1.5), np.zeros_like(xs)])
    clutter = rng.normal(size=(max(50, int(args.n_points) // 5), 3)) * 0.3
    saved = np.concatenate([wall_a, wall_b, clutter], axis=0)
    t_gt = se2_matrix(float(args.dx), float(args.dy), float(np.deg2rad(args.dyaw_deg)))
    live = transform_points_xyz(saved, t_gt)

    # Pre: error if we assumed identity (perfect pose) despite fudge
    pre_err = float(np.mean(np.linalg.norm(saved - live, axis=1)))
    result = refine_start_pose(saved, live, min_points=64)
    aligned = transform_points_xyz(saved, result.transform)
    post_err = float(np.mean(np.linalg.norm(aligned - live, axis=1)))

    payload = {
        "fudge_xy_m": [float(args.dx), float(args.dy)],
        "fudge_yaw_deg": float(args.dyaw_deg),
        "pre_mean_err_m": pre_err,
        "post_mean_err_m": post_err,
        "accepted": bool(result.accepted),
        "reason": result.reason,
        "estimated_xy_m": float(result.translation_xy_m),
        "estimated_yaw_deg": float(np.degrees(result.yaw_rad)),
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "num_inliers": int(result.num_inliers),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ok = result.accepted and post_err < 0.05 and post_err < pre_err * 0.25
    if not ok:
        print("FAIL: refine did not recover the fudge within tolerance", file=sys.stderr)
        return 1
    print(
        f"OK: pre_err={pre_err:.4f} m → post_err={post_err:.4f} m "
        f"(est xy={result.translation_xy_m:.3f} m yaw={np.degrees(result.yaw_rad):.1f}°)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
