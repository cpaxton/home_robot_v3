#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
"""Write reusable frontier-pick step panels (demo or custom).

Examples:
  uv run python scripts/render_frontier_pick_demo.py
  uv run python scripts/render_frontier_pick_demo.py --out /tmp/frontier_demo --iters 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emet.visualization.frontier_pick_viz import (
    make_long_motion_demo_steps,
    write_frontier_pick_steps,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "emet" / "frontier_pick_demo",
        help="Directory for iter_XX.png panels",
    )
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--resolution", type=float, default=0.1)
    args = p.parse_args()

    steps, go = make_long_motion_demo_steps(
        n_iters=int(args.iters),
        grid_resolution=float(args.resolution),
    )
    paths = write_frontier_pick_steps(
        steps,
        args.out,
        grid_origin_xy=go,
        grid_resolution=float(args.resolution),
        max_side=int(args.max_side),
    )
    print(f"wrote {len(paths)} panels to {args.out}")
    for path, step in zip(paths, steps, strict=True):
        print(f"  {path.name}: {step.title} — {step.subtitle}")


if __name__ == "__main__":
    main()
