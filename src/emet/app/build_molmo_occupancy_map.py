# Copyright (c) Hello Robot, Inc. All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

"""Build MolmoSpaces-style iTHOR orthographic occupancy map from a merged MJCF (debug / QA).

Writes PNG (+ embedded JSON transforms) and ``occupancy_meta.json`` next to the output basename.

Example::

    uv run python -m emet.app.build_molmo_occupancy_map /path/to/molmospaces_merged_abc.xml -o /tmp/occ_out

Environment: ``EMET_MOLMOSPACES_OCC_SEED`` (optional), ``MUJOCO_GL`` (e.g. egl for headless).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from emet.simulation.molmo_occupancy.ithor_map import iTHORMap


def run_build(
    mjcf: Path,
    output_dir: Path | None,
    *,
    agent_radius: float = 0.32,
    px_per_m: int = 120,
) -> tuple[Path, Path]:
    mjcf = mjcf.resolve()
    if not mjcf.is_file():
        raise FileNotFoundError(str(mjcf))
    out_dir = (output_dir or mjcf.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    th = iTHORMap.from_mj_model_path(str(mjcf), camera=None, agent_radius=float(agent_radius), px_per_m=int(px_per_m))
    png = out_dir / "occupancy.png"
    th.save(str(png))
    meta = {
        "source_mjcf": str(mjcf),
        "px_per_m": th.px_per_m,
        "world_to_map": th.world_to_map.tolist(),
        "map_to_world": th.map_to_world.tolist(),
        "occupancy_png": str(png),
    }
    meta_path = out_dir / "occupancy_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return png, meta_path


def main() -> None:
    p = argparse.ArgumentParser(description="Build iTHOR-style 2D occupancy map from merged MJCF.")
    p.add_argument("mjcf", type=str, help="Path to merged scene+robot MJCF on disk.")
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="Directory for occupancy.png and occupancy_meta.json (default: same dir as mjcf).",
    )
    p.add_argument("--agent-radius", type=float, default=0.32, help="Dilation radius in meters.")
    p.add_argument("--px-per-m", type=int, default=120, help="Orthographic map resolution.")
    args = p.parse_args()
    mjcf = Path(args.mjcf)
    out = Path(args.output_dir) if str(args.output_dir).strip() else None
    png, meta = run_build(mjcf, out, agent_radius=args.agent_radius, px_per_m=args.px_per_m)
    print(f"Wrote {png} and {meta}")


if __name__ == "__main__":
    main()
