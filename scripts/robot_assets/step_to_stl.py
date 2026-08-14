#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Convert STEP CAD parts into normalized STL meshes for vendored MJCF assets.

Run with a venv that has ``cadquery`` (the main emet venv does not install it)::

    uv venv /tmp/robot_assets_venv --python 3.10
    uv pip install --python /tmp/robot_assets_venv/bin/python cadquery
    /tmp/robot_assets_venv/bin/python scripts/robot_assets/step_to_stl.py \\
        /path/to/CAD/Robot/Arms/Base.step --out /tmp/meshes/arm_base.stl --scale 0.001

Batch mode: pass a JSON mapping ``{out_name: step_path}`` with ``--manifest``.
Output meshes are scaled to meters (``--scale``, default 0.001 since STEP is mm)
and recentered on their bounding-box centroid so MJCF authors place them by frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_one(step_path: Path, out_path: Path, scale: float, tolerance: float) -> tuple[list[float], list[float]]:
    """Convert a single STEP file to STL. Returns (size_m, centroid_m) in output units."""
    import cadquery as cq

    shape = cq.importers.importStep(str(step_path))
    cq.exporters.export(shape, str(out_path), tolerance=tolerance, angularTolerance=tolerance)
    box = shape.val().BoundingBox()
    size = [
        (box.xmax - box.xmin) * scale,
        (box.ymax - box.ymin) * scale,
        (box.zmax - box.zmin) * scale,
    ]
    centroid = [
        (box.xmin + box.xmax) / 2.0 * scale,
        (box.ymin + box.ymax) / 2.0 * scale,
        (box.zmin + box.zmax) / 2.0 * scale,
    ]
    return size, centroid


def recenter_stl(out_path: Path) -> None:
    """Translate an STL so its bounding-box centroid sits at the origin (idempotent)."""
    import trimesh

    mesh = trimesh.load(str(out_path), force="mesh")
    centroid = mesh.bounding_box.centroid
    if centroid.shape != (3,):
        return
    mesh.apply_translation(-centroid)
    mesh.export(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", type=Path, nargs="?", help="Single STEP file to convert.")
    ap.add_argument("--out", type=Path, help="Output STL path (single-file mode).")
    ap.add_argument("--manifest", type=Path, help="JSON mapping {out_name: step_path} for batch mode.")
    ap.add_argument("--out-dir", type=Path, default=Path("meshes"), help="Output dir (batch mode).")
    ap.add_argument("--scale", type=float, default=0.001, help="mm->m scale (STEP is mm).")
    ap.add_argument("--tolerance", type=float, default=0.2, help="Tessellation tolerance (mm).")
    ap.add_argument("--no-recenter", action="store_true", help="Keep original CAD frame.")
    args = ap.parse_args()

    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        info = {}
        for out_name, step in manifest.items():
            step_path = Path(step)
            out_path = out_dir / f"{out_name}.stl"
            size, centroid = convert_one(step_path, out_path, args.scale, args.tolerance)
            if not args.no_recenter:
                recenter_stl(out_path)
            info[out_name] = {
                "step": str(step_path),
                "size_m": [round(v, 6) for v in size],
                "centroid_m": [round(v, 6) for v in centroid],
            }
            print(f"{out_name:24s} size_m={size}")
        (out_dir / "mesh_info.json").write_text(json.dumps(info, indent=2))
        print(f"Wrote {len(info)} meshes to {out_dir}")
        return

    if args.step is None or args.out is None:
        ap.error("Provide --step with --out, or --manifest with --out-dir.")
    size, centroid = convert_one(args.step, args.out, args.scale, args.tolerance)
    if not args.no_recenter:
        recenter_stl(args.out)
    print(f"size_m={size} centroid_m={centroid} -> {args.out}")


if __name__ == "__main__":
    main()
