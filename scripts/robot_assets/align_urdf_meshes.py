#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Bake URDF-joint alignment into arm-link STL meshes so consecutive links connect.

Motivation: a vendor URDF (e.g. ``lerobot-vulcan``'s ``Arm.urdf``) ships visual origins
tuned for its own STL meshes. Our STEP-derived meshes have a different local frame, so
placing them at the URDF visual origins leaves visible gaps between links. This script
instead rotates each link mesh so its long axis points along the link's arm direction
(``+y`` or ``-y`` in the body frame, from the URDF joint chain) and anchors its entry
end at the body origin — guaranteeing consecutive links overlap regardless of the
original mesh frame.

Run with the main emet venv (trimesh + scipy)::

    uv run python scripts/robot_assets/align_urdf_meshes.py \
        --in-dir /tmp/raw_meshes_mm --out-dir /tmp/aligned_meshes \
        --links arm_shoulder=+1,arm_bicep_l=-1,arm_forearm=+1,arm_wrist=+1

``--links`` maps mesh basename -> sign (+1 or -1) of the arm direction in the link frame
(long axis aligns to ``+y`` or ``-y``; the sign matches whether the body's +y points
toward or away from the next joint, verified against the assembled MJCF).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True, help="Raw (mm) meshes dir.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Aligned meshes dir (mm).")
    ap.add_argument(
        "--links",
        required=True,
        help="Comma list mesh_basename=sign (e.g. arm_shoulder=+1,arm_bicep_l=-1,arm_forearm=+1,arm_wrist=+1).",
    )
    args = ap.parse_args()

    import trimesh
    from scipy.spatial.transform import Rotation as R

    links = {}
    for item in args.links.split(","):
        name, sign = item.split("=")
        links[name] = int(sign)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for mesh_name, sign in links.items():
        m = trimesh.load(str(args.in_dir / f"{mesh_name}.stl"))
        size = m.bounds[1] - m.bounds[0]
        ax = int(np.argmax(size))
        longdir = np.zeros(3)
        longdir[ax] = 1.0
        target = np.array([0.0, float(sign), 0.0])
        q = R.align_vectors([target], [longdir])[0]
        T = np.eye(4)
        T[:3, :3] = q.as_matrix()
        m.apply_transform(T)
        # Anchor the mesh's ENTRY end (the end toward -sign*y, i.e. the joint at the body
        # origin) at y=0 so the mesh spans outward from the entry joint toward the exit.
        # For sign=+1 the long axis points +y, so the entry (min-y) sits at 0; for sign=-1
        # the long axis points -y, so the entry (max-y) sits at 0.
        if sign == 1:
            shift = -m.bounds[0, 1]
        else:
            shift = -m.bounds[1, 1]
        m.apply_translation([0.0, shift, 0.0])
        out = args.out_dir / f"{mesh_name}.stl"
        m.export(str(out))
        print(f"{mesh_name:14s} sign={sign:+d} yspan=[{m.bounds[0, 1]:.0f},{m.bounds[1, 1]:.0f}]mm -> {out}")


if __name__ == "__main__":
    main()
