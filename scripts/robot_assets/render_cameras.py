#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Render what each robot camera actually sees (RGB + depth) with objects in view.

Builds a scene = robot MJCF + a table + a few colorful objects in front, applies the
robot home keyframe, then renders every ``<camera>`` in the MJCF. Use this to eyeball
camera extrinsics/intrinsics before trusting perception in Robocasa/MolmoSpaces.

Requires the main emet venv (mujoco + PIL) and a working offscreen GL backend::

    uv run python scripts/robot_assets/render_cameras.py \
        --mjcf src/emet/assets/robot/sourccey/sourccey.xml \
        --out-dir /tmp/sourccey_cams

The tooling picks ``MUJOCO_GL=egl`` (fall back to ``osmesa`` if EGL is unavailable).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_SCENE_OBJECTS = (
    # (name, pos, size, rgba)
    ("table", (0.0, 1.05, 0.30), (0.45, 0.45, 0.06, 10.0), (0.55, 0.45, 0.35, 1.0)),
    ("red_cube", (0.10, 1.05, 0.72), (0.08, 0.08, 0.08, 0.5), (1.0, 0.15, 0.15, 1.0)),
    ("blue_cylinder", (-0.12, 1.05, 0.70), (0.05, 0.05, 0.18, 0.4), (0.15, 0.3, 1.0, 1.0)),
    ("green_sphere", (0.0, 1.30, 0.42), (0.07, 0.07, 0.07, 0.3), (0.15, 0.8, 0.2, 1.0)),
)


def render_cameras(mjcf: Path, out_dir: Path, width: int = 640, height: int = 480) -> list[Path]:
    import mujoco
    import numpy as np
    from PIL import Image

    robot_xml = Path(mjcf).read_text(encoding="utf-8")
    scene_xml = robot_xml.replace(
        '<geom type="plane" size="4 4 0.02" material="plastic_dark"/>',
        '<geom type="plane" size="4 4 0.02" material="plastic_dark"/>'
        + "".join(
            f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">'
            f'<geom type="box" size="{size[0] / 2} {size[1] / 2} {size[2] / 2}" mass="{size[3]}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/></body>'
            for name, pos, size, rgba in _SCENE_OBJECTS
        ),
    )

    cwd = Path.cwd()
    os.chdir(Path(mjcf).resolve().parent)
    try:
        model = mujoco.MjModel.from_xml_string(scene_xml)
        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        # settle onto the floor
        for _ in range(30):
            mujoco.mj_step(model, data)
    finally:
        os.chdir(cwd)

    renderer = mujoco.Renderer(model, width, height)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i in range(model.ncam):
        cam = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
        if cam.startswith("preview_"):
            # third-person orbit shots of the whole robot
            renderer.update_scene(data, camera=cam)
            rgb = np.asarray(renderer.render())
            p = out_dir / f"{cam}.png"
            Image.fromarray(rgb).save(p)
            saved.append(p)
            print(f"{cam:14s} rgb={p}")
            continue
        renderer.update_scene(data, camera=cam)
        rgb = np.asarray(renderer.render())
        renderer.enable_depth_rendering()
        depth = np.asarray(renderer.render()).copy()
        renderer.disable_depth_rendering()
        rgb_path = out_dir / f"{cam}_rgb.png"
        depth_path = out_dir / f"{cam}_depth.png"
        raw_path = out_dir / f"{cam}_depth.npy"
        Image.fromarray(rgb).save(rgb_path)
        np.save(raw_path, depth)
        # depth -> grey image, clipped to a near window (0.2..3 m) so close objects are visible.
        # In this window: close = bright, far / invalid = black. Values are meters.
        finite = np.isfinite(depth)
        d = depth.copy()
        d[~finite] = np.nan
        near, far = 0.2, 3.0
        grey = np.full_like(d, np.nan)
        grey[finite] = np.clip((d[finite] - near) / (far - near), 0.0, 1.0)
        grey[~finite] = 0.0
        Image.fromarray((grey * 255).astype(np.uint8)).save(depth_path)
        saved.extend([rgb_path, depth_path, raw_path])
        print(f"{cam:14s} rgb={rgb_path} depth={depth_path} raw={raw_path}")
    renderer.close()
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mjcf", type=Path, required=True, help="Robot MJCF (e.g. src/emet/assets/robot/sourccey/sourccey.xml)."
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/robot_cameras"), help="Where to write PNGs.")
    args = ap.parse_args()
    render_cameras(args.mjcf, args.out_dir)


if __name__ == "__main__":
    main()
