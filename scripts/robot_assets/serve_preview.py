#!/usr/bin/env python3
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Render a top/front/side preview of a generated MJCF so you can eyeball an assembly.

Requires the main emet venv (mujoco). Offscreen rendering works without a display
when ``MUJOCO_GL=egl`` (or ``osmesa``) is set::

    uv run python scripts/robot_assets/serve_preview.py \\
        --mjcf src/emet/assets/robot/sourccey/sourccey.xml \\
        --out /tmp/sourccey_preview.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def render_preview(mjcf: Path, out: Path) -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    # apply first keyframe (robot home pose) if present, else just forward
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    model.vis.global_.offwidth = max(1024, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(1024, model.vis.global_.offheight)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, 1024, 1024)
    views = ["preview_front", "preview_34", "preview_top", "preview_side"]
    angles = [(0, -10), (135, -20), (90, -90), (90, -10)]
    for name, (azimuth, elevation) in zip(views, angles, strict=True):
        camera = name
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = model.stat.center
            camera.distance = 2.0 * model.stat.extent
            camera.azimuth, camera.elevation = azimuth, elevation
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        p = out.with_name(f"{out.stem}_{name}{out.suffix}")
        from PIL import Image

        Image.fromarray(pixels).save(p)
        print(f"wrote {p}")
    renderer.close()


def _cam_xyz(azimuth: float, elevation: float) -> tuple[float, float, float]:
    import math

    az = math.radians(azimuth)
    el = math.radians(elevation)
    return (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/robot_preview.png"))
    args = ap.parse_args()
    render_preview(args.mjcf, args.out)


if __name__ == "__main__":
    main()
