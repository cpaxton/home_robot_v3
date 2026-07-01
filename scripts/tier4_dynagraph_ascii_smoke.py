#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Lightweight Dynagraph + ASCII nav grid smoke (no Qwen VL load)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from molmospaces_navgrid_smoke import collision_clip_from_merged_xml, run_navgrid_session  # noqa: E402

PORT_OFFSET = int(os.environ.get("EMET_TIER4_PORT_OFFSET", "110"))
EXPLORE_ITERS = int(os.environ.get("EMET_TIER4_EXPLORE_ITERS", "4"))


def main() -> int:
    os.environ["EMET_NAVGRID_ASCII"] = "1"
    os.environ.setdefault("EMET_NAVGRID_CONTEXTS", "rotate_in_place,explore")
    os.environ.setdefault("MUJOCO_GL", "egl")

    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.mapping.navgrid_compare import render_world_raster_ascii
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    ref_argv = prepare_mujoco_server_argv(
        SimLaunchMolmospaces(
            robot="stretch",
            scene="ithor",
            split="train",
            index=0,
            headless=True,
            molmospaces_install=False,
            port_offset=PORT_OFFSET,
        )
    )
    merged = None
    for i, a in enumerate(ref_argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(ref_argv):
            merged = ref_argv[i + 1]
            break
    if not merged:
        print("FAIL: no merged MJCF path", file=sys.stderr)
        return 1
    clip_rect = collision_clip_from_merged_xml(merged)

    print(
        f"tier4: stretch rotate_in_place + {EXPLORE_ITERS} explore steps",
        file=sys.stderr,
    )
    try:
        res = run_navgrid_session(
            "stretch",
            port_offset=PORT_OFFSET,
            clip_rect=clip_rect,
            explore_iters=EXPLORE_ITERS,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    ascii_map = render_world_raster_ascii(res.world_raster)
    print(ascii_map, file=sys.stderr)
    if "#" not in ascii_map or "." not in ascii_map:
        print("FAIL: expected navgrid ASCII with # and .", file=sys.stderr)
        return 1
    if res.explored_cells < 120:
        print(f"FAIL: too few explored cells ({res.explored_cells})", file=sys.stderr)
        return 1
    if res.explore_iters > 0 and res.explore_successes == 0:
        print(
            f"WARN: explore did not succeed ({res.explore_successes}/{res.explore_iters}); "
            f"scan map explored={res.explored_cells}",
            file=sys.stderr,
        )

    print(
        f"PASS: explored={res.explored_cells} explore_ok={res.explore_successes}/{res.explore_iters}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
