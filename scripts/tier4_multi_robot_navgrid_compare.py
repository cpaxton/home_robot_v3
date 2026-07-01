#!/usr/bin/env python3
# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Compare ASCII/world nav maps across MolmoSpaces robots on the same iTHOR scene.

Runs rotate_in_place + frontier exploration for each robot sequentially, rasterizes
occupancy in a shared world-frame clip rect, and reports IoU vs a reference robot
(default: stretch).

Example::

  EMET_NAVGRID_ASCII=1 uv run python scripts/tier4_multi_robot_navgrid_compare.py

  EMET_NAVGRID_COMPARE_ROBOTS=stretch,rby1 EMET_TIER4_EXPLORE_ITERS=2 \\
    uv run python scripts/tier4_multi_robot_navgrid_compare.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from molmospaces_navgrid_smoke import (  # noqa: E402
    DEFAULT_MOLMO_ROBOTS,
    collision_clip_from_merged_xml,
    run_navgrid_session,
)


def _parse_robots() -> list[str]:
    raw = os.environ.get("EMET_NAVGRID_COMPARE_ROBOTS", "").strip()
    if raw:
        return [r.strip().lower().replace("-", "_") for r in raw.split(",") if r.strip()]
    return list(DEFAULT_MOLMO_ROBOTS)


def main() -> int:
    os.environ.setdefault("EMET_NAVGRID_ASCII", "1")
    os.environ.setdefault("EMET_NAVGRID_CONTEXTS", "rotate_in_place,explore")
    os.environ.setdefault("MUJOCO_GL", "egl")

    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.mapping.navgrid_compare import (
        compare_world_rasters_in_shared_view,
        format_similarity_table,
        render_world_raster_ascii,
    )
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    robots = _parse_robots()
    if len(robots) < 2:
        print("Need at least 2 robots in EMET_NAVGRID_COMPARE_ROBOTS", file=sys.stderr)
        return 2

    explore_iters = int(os.environ.get("EMET_TIER4_EXPLORE_ITERS", "3"))
    port_base = int(os.environ.get("EMET_TIER4_PORT_OFFSET", "110"))
    port_step = int(os.environ.get("EMET_TIER4_PORT_STEP", "10"))
    reference = os.environ.get("EMET_NAVGRID_REFERENCE_ROBOT", "stretch").lower().replace("-", "_")
    min_exp_iou = float(os.environ.get("EMET_NAVGRID_MIN_EXPLORED_IOU", "0.25"))
    min_obs_iou = float(os.environ.get("EMET_NAVGRID_MIN_OBSTACLE_IOU", "0.20"))

    if reference not in robots:
        robots = [reference] + [r for r in robots if r != reference]

    # Shared world clip from reference robot merge (scene geometry; robot bodies excluded).
    ref_argv = prepare_mujoco_server_argv(
        SimLaunchMolmospaces(
            robot=reference,
            scene="ithor",
            split="train",
            index=0,
            headless=True,
            molmospaces_install=False,
            port_offset=port_base,
        )
    )
    ref_merged = None
    for i, a in enumerate(ref_argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(ref_argv):
            ref_merged = ref_argv[i + 1]
            break
    if not ref_merged:
        print("FAIL: could not resolve reference merged MJCF", file=sys.stderr)
        return 1
    clip_rect = collision_clip_from_merged_xml(ref_merged)
    print(
        f"multi-robot navgrid: robots={robots} reference={reference} clip={clip_rect} explore_iters={explore_iters}",
        file=sys.stderr,
    )

    results = []
    rasters: dict[str, object] = {}
    for idx, robot in enumerate(robots):
        port_offset = port_base + idx * port_step
        print(f"\n=== robot {robot} (port_offset={port_offset}) ===", file=sys.stderr)
        try:
            res = run_navgrid_session(
                robot,
                port_offset=port_offset,
                clip_rect=clip_rect,
                explore_iters=explore_iters,
            )
        except Exception as exc:
            print(f"FAIL: session for {robot}: {exc}", file=sys.stderr)
            return 1
        results.append(res)
        rasters[robot] = res.world_raster
        print(
            f"{robot}: explored={res.explored_cells} obstacles={res.obstacle_cells} "
            f"explore_ok={res.explore_successes}/{res.explore_iters}",
            file=sys.stderr,
        )

    table = format_similarity_table(robots, rasters, reference=reference)
    print("\nSimilarity vs reference (world 0.1m raster, shared-view IoU):\n" + table, file=sys.stderr)

    for robot in robots:
        print(f"\n[world_raster:{robot}]\n{render_world_raster_ascii(rasters[robot])}", file=sys.stderr)

    ref_raster = rasters[reference]
    failed: list[str] = []
    for robot in robots:
        if robot == reference:
            continue
        sim = compare_world_rasters_in_shared_view(ref_raster, rasters[robot])
        if sim.explored_iou < min_exp_iou:
            failed.append(f"{robot} explored_iou={sim.explored_iou:.3f} < {min_exp_iou}")
        if sim.obstacle_iou < min_obs_iou:
            failed.append(f"{robot} obstacle_iou={sim.obstacle_iou:.3f} < {min_obs_iou}")

    if failed:
        print("FAIL: maps not similar enough vs reference:", file=sys.stderr)
        for line in failed:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print(
        f"\nPASS: {len(robots)} robots produced similar world maps "
        f"(explored IoU>={min_exp_iou}, obstacle IoU>={min_obs_iou} vs {reference})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
