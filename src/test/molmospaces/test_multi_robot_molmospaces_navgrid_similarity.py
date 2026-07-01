# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SRC_ROOT.parent / "scripts") not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT.parent / "scripts"))


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").strip().lower() in ("1", "true", "yes", "on")


_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


def _skip_reason() -> str | None:
    if not RUN_SIM_TESTS:
        return "RUN_SIM_TESTS=0"
    if not _truthy("RUN_MOLMOSPACES_TESTS"):
        return "RUN_MOLMOSPACES_TESTS=1 required"
    if not _truthy("RUN_MULTI_ROBOT_NAVGRID"):
        return "RUN_MULTI_ROBOT_NAVGRID=1 required (heavy ~30+ min)"
    return None


_SKIP = _skip_reason()


@pytest.mark.timeout(2400)
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "skipped")
def test_multi_robot_molmospaces_navgrid_similarity():
    from molmospaces_navgrid_smoke import collision_clip_from_merged_xml, run_navgrid_session

    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.mapping.navgrid_compare import compare_world_rasters_in_shared_view
    from emet.simulation.molmospaces_config import build_molmospaces_wrapper_command
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    if build_molmospaces_wrapper_command(["merge-scene", "--help"]) is None:
        pytest.skip("MolmoSpaces wrapper missing")

    robots = ["stretch", "rby1", "innate_mars", "xlerobot"]
    port_base = max(1, int(os.environ.get("EMET_TIER4_PORT_OFFSET", "110")))
    explore_iters = max(1, int(os.environ.get("EMET_TIER4_EXPLORE_ITERS", "2")))
    min_exp = float(os.environ.get("EMET_NAVGRID_MIN_EXPLORED_IOU", "0.25"))
    min_obs = float(os.environ.get("EMET_NAVGRID_MIN_OBSTACLE_IOU", "0.20"))

    ref_argv = prepare_mujoco_server_argv(
        SimLaunchMolmospaces(
            robot="stretch",
            scene="ithor",
            split="train",
            index=0,
            headless=True,
            molmospaces_install=False,
            port_offset=port_base,
        )
    )
    merged = None
    for i, a in enumerate(ref_argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(ref_argv):
            merged = ref_argv[i + 1]
            break
    assert merged
    clip = collision_clip_from_merged_xml(merged)

    rasters = {}
    for idx, robot in enumerate(robots):
        res = run_navgrid_session(
            robot,
            port_offset=port_base + idx * 10,
            clip_rect=clip,
            explore_iters=explore_iters,
        )
        rasters[robot] = res.world_raster

    ref = rasters["stretch"]
    for robot in ("rby1", "innate_mars", "xlerobot"):
        sim = compare_world_rasters_in_shared_view(ref, rasters[robot])
        assert sim.explored_iou >= min_exp, f"{robot} explored_iou={sim.explored_iou}"
        assert sim.obstacle_iou >= min_obs, f"{robot} obstacle_iou={sim.obstacle_iou}"
