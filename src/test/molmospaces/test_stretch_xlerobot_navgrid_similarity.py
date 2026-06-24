# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Stretch vs XLeRobot MolmoSpaces navgrid similarity (frontier explore, no neural nets).

Uses DynagraphController in ``manipulation_only`` mode: ``rotate_in_place`` + ``run_exploration``
(voxel frontier sampling + A*, no CLIP/OWL). Compares world-aligned explored/obstacle rasters via IoU.

Run (heavy, ~15–25 min depending on explore iters)::

    RUN_MOLMOSPACES_TESTS=1 RUN_STRETCH_XLEROBOT_NAVGRID=1 \\
      uv run emet test src/test/molmospaces/test_stretch_xlerobot_navgrid_similarity.py -v

Tune::

    EMET_TIER4_EXPLORE_ITERS=3          # frontier explore steps per robot (default 2)
    EMET_NAVGRID_MIN_EXPLORED_IOU=0.20   # vs stretch reference
    EMET_NAVGRID_MIN_OBSTACLE_IOU=0.18
    EMET_TIER4_PORT_OFFSET=120          # ZMQ port base offset
"""

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
        return "RUN_MOLMOSPACES_TESTS=1 required (MolmoSpaces assets + wrapper)"
    if not _truthy("RUN_STRETCH_XLEROBOT_NAVGRID"):
        return "RUN_STRETCH_XLEROBOT_NAVGRID=1 required (heavy sim integration)"
    return None


_SKIP = _skip_reason()


@pytest.mark.timeout(1800)
@pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "skipped")
def test_stretch_xlerobot_molmospaces_navgrid_similarity():
    """Same iTHOR scene: stretch and xlerobot should build similar floor/obstacle maps."""
    from molmospaces_navgrid_smoke import collision_clip_from_merged_xml, run_navgrid_session

    from emet.config.sim_launch_config import SimLaunchMolmospaces
    from emet.mapping.navgrid_compare import compare_world_rasters_in_shared_view, format_similarity_table
    from emet.robots import get_robot_spec
    from emet.simulation.molmospaces_config import build_molmospaces_wrapper_command
    from emet.simulation.mujoco_serve_argv import prepare_mujoco_server_argv

    if build_molmospaces_wrapper_command(["merge-scene", "--help"]) is None:
        pytest.skip("MolmoSpaces wrapper missing")

    robots = ("stretch", "xlerobot")
    port_base = max(1, int(os.environ.get("EMET_TIER4_PORT_OFFSET", "120")))
    explore_iters = max(1, int(os.environ.get("EMET_TIER4_EXPLORE_ITERS", "2")))
    min_exp = float(os.environ.get("EMET_NAVGRID_MIN_EXPLORED_IOU", "0.20"))
    min_obs = float(os.environ.get("EMET_NAVGRID_MIN_OBSTACLE_IOU", "0.18"))

    stretch_argv = prepare_mujoco_server_argv(
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
    for i, a in enumerate(stretch_argv):
        if a in ("--scene_path", "--scene-path") and i + 1 < len(stretch_argv):
            merged = stretch_argv[i + 1]
            break
    assert merged

    stretch_spec = get_robot_spec("stretch")
    clip = collision_clip_from_merged_xml(merged, base_body_name=stretch_spec.base_link_name if stretch_spec else None)

    rasters = {}
    for idx, robot in enumerate(robots):
        spec = get_robot_spec(robot)
        base = spec.base_link_name if spec else None
        robot_clip = collision_clip_from_merged_xml(merged, base_body_name=base)
        assert robot_clip == clip, f"{robot} clip_rect differs from stretch (spawn/footprint mismatch?)"
        res = run_navgrid_session(
            robot,
            port_offset=port_base + idx * 10,
            clip_rect=clip,
            explore_iters=explore_iters,
        )
        rasters[robot] = res.world_raster

    table = format_similarity_table(list(robots), rasters, reference="stretch")
    print(table)

    sim = compare_world_rasters_in_shared_view(rasters["stretch"], rasters["xlerobot"])
    assert sim.explored_iou >= min_exp, (
        f"xlerobot explored_iou={sim.explored_iou:.3f} < {min_exp} "
        f"(stretch cells={sim.explored_a}, xlerobot={sim.explored_b})"
    )
    assert sim.obstacle_iou >= min_obs, (
        f"xlerobot obstacle_iou={sim.obstacle_iou:.3f} < {min_obs} "
        f"(stretch obs={sim.obstacles_a}, xlerobot={sim.obstacles_b})"
    )
