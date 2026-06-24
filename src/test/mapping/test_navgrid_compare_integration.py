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

"""Unit tests for navgrid clip + raster compare (no sim server)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _SRC_ROOT.parent / "scripts"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_resolve_base_body_id_prefers_chassis():
    import mujoco
    from molmospaces_navgrid_smoke import _resolve_base_body_id

    xml = """<mujoco><worldbody>
      <body name="chassis"><geom type="sphere" size="0.1"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    assert _resolve_base_body_id(model, "chassis") >= 0
    assert _resolve_base_body_id(model, None) >= 0


def test_compare_world_rasters_in_shared_view_partial_overlap():
    from emet.mapping.navgrid_compare import WorldMapRaster, compare_world_rasters_in_shared_view

    meta = {"resolution_m": 0.1, "origin_xy": (0.0, 0.0), "clip_rect": (0.0, 2.0, 0.0, 2.0)}
    exp = np.zeros((20, 20), dtype=bool)
    obs = np.zeros((20, 20), dtype=bool)
    obs[5:15, 5:15] = True
    exp[8:12, 8:12] = True
    a = WorldMapRaster(explored=exp.copy(), obstacles=obs.copy(), **meta)
    b = WorldMapRaster(explored=exp.copy(), obstacles=obs.copy(), **meta)
    b.explored[10:18, 10:18] = True
    sim = compare_world_rasters_in_shared_view(a, b)
    assert 0.15 < sim.explored_iou < 1.0
    assert sim.obstacle_iou == pytest.approx(1.0)
