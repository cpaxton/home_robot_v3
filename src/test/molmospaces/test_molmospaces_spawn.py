# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

from __future__ import annotations

import mujoco

from emet.simulation.molmospaces_spawn import (
    upward_ray_hit_distance,
    walkable_floor_z_at_xy,
)

SLAB_OVER_ORIGIN = """<?xml version="1.0"?>
<mujoco model="slab_scene">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="8 8 0.01" rgba="0.7 0.7 0.7 1"/>
    <!-- Horizontal plate above origin blocks a naive single-segment downward ray -->
    <geom name="slab" type="box" pos="0 0 0.55" size="2.5 2.5 0.06" rgba="0.2 0.5 0.8 0.3"/>
  </worldbody>
</mujoco>
"""


def test_walkable_floor_z_multisegment_past_slab():
    m = mujoco.MjModel.from_xml_string(SLAB_OVER_ORIGIN)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    z0 = walkable_floor_z_at_xy(m, d, 0.0, 0.0)
    assert z0 is not None
    assert abs(z0) < 0.02
    z_clear = walkable_floor_z_at_xy(m, d, 4.0, 0.0)
    assert z_clear is not None
    assert abs(z_clear) < 0.02


def test_upward_ray_detects_open_void_beside_slab_scene():
    """Outside the slab footprint the upward ray misses geometry (void); under slab it hits."""
    m = mujoco.MjModel.from_xml_string(SLAB_OVER_ORIGIN)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert upward_ray_hit_distance(m, d, 4.0, 0.0, 0.12) is None
    d_hit = upward_ray_hit_distance(m, d, 0.0, 0.0, 0.12)
    assert d_hit is not None and d_hit >= 0.15
