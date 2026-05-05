# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

from __future__ import annotations

import math
import os
from pathlib import Path

import mujoco
import pytest

import emet.simulation.molmospaces_spawn as molmospaces_spawn
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

SLAB_GROUND_NAME = """<?xml version="1.0"?>
<mujoco model="slab_ground">
  <worldbody>
    <geom name="Ground" type="plane" pos="0 0 0" size="8 8 0.01" rgba="0.7 0.7 0.7 1"/>
    <geom name="slab" type="box" pos="0 0 0.55" size="2.5 2.5 0.06" rgba="0.2 0.5 0.8 0.3"/>
  </worldbody>
</mujoco>
"""


def test_resolve_floor_geom_name_finds_ground():
    from emet.simulation.molmospaces_spawn import resolve_floor_geom_name

    m = mujoco.MjModel.from_xml_string(SLAB_GROUND_NAME)
    assert resolve_floor_geom_name(m) == "Ground"


def test_walkable_floor_z_uses_effective_floor_name():
    m = mujoco.MjModel.from_xml_string(SLAB_GROUND_NAME)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    z0 = walkable_floor_z_at_xy(m, d, 0.0, 0.0, floor_geom_name="floor")
    assert z0 is not None
    assert abs(z0) < 0.02


def test_want_molmospaces_autoplace_off():
    from emet.simulation.molmospaces_spawn import want_molmospaces_autoplace

    assert not want_molmospaces_autoplace(
        environment={"kind": "molmospaces"},
        scene_source_basename=None,
        molmospaces_autoplace_env="0",
    )


def test_want_molmospaces_autoplace_kind_and_merged_basename():
    from emet.simulation.molmospaces_spawn import want_molmospaces_autoplace

    assert want_molmospaces_autoplace(
        environment={"kind": "molmospaces"},
        scene_source_basename="x.xml",
        molmospaces_autoplace_env="1",
    )
    assert want_molmospaces_autoplace(
        environment={"kind": "default_table"},
        scene_source_basename="molmospaces_merged_abc.xml",
        molmospaces_autoplace_env="1",
    )
    assert want_molmospaces_autoplace(
        environment={"kind": "default_table"},
        scene_source_basename="house_merged_rby1.xml",
        molmospaces_autoplace_env="1",
    )
    assert not want_molmospaces_autoplace(
        environment={"kind": "default_table"},
        scene_source_basename="FloorPlan12.xml",
        molmospaces_autoplace_env="1",
    )
    assert want_molmospaces_autoplace(
        environment={"kind": "default_table"},
        scene_source_basename="FloorPlan12.xml",
        molmospaces_autoplace_env="extended",
    )



def test_iter_annulus_xy_candidates_respects_xy_origin():
    """Rings must be around *xy_origin*, not world (0,0), so offset houses get interior samples."""
    from emet.simulation.molmospaces_spawn import iter_annulus_xy_candidates

    clip = (4.0, 12.0, -1.0, 7.0)
    pts = list(
        iter_annulus_xy_candidates(
            r_min=0.4,
            r_max=2.0,
            n_radii=4,
            base_angles_per_ring=8,
            xy_clip=clip,
            xy_origin=(8.0, 3.0),
        )
    )
    assert len(pts) >= 4
    for x, y in pts:
        assert clip[0] <= x <= clip[1] and clip[2] <= y <= clip[3]
    # (0,0) is outside clip; if we wrongly used origin, almost no points would land in clip
    assert any(math.hypot(x - 8.0, y - 3.0) < 1.5 for x, y in pts)


MINIMAL_OPEN_FLOOR_WITH_BASE = """<?xml version="1.0"?>
<mujoco model="open_floor_base">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="12 12 0.01" rgba="0.75 0.75 0.75 1"/>
    <body name="base_link" pos="-2.5 0 1.35">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="0.2 0.2 0.12" contype="1" conaffinity="1" rgba="0 0.85 0.25 0.55"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_settle_free_base_z_lowers_foot_near_floor_on_open_plane():
    """Starting high above the floor, settle descends until the collision hull nears z_floor."""
    m = mujoco.MjModel.from_xml_string(MINIMAL_OPEN_FLOOR_WITH_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert bid >= 0
    rb = molmospaces_spawn._bodies_descending_from(m, bid)  # noqa: SLF001
    x, y = -2.5, 0.0
    zf = molmospaces_spawn.walkable_floor_z_at_xy(m, d, x, y, exclude_body_id=bid)
    assert zf is not None and abs(float(zf)) < 0.02
    z_done = molmospaces_spawn.settle_free_base_z_to_floor(
        m,
        d,
        base_body_name="base_link",
        floor_geom_name="floor",
        x=x,
        y=y,
        z_floor=float(zf),
        z_start=float(zf) + 1.25,
        robot_bodies=rb,
        min_nonfloor_clearance=-5e-5,
        max_steps=500,
    )
    assert z_done is not None
    assert molmospaces_spawn.write_freejoint_base_xyzw(
        m, d, base_body_name="base_link", x=x, y=y, z=float(z_done)
    )
    mujoco.mj_forward(m, d)
    zb = molmospaces_spawn._min_robot_collision_geom_bottom_z(m, d, rb)  # noqa: SLF001
    assert zb is not None
    assert float(zb) <= float(zf) + 0.10
    assert float(zb) >= float(zf) - 0.05


MINIMAL_BASE_INSIDE_WALL = """<?xml version="1.0"?>
<mujoco model="base_in_wall">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="12 12 0.01" rgba="0.75 0.75 0.75 1"/>
    <geom name="wall" type="box" pos="0 0 1.0" size="0.14 2.2 1.0" rgba="0.55 0.35 0.2 1"/>
    <body name="base_link" pos="0 0 1.15">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="0.18 0.18 0.12" contype="1" conaffinity="1" rgba="0 0.85 0.25 0.55"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_settle_free_base_z_returns_none_when_base_xy_intersects_wall():
    """If the hull already clips non-floor geometry at the first height sample, settle must not fake success."""
    m = mujoco.MjModel.from_xml_string(MINIMAL_BASE_INSIDE_WALL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rb = molmospaces_spawn._bodies_descending_from(m, bid)  # noqa: SLF001
    zf = molmospaces_spawn.walkable_floor_z_at_xy(m, d, 0.0, 0.0, exclude_body_id=bid)
    assert zf is not None
    z_done = molmospaces_spawn.settle_free_base_z_to_floor(
        m,
        d,
        base_body_name="base_link",
        floor_geom_name="floor",
        x=0.0,
        y=0.0,
        z_floor=float(zf),
        z_start=float(zf) + 1.0,
        robot_bodies=rb,
        min_nonfloor_clearance=-5e-5,
    )
    assert z_done is None


MINIMAL_WALLED_ROOM_WITH_BASE = """<?xml version="1.0"?>
<mujoco model="walled_room">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.01" rgba="0.75 0.75 0.75 1"/>
    <geom name="w_n" type="box" pos="0 2.65 1.0" size="2.7 0.12 1.0" rgba="0.55 0.35 0.2 1"/>
    <geom name="w_s" type="box" pos="0 -2.65 1.0" size="2.7 0.12 1.0" rgba="0.55 0.35 0.2 1"/>
    <geom name="w_e" type="box" pos="2.65 0 1.0" size="0.12 2.7 1.0" rgba="0.55 0.35 0.2 1"/>
    <geom name="w_w" type="box" pos="-2.65 0 1.0" size="0.12 2.7 1.0" rgba="0.55 0.35 0.2 1"/>
    <body name="base_link" pos="0 0 0.9">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="0.22 0.22 0.14" contype="1" conaffinity="1" rgba="0 0.85 0.25 0.55"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_find_molmospaces_freejoint_xyz_finds_valid_pose_in_walled_room():
    """End-to-end spawn search must leave the base on the floor with acceptable non-floor clearance."""
    m = mujoco.MjModel.from_xml_string(MINIMAL_WALLED_ROOM_WITH_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
        m, d, base_body_name="base_link", min_nonfloor_clearance=-5e-5
    )
    assert out is not None
    x, y, z = out
    assert molmospaces_spawn.write_freejoint_base_xyzw(
        m, d, base_body_name="base_link", x=x, y=y, z=float(z)
    )
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rb = molmospaces_spawn._bodies_descending_from(m, bid)  # noqa: SLF001
    zf = molmospaces_spawn.walkable_floor_z_at_xy(m, d, x, y, exclude_body_id=bid)
    assert zf is not None
    worst = molmospaces_spawn.worst_robot_nonfloor_contact_dist(
        m, d, base_body_name="base_link", floor_geom_name="floor"
    )
    assert worst >= -0.002
    zb = molmospaces_spawn._min_robot_collision_geom_bottom_z(m, d, rb)  # noqa: SLF001
    assert zb is not None
    assert float(zb) <= float(zf) + 0.12
    assert float(zb) >= float(zf) - 0.06


@pytest.mark.skipif(os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1", reason="set RUN_MOLMOSPACES_TESTS=1")
def test_find_molmospaces_freejoint_on_installed_ithor_if_available():
    """With assets + emet_molmospaces: merged FloorPlan + rby1 must admit a floor-valid spawn."""
    try:
        from emet_molmospaces.runner import (
            _find_installed_scene_xml,
            _get_robot_mjcf_path,
            _merge_robot_into_scene,
        )
    except ImportError:
        pytest.skip("emet_molmospaces not installed in this environment")
    scene = _find_installed_scene_xml("ithor", 0)
    robot = _get_robot_mjcf_path("rby1")
    if scene is None or robot is None or not scene.is_file():
        pytest.skip("iTHOR FloorPlan1 or rby1 MJCF not on disk")
    merged = _merge_robot_into_scene(Path(scene), robot)
    try:
        m = mujoco.MjModel.from_xml_path(str(merged))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
            m, d, base_body_name="base_link", min_nonfloor_clearance=-5e-5
        )
        assert out is not None
        x, y, z = out
        assert molmospaces_spawn.write_freejoint_base_xyzw(
            m, d, base_body_name="base_link", x=x, y=y, z=float(z)
        )
        mujoco.mj_forward(m, d)
        worst = molmospaces_spawn.worst_robot_nonfloor_contact_dist(
            m, d, base_body_name="base_link", floor_geom_name="floor"
        )
        assert worst >= -0.003
        lines = molmospaces_spawn.format_spawn_floor_alignment_report(
            m, d, base_body_name="base_link", floor_geom_name="floor", xy=(float(x), float(y))
        )
        assert lines and "zb_minus_zfloor" in lines[0]
    finally:
        merged.unlink(missing_ok=True)
