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
import numpy as np
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


def test_clamp_xy_into_rect_and_corner_reach():
    from emet.simulation.molmospaces_spawn import (
        _clamp_xy_into_rect,
        _max_xy_distance_to_rect_corners,
    )

    rect = (2.0, 8.0, 4.0, 10.0)
    assert _clamp_xy_into_rect(5.0, 3.0, rect) == (5.0, 4.0)
    ox, oy = _clamp_xy_into_rect(-10.0, 20.0, rect)
    assert (ox, oy) == (2.0, 10.0)
    r = _max_xy_distance_to_rect_corners(5.0, 7.0, rect)
    assert abs(r - math.hypot(3.0, 3.0)) < 1e-6


MEGA_SHELL_OFFSET = """<?xml version="1.0"?>
<mujoco model="mega_shell">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="80 80 0.01" rgba="0.72 0.72 0.72 1"/>
    <!-- Wide deck: geom_rbound >> 15 so clip must cap (not skip). Ceiling gives upward-ray clearance. -->
    <geom name="deck" type="box" pos="18 14 0.06" size="24 24 0.06" contype="1" conaffinity="1" rgba="0.55 0.5 0.45 0.4"/>
    <geom name="ceiling" type="box" pos="18 14 2.75" size="26 26 0.1" contype="1" conaffinity="1" rgba="0.55 0.55 0.6 0.2"/>
    <body name="base_link" pos="0 0 0.95">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="0.22 0.22 0.14" contype="1" conaffinity="1" rgba="0 0.85 0.25 0.55"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_collision_clip_and_centroid_use_capped_rbound_for_mega_mesh():
    """Regression: skipping geoms with rb > max_geom_rbound left no clip; search fell back to (0,0) outside the shell."""
    from emet.simulation.molmospaces_spawn import collision_scene_xy_clip_rect, scene_collision_centroid_xy

    m = mujoco.MjModel.from_xml_string(MEGA_SHELL_OFFSET)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    robot_bodies = molmospaces_spawn._bodies_descending_from(m, bid)  # noqa: SLF001
    clip = collision_scene_xy_clip_rect(
        m, d, robot_bodies=robot_bodies, floor_geom_name="floor", margin=0.0, max_geom_rbound=15.0
    )
    assert clip is not None
    xmin, xmax, ymin, ymax = clip
    assert xmin <= 18.0 <= xmax and ymin <= 14.0 <= ymax
    c = scene_collision_centroid_xy(m, d, robot_bodies=robot_bodies, max_geom_rbound=15.0)
    assert c is not None
    cx, cy = c
    assert abs(cx - 18.0) < 2.5 and abs(cy - 14.0) < 2.5


def test_find_molmospaces_moves_base_into_offset_mega_shell():
    m = mujoco.MjModel.from_xml_string(MEGA_SHELL_OFFSET)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
        m, d, base_body_name="base_link", min_nonfloor_clearance=-5e-5
    )
    assert out is not None
    x, y, _z = out
    assert math.hypot(x - 18.0, y - 14.0) < 12.0, f"expected spawn near house center (18,14), got ({x},{y})"


def test_mega_shell_placed_passes_horizontal_and_optional_upward_ceiling_gate():
    """Synthetic shell has a ceiling geom; optional gate asserts +z ray hit with clearance."""
    m = mujoco.MjModel.from_xml_string(MEGA_SHELL_OFFSET)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
        m, d, base_body_name="base_link", min_nonfloor_clearance=-5e-5
    )
    assert out is not None
    assert molmospaces_spawn.molmospaces_placed_pose_passes_horizontal_interior_gate(
        m, d, base_body_name="base_link", placed=out
    )
    assert molmospaces_spawn.molmospaces_placed_pose_passes_horizontal_interior_gate(
        m,
        d,
        base_body_name="base_link",
        placed=out,
        require_upward_ceiling_hit=True,
        min_upward_clearance_m=0.03,
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


def test_horizontal_spawn_rejects_exterior_tongue_infinite_plane_open_horizon():
    from emet.simulation.molmospaces_spawn import horizontal_spawn_rejects_exterior_tongue

    xml = """<?xml version="1.0"?>
<mujoco model="plane">
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.01"/>
    <body name="base_link" pos="0 0 1.5">
      <freejoint name="root"/>
      <geom name="hull" type="sphere" size="0.02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert bid >= 0
    assert horizontal_spawn_rejects_exterior_tongue(m, d, 0.0, 0.0, 0.08, exclude_body_id=int(bid))


def test_horizontal_spawn_rejects_exterior_tongue_false_center_of_walled_room():
    from emet.simulation.molmospaces_spawn import horizontal_spawn_rejects_exterior_tongue

    m = mujoco.MjModel.from_xml_string(MINIMAL_WALLED_ROOM_WITH_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert bid >= 0
    assert not horizontal_spawn_rejects_exterior_tongue(m, d, 0.0, 0.0, 0.08, exclude_body_id=int(bid))


def test_restore_freejoint_base_from_model_qpos0_after_spawn_search_hoist():
    """Failed spawn search hoists the base to high z; restore must put qpos back to qpos0."""
    m = mujoco.MjModel.from_xml_string(MINIMAL_WALLED_ROOM_WITH_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    assert bid >= 0
    qadr = None
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j]) == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = int(m.jnt_qposadr[j])
            break
    assert qadr is not None
    assert molmospaces_spawn.write_freejoint_base_xyzw(
        m, d, base_body_name="base_link", x=1.0, y=-2.0, z=88.0
    )
    mujoco.mj_forward(m, d)
    assert molmospaces_spawn.restore_freejoint_base_from_model_qpos0(m, d, base_body_name="base_link")
    np.testing.assert_allclose(d.qpos[qadr : qadr + 7], m.qpos0[qadr : qadr + 7], rtol=0, atol=1e-9)


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


def test_find_molmospaces_placed_passes_horizontal_interior_gate_minimal_room():
    """After spawn, re-run the horizontal exterior-tongue rule at the placed XY.

    This MJCF has no ceiling; we only assert the horizontal gate (default), not overhead rays.
    """
    m = mujoco.MjModel.from_xml_string(MINIMAL_WALLED_ROOM_WITH_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
        m, d, base_body_name="base_link", min_nonfloor_clearance=-5e-5
    )
    assert out is not None
    assert molmospaces_spawn.molmospaces_placed_pose_passes_horizontal_interior_gate(
        m, d, base_body_name="base_link", placed=out
    )


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
        assert molmospaces_spawn.molmospaces_placed_pose_passes_horizontal_interior_gate(
            m, d, base_body_name="base_link", placed=out
        )
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


@pytest.mark.skipif(os.environ.get("RUN_MOLMOSPACES_TESTS", "") != "1", reason="set RUN_MOLMOSPACES_TESTS=1")
def test_merged_ithor_index3_spawn_passes_horizontal_interior_gate():
    """Regression: iTHOR train index 3 merge must spawn and pass horizontal tongue post-check.

    Many iTHOR scenes omit a ceiling geom; we do not require ``require_upward_ceiling_hit`` here.
    """
    try:
        from emet_molmospaces.runner import (
            _find_installed_scene_xml,
            _get_robot_mjcf_path,
            _merge_robot_into_scene,
        )
    except ImportError:
        pytest.skip("emet_molmospaces not installed in this environment")
    scene = _find_installed_scene_xml("ithor", 3)
    robot = _get_robot_mjcf_path("rby1")
    if scene is None or robot is None or not scene.is_file():
        pytest.skip("iTHOR scene index 3 or rby1 MJCF not on disk")
    merged = _merge_robot_into_scene(Path(scene), robot)
    try:
        m = mujoco.MjModel.from_xml_path(str(merged))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        env = {"kind": "molmospaces", "scene": "ithor", "split": "train", "index": 3}
        out = molmospaces_spawn.find_molmospaces_freejoint_xyz(
            m,
            d,
            base_body_name="base_link",
            min_nonfloor_clearance=-5e-5,
            merged_mjcf_path=str(merged),
            environment=env,
        )
        assert out is not None, "find_molmospaces_freejoint_xyz returned None for iTHOR index 3 merge"
        assert molmospaces_spawn.molmospaces_placed_pose_passes_horizontal_interior_gate(
            m, d, base_body_name="base_link", placed=out
        ), f"placed XY failed horizontal interior gate: {out!r}"
    finally:
        merged.unlink(missing_ok=True)
