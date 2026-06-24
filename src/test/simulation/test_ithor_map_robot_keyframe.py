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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""iTHOR occupancy map must compile merged scene+robot MJCF (robot keyframes, MjSpec pass 2)."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import pytest

from emet.robots.xlerobot import XLeRobotBackend
from emet.simulation.molmo_occupancy.ithor_map import (
    _safe_model_data,
    _strip_spec_keyframes,
    iTHORMap,
)

# Full iTHORMap builds an OpenGL context; run only when explicitly enabled (CI / local GPU).
_ithor_gl = pytest.mark.skipif(
    os.environ.get("RUN_ITHOR_MAP_GL_TESTS", "") != "1",
    reason="Set RUN_ITHOR_MAP_GL_TESTS=1 to run iTHORMap OpenGL integration tests",
)


def _write_minimal_merged_mjcf(tmp_path: Path) -> Path:
    """Scene floor + clutter body + included robot with ``xlerobot_home`` keyframe."""
    robot = tmp_path / "robot.xml"
    robot.write_text(
        """<mujoco>
  <worldbody>
    <body name="chassis">
      <joint name="slide_joint_x" type="slide" axis="1 0 0"/>
      <joint name="slide_joint_y" type="slide" axis="0 1 0"/>
      <joint name="hinge_joint_z" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.12" mass="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="xlerobot_home" qpos="0 0 0"/>
  </keyframe>
</mujoco>
"""
    )
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.1" contype="0" conaffinity="0"/>
    <body name="cabinet_0_0">
      <geom type="box" size="0.25 0.25 0.6" contype="0" conaffinity="0"/>
    </body>
    <body name="ceiling_light">
      <geom type="box" size="0.1 0.1 0.05" contype="0"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="scene_home" qpos="0"/>
  </keyframe>
</mujoco>
"""
    )
    merged = tmp_path / "merged.xml"
    merged.write_text(
        f"""<?xml version="1.0"?>
<mujoco model="test_merged">
  <include file="{scene.resolve()}"/>
  <include file="{robot.resolve()}"/>
</mujoco>
"""
    )
    return merged


def test_merged_mjcf_loads_single_robot_home_key(tmp_path):
    merged = _write_minimal_merged_mjcf(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(merged))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
    assert "xlerobot_home" in names
    assert names.count("xlerobot_home") == 1


def test_mjspec_pass2_survives_robot_keyframe_after_body_deletes(tmp_path):
    """Regression: pass 2 used to fail with repeated robot key after ``spec.delete(body)``."""
    merged = _write_minimal_merged_mjcf(tmp_path)
    spec = mujoco.MjSpec.from_file(str(merged))
    for body in list(spec.worldbody.bodies):
        if body.name == "chassis":
            spec.delete(body)
            break
    _strip_spec_keyframes(spec)
    for body in list(spec.worldbody.bodies):
        name = body.name or ""
        if "cabinet" in name or "ceiling" in name:
            spec.delete(body)
    _strip_spec_keyframes(spec)
    model, _data = _safe_model_data(spec)
    assert model.ngeom >= 1


@_ithor_gl
def test_ithor_map_from_xlerobot_merged_wrapper(tmp_path):
    merged = _write_minimal_merged_mjcf(tmp_path)
    th = iTHORMap.from_mj_model_path(str(merged), robot_root_body_name="chassis", px_per_m=80)
    fp = th.get_free_points()
    assert fp.size >= 1


@_ithor_gl
def test_ithor_occupancy_priority_xy_does_not_skip_robot_keyframe(tmp_path):
    from emet.simulation.scene_base_spawn import _ithor_occupancy_priority_xy

    merged = _write_minimal_merged_mjcf(tmp_path)
    occ_xy, _map = _ithor_occupancy_priority_xy(
        str(merged),
        {"scene": "ithor"},
        robot_root_body_name="chassis",
        px_per_m=80,
        max_points=200,
    )
    assert len(occ_xy) > 0


@_ithor_gl
@pytest.mark.parametrize(
    "merged_path",
    sorted(
        (Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "xlerobot").glob(
            "molmospaces_merged_*.xml"
        )
    ),
)
def test_cached_xlerobot_merged_mjcf_ithor_map_smoke(merged_path: Path):
    """Smoke cached merge wrappers checked into or generated beside xlerobot assets."""
    if not merged_path.is_file():
        pytest.skip("no cached merge")
    spec = XLeRobotBackend().get_spec()
    model = mujoco.MjModel.from_xml_path(str(merged_path))
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "xlerobot_home")
    assert kid >= 0, "merged xlerobot MJCF should expose xlerobot_home keyframe"
    th = iTHORMap.from_mj_model_path(str(merged_path), robot_root_body_name=spec.base_link_name, px_per_m=100)
    assert len(th.get_free_points()) > 0
