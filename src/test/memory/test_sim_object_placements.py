# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for sim_object_placements helpers."""

from __future__ import annotations

import numpy as np

from emet.simulation.sim_object_placements import (
    DEFAULT_TABLE_SCENE_PLACEMENTS,
    SIM_OBJECT_PLACEMENTS_FRAME,
    apply_navigation_origin_to_session,
    assert_default_table_gt,
    attach_sim_object_placements_to_session,
    build_sim_object_placements_for_session,
    is_default_table_environment,
    overlay_live_mujoco_body_poses,
    placements_from_objects_info,
    placements_to_session_dict,
)
from emet.utils.geometry import nav_xyt_to_world_xyt


def test_placements_to_session_dict_json_safe():
    raw = {
        "apple_main": {
            "cat": "apple",
            "pos": np.array([0.1, -0.5, 0.9]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        "_emet_spawn_hint_xyt": [0.0, 0.0, 0.0],
    }
    out = placements_to_session_dict(raw)
    assert out is not None
    assert "_emet_spawn_hint_xyt" not in out
    assert out["apple_main"]["pos"] == [0.1, -0.5, 0.9]


def test_is_default_table_environment():
    assert is_default_table_environment(None)
    assert is_default_table_environment("stretch_default_scene")
    assert is_default_table_environment("default_table")
    assert not is_default_table_environment("molmospaces")
    assert not is_default_table_environment("robocasa")


def test_default_table_scene_when_no_wizard():
    for kind in ("stretch_default_scene", "default_table", None):
        out = build_sim_object_placements_for_session(
            objects_info=None,
            environment_kind=kind,
            model=None,
            data=None,
        )
        assert out is not None
        assert_default_table_gt(out)


def test_assert_default_table_gt():
    assert_default_table_gt(placements_to_session_dict(DEFAULT_TABLE_SCENE_PLACEMENTS))


def test_robocasa_wizard_priority():
    wizard = placements_from_objects_info(
        {
            "mug_main": {"cat": "mug", "pos": [0.0, 0.0, 1.0], "quat": [1, 0, 0, 0]},
            "_emet_spawn_hint_xyt": [1, 2, 0.5],
        }
    )
    assert wizard is not None
    assert "mug_main" in wizard
    assert wizard["mug_main"]["cat"] == "mug"


def test_robocasa_wizard_placements_reach_session_shape():
    wizard = {
        "obj_main": {"cat": "garlic", "pos": [3.69, -0.41, 0.96], "quat": [1, 0, 0, 0.08]},
        "distr_counter_main": {"cat": "cinnamon", "pos": [4.11, -0.09, 1.0], "quat": [0.99, 0, 0, 0.13]},
    }
    out = build_sim_object_placements_for_session(
        objects_info=wizard,
        environment_kind="robocasa",
        model=None,
        data=None,
    )
    assert out is not None
    assert "obj_main" in out
    assert out["obj_main"]["cat"] == "garlic"


def test_overlay_live_mujoco_body_poses():
    import mujoco

    xml = """
    <mujoco><worldbody>
      <body name="object1" pos="1.0 2.0 0.5"><geom type="sphere" size="0.04"/></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    static = {"object1": {"cat": "cube", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}}
    out = overlay_live_mujoco_body_poses(static, model, data)
    assert out is not None
    pos = out["object1"]["pos"]
    assert abs(pos[0] - 1.0) < 1e-4
    assert abs(pos[1] - 2.0) < 1e-4


def test_attach_sim_object_placements_to_session():
    session: dict = {"schema_version": 1}
    attach_sim_object_placements_to_session(
        session,
        objects_info=None,
        environment_kind="default_table",
        model=None,
        data=None,
    )
    assert "sim_object_placements" in session
    assert session.get("sim_object_placements_frame") == SIM_OBJECT_PLACEMENTS_FRAME
    assert_default_table_gt(session["sim_object_placements"])


def test_default_table_constants_match_environment_xml():
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["object2"]["cat"] == "red cylinder"
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["object1"]["cat"] == "blue cube"
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["table"]["cat"] == "table"


def test_apply_navigation_origin_to_session_sets_note():
    session: dict = {"sim_object_placements": {"mug": {"cat": "mug", "pos": [1.0, 2.0, 0.9]}}}
    apply_navigation_origin_to_session(session, [3.5, -0.6, 1.2])
    assert session["navigation_origin_xyt"] == [3.5, -0.6, 1.2]
    assert "navigation_origin_xyt" in session["sim_object_placements_note"]


def test_nav_origin_composes_spawn_gps_to_gt_world_frame():
    """Stretch-style session: GT world XYZ + nav gps at spawn → same kitchen frame as Rerun."""
    origin = np.array([3.52, -0.58, 0.0])
    session = {
        "navigation_origin_xyt": origin.tolist(),
        "sim_object_placements": {
            "garlic_main": {"cat": "garlic", "pos": [3.69, -0.41, 1.02]},
        },
    }
    spawn_gps = np.array([0.0, 0.0, 0.0])
    world_xyt = nav_xyt_to_world_xyt(spawn_gps, session)
    np.testing.assert_allclose(world_xyt[:2], origin[:2], atol=1e-6)
    gt_xy = np.asarray(session["sim_object_placements"]["garlic_main"]["pos"][:2])
    dist = float(np.linalg.norm(gt_xy - world_xyt[:2]))
    assert 0.05 < dist < 1.5


def test_fixture_group_scan_merges_wizard_and_fixtures():
    import mujoco

    from emet.simulation.sim_object_placements import (
        build_sim_object_placements_for_session,
        fixture_group_key,
    )

    assert fixture_group_key("sink_main_group_basin_main") == "sink"
    assert fixture_group_key("obj_main") == "obj_main"
    assert fixture_group_key("wall_left_room_main") is None

    xml = """
    <mujoco><worldbody>
      <body name="sink_main_group_basin_main" pos="1 0 0.8">
        <geom type="box" size="0.2 0.15 0.05"/>
      </body>
      <body name="counter_main_main_group_top_main" pos="0 0 0.9">
        <geom type="box" size="0.5 0.3 0.02"/>
      </body>
      <body name="obj_main" pos="0.1 0 0.95">
        <geom type="sphere" size="0.03"/>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    wizard = {"obj_main": {"cat": "garlic", "pos": [0.1, 0.0, 0.95], "quat": [1, 0, 0, 0]}}
    out = build_sim_object_placements_for_session(
        objects_info=wizard,
        environment_kind="robocasa",
        model=model,
        data=data,
    )
    assert out is not None
    assert "sink" in out
    assert "counter_main" in out
    assert out["obj_main"]["cat"] == "garlic"
    assert "bounds" in out["sink"]
    assert "bounds" in out["counter_main"]


def test_fixture_bounds_exclude_robocasa_anchor_geoms():
    import mujoco

    from emet.simulation.sim_object_placements import build_sim_object_placements_for_session

    # Robocasa counters include ``*_main_group_base_*`` geoms offset ~10 m from the body origin.
    xml = """
    <mujoco><worldbody>
      <body name="counter_main_main_group_main" pos="0 0 0.46">
        <geom name="counter_main_main_group_base_front" type="box" pos="0 0 10" size="0.01 0.01 0.01"/>
        <geom name="counter_main_main_group_top_0" type="box" pos="0 0 0.45" size="0.5 0.3 0.02"/>
      </body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    out = build_sim_object_placements_for_session(
        objects_info=None,
        environment_kind="robocasa",
        model=model,
        data=data,
    )
    assert out is not None and "counter_main" in out
    bounds = np.asarray(out["counter_main"]["bounds"], dtype=np.float64)
    assert bounds[1, 2] - bounds[0, 2] < 0.2
    assert bounds[1, 2] < 1.5


def test_mujoco_model_data_for_gt_scan_stretch_model_attr():
    import mujoco

    from emet.simulation.sim_object_placements import (
        build_sim_object_placements_for_session,
        mujoco_model_data_for_gt_scan,
    )

    xml = """
    <mujoco><worldbody>
      <body name="sink_main_group_basin_main"><geom type="box" size="0.2 0.15 0.05"/></body>
      <body name="obj_main"><geom type="sphere" size="0.03"/></body>
    </worldbody></mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)

    class _StretchLikeSim:
        pass

    _StretchLikeSim.model = model

    m, d = mujoco_model_data_for_gt_scan(_StretchLikeSim())
    assert m is model and d is not None
    wizard = {"obj_main": {"cat": "mug", "pos": [0, 0, 1], "quat": [1, 0, 0, 0]}}
    out = build_sim_object_placements_for_session(
        objects_info=wizard,
        environment_kind="robocasa",
        model=m,
        data=d,
    )
    assert out is not None
    assert "sink" in out
    assert "obj_main" in out
