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
    assert_default_table_gt,
    attach_sim_object_placements_to_session,
    build_sim_object_placements_for_session,
    is_default_table_environment,
    placements_from_objects_info,
    placements_to_session_dict,
)


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
    assert_default_table_gt(session["sim_object_placements"])


def test_default_table_constants_match_environment_xml():
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["object2"]["cat"] == "red cylinder"
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["object1"]["cat"] == "blue cube"
    assert DEFAULT_TABLE_SCENE_PLACEMENTS["table"]["cat"] == "table"
