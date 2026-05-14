# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

from pathlib import Path

import emet


def test_molmospaces_spawn_json_exists_for_merge_robots():
    root = Path(emet.__file__).resolve().parent / "assets" / "robot"
    assert (root / "galaxea_r1" / "molmospaces_spawn.json").is_file()
    assert (root / "innate_mars" / "molmospaces_spawn.json").is_file()


def test_load_molmospaces_spawn_metadata_for_rby1_family():
    from emet.simulation.molmospaces_spawn_metadata import load_molmospaces_spawn_metadata

    m = load_molmospaces_spawn_metadata("rby1")
    assert m is not None
    assert m.schema_version == 1


def test_load_molmospaces_spawn_metadata_galaxea_foot_clearance():
    from emet.simulation.molmospaces_spawn_metadata import load_molmospaces_spawn_metadata

    m = load_molmospaces_spawn_metadata("galaxea_r1")
    assert m is not None
    assert m.molmospaces_target_foot_clearance_above_floor_m == 0.02


def test_load_molmospaces_spawn_metadata_unknown_robot():
    from emet.simulation.molmospaces_spawn_metadata import load_molmospaces_spawn_metadata

    assert load_molmospaces_spawn_metadata("not_a_real_robot_id") is None
