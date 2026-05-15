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

"""Tests for :mod:`emet.simulation.scene_resolution`."""

from __future__ import annotations

from pathlib import Path

from emet.simulation.scene_resolution import (
    build_zmq_environment,
    resolve_merged_physics_scene,
    scene_source_basename_from_path,
    spawn_scene_disk_path,
)


def test_build_zmq_environment_molmospaces() -> None:
    d = build_zmq_environment(
        molmospaces_session_scene="ithor",
        molmospaces_session_split="val",
        molmospaces_session_index=3,
        use_robocasa=False,
        robocasa_task="PickPlaceCounterToCabinet",
        robocasa_style=1,
        robocasa_layout=1,
    )
    assert d is not None
    assert d["kind"] == "molmospaces"
    assert d["scene"] == "ithor"
    assert d["split"] == "val"
    assert d["index"] == 3


def test_build_zmq_environment_robocasa_when_no_molmo() -> None:
    d = build_zmq_environment(
        molmospaces_session_scene=None,
        molmospaces_session_split=None,
        molmospaces_session_index=None,
        use_robocasa=True,
        robocasa_task="OpenCabinet",
        robocasa_style=2,
        robocasa_layout=3,
    )
    assert d is not None
    assert d["kind"] == "robocasa"
    assert d["task"] == "OpenCabinet"
    assert d["style"] == 2
    assert d["layout"] == 3


def test_build_zmq_environment_none_when_default() -> None:
    assert (
        build_zmq_environment(
            molmospaces_session_scene=None,
            molmospaces_session_split=None,
            molmospaces_session_index=None,
            use_robocasa=False,
            robocasa_task="t",
            robocasa_style=1,
            robocasa_layout=1,
        )
        is None
    )


def test_scene_source_basename_from_path() -> None:
    assert scene_source_basename_from_path("/a/b/foo.xml") == "foo.xml"
    assert scene_source_basename_from_path(None) is None


def test_spawn_scene_disk_path(tmp_path: Path) -> None:
    p = tmp_path / "x.xml"
    p.write_text("<mujoco/>")
    assert spawn_scene_disk_path(str(p)) == str(p.resolve())


def test_resolve_merged_physics_scene_robocasa_passes_wizard_payload() -> None:
    sentinel = object()
    loaded = resolve_merged_physics_scene(
        robot_key="rby1",
        scene_path=None,
        use_robocasa=True,
        wizard_scene_model=sentinel,
        wizard_scene_xml="<mujoco/>",
        wizard_objects_info={"objects": []},
        zmq_environment={"kind": "robocasa"},
        scene_source_basename="kitchen.xml",
    )
    assert loaded.scene_model is sentinel
    assert loaded.scene_xml == "<mujoco/>"
    assert loaded.objects_info == {"objects": []}
    assert loaded.zmq_environment == {"kind": "robocasa"}
