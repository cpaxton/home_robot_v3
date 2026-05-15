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

"""Tests for :func:`emet.dataset.mujoco_gt.gt_objects_for_zmq_message`."""

from __future__ import annotations

import mujoco
import pytest

from emet.dataset.mujoco_gt import gt_objects_for_zmq_message

_MIN_XML = """<?xml version="1.0"?>
<mujoco model="gt_test">
  <worldbody>
    <body name="object_cube" pos="0.1 0 0.05">
      <geom name="g1" type="sphere" size="0.02"/>
    </body>
    <body name="base_link" pos="0 0 0.02">
      <geom type="sphere" size="0.01"/>
    </body>
    <body name="OBJ_thing" pos="-0.1 0 0.05">
      <geom type="sphere" size="0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def tiny_model_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_string(_MIN_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_gt_objects_default_kind_matches_object_glob(tiny_model_data) -> None:
    m, d = tiny_model_data
    out = gt_objects_for_zmq_message(m, d, environment=None)
    names = {x["name"] for x in out}
    assert "object_cube" in names
    assert "base_link" not in names


def test_gt_objects_molmospaces_includes_obj_pattern(tiny_model_data) -> None:
    m, d = tiny_model_data
    out = gt_objects_for_zmq_message(m, d, environment={"kind": "molmospaces"})
    names = {x["name"] for x in out}
    assert "object_cube" in names
    assert "OBJ_thing" in names
