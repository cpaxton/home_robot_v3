# Copyright (c) Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import mujoco

from emet.simulation.mujoco_ground_truth import collect_body_world_rows, subtree_body_ids

_MINI_XML = """<mujoco model="tiny">
  <worldbody>
    <body name="floor"><geom type="plane" size="2 2 0.1"/></body>
    <body name="gadget" pos="1 2 0.4"><geom type="sphere" size="0.08"/></body>
    <body name="arm_base" pos="0 0 0.15">
      <geom type="box" size="0.1 0.1 0.05"/>
      <body name="arm_link"><geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.03"/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_subtree_and_collect_rows():
    model = mujoco.MjModel.from_xml_string(_MINI_XML)
    data = mujoco.MjData(model)
    subtree = subtree_body_ids(model, "arm_base")
    for nm in ("arm_base", "arm_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        assert bid >= 0
        assert bid in subtree

    gadgets = collect_body_world_rows(model, data, exclude_body_ids=subtree)
    names = [r.name for r in gadgets]
    assert "gadget" in names
    assert "floor" in names
    assert "arm_base" not in names and "arm_link" not in names
