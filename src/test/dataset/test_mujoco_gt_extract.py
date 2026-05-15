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

import mujoco
import numpy as np

from emet.dataset.graph_blob import gt_object_dicts_to_graph_blob
from emet.dataset.mujoco_gt import extract_gt_object_dicts
from emet.dataset.schema import object_record_from_dict


def test_extract_object_star_bodies() -> None:
    xml = """
    <mujoco model="tiny">
      <worldbody>
        <body name="object1" pos="1 0 0.1">
          <geom name="g1" type="sphere" size="0.05"/>
        </body>
        <body name="object2" pos="-0.5 0.2 0">
          <geom type="box" size="0.1 0.05 0.02"/>
        </body>
        <body name="table_top" pos="0 0 0.4">
          <geom type="box" size="0.5 0.3 0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objs = extract_gt_object_dicts(model, data)
    names = {o["body_name"] for o in objs}
    assert names == {"object1", "object2"}
    r0 = object_record_from_dict(objs[0])
    assert np.allclose(r0.pos_xyz, (1.0, 0.0, 0.1), atol=1e-5)
    assert len(r0.quat_wxyz) == 4


def test_allowlist_overrides_globs() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="foo_special" pos="0 0 1">
          <geom type="sphere" size="0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objs = extract_gt_object_dicts(model, data, allowlist=["foo_special"], name_globs=("object*",))
    assert len(objs) == 1
    assert objs[0]["body_name"] == "foo_special"


def test_gt_graph_blob_has_floor_and_objects() -> None:
    blob = gt_object_dicts_to_graph_blob(
        [
            {"body_name": "object1", "name": "object1", "pos_xyz": [1, 2, 3], "quat_wxyz": [1, 0, 0, 0]},
        ]
    )
    assert len(blob.nodes) == 2
    assert blob.nodes[0].labels == ["floor"]
    assert blob.edges == []
