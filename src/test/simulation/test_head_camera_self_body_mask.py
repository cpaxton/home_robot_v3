# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Head camera must not render the robot's own body into the mapping depth.

The sim head camera (d435i) renders the Stretch mast / head / camera housings into the depth
image.  The voxel map then treats the robot itself as a wall of obstacles around the base; in
tight scenes (Robocasa kitchens) this leaves no A* path with sufficient clearance so exploration
stalls with ``rejected_low_clearance`` for every frontier.  The ee camera (d405) keeps the robot
geoms so the gripper stays visible for manipulation.

This test builds a tiny hand-crafted model (no robosuite) so it stays fast and deterministic.
"""

from __future__ import annotations

import os

import mujoco
import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


_TINY_XML = """<mujoco model="robot_groups">
  <worldbody>
    <body name="base_link">
      <joint name="root" type="free"/>
      <geom name="robot_head" group="2" type="box" size="0.1 0.1 0.1" pos="0 0 0.5"/>
      <geom name="robot_mast" group="3" type="box" size="0.05 0.05 0.4" pos="0 0 0.2"/>
      <body name="child">
        <geom name="robot_gripper" group="2" type="box" size="0.05 0.05 0.05" pos="0.2 0 0.6"/>
      </body>
    </body>
    <geom name="scene_counter" group="0" type="box" size="0.5 0.1 0.5" pos="1 0 0.4"/>
    <geom name="scene_floor" group="1" type="plane" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""


def _tiny_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_TINY_XML)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
def test_robot_body_geom_groups_detected():
    from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import robot_body_geom_groups

    model = _tiny_model()
    # Robot bodies: base_link + child -> geoms on groups 2 and 3.
    assert robot_body_geom_groups(model) == {2, 3}


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
def test_head_camera_geomgroup_mask_hides_robot_only():
    """Head-camera mask clears robot geom groups and keeps scene groups (0/1) on."""
    from emet.simulation.stretch_mujoco.mujoco_server_camera_manager import (
        head_camera_geomgroup_mask,
    )

    model = _tiny_model()
    mask = head_camera_geomgroup_mask(model)
    assert mask.shape == (6,)
    # Robot groups hidden.
    assert int(mask[2]) == 0 and int(mask[3]) == 0
    # Scene groups stay rendered.
    assert int(mask[0]) == 1 and int(mask[1]) == 1


def test_robosuite_primary_camera_applies_self_body_geom_mask():
    """RobosuiteZmqServer must mask robot geoms for the primary mapping camera only."""
    from types import SimpleNamespace

    import numpy as np

    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    model = _tiny_model()
    spec = Rby1Backend().get_spec()
    server = RobosuiteZmqServer.__new__(RobosuiteZmqServer)
    server._spec = spec
    server._mjmodel = model

    scene_option = SimpleNamespace(geomgroup=np.ones(6, dtype=np.uint8))
    renderer = SimpleNamespace(_scene_option=scene_option)

    server._configure_renderer_geomgroups_for_camera(renderer, spec.camera_names[0])
    assert scene_option.geomgroup.tolist() == [1, 1, 0, 0, 1, 1]

    server._configure_renderer_geomgroups_for_camera(renderer, spec.camera_names[1])
    assert scene_option.geomgroup.tolist() == [1, 1, 1, 1, 1, 1]
