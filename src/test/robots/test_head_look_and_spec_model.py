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
# This source code is licensed under the LICENSE file in the
# root directory of this source tree.

"""Tests for non-Stretch ``head_to`` mapping and spec-based :class:`SpecRobotModel`."""

from unittest.mock import MagicMock, patch

import numpy as np

from emet.robots.galaxea_r1 import GalaxeaR1Backend
from emet.robots.spec_robot_model import SpecRobotModel
from emet.simulation.head_look_action import apply_head_to_robosuite


def test_apply_head_to_r1_sets_torso_when_no_head():
    spec = GalaxeaR1Backend().get_spec()
    model = MagicMock()
    data = MagicMock()
    data.ctrl = np.zeros(32, dtype=np.float64)

    def fake_name2id(model_arg, kind, name):
        if name == "head_pan":
            return -1
        if name == "head_tilt":
            return -1
        if name == "torso1":
            return 6
        if name == "torso4":
            return 7
        return -1

    with patch("emet.simulation.head_look_action.mujoco.mj_name2id", side_effect=fake_name2id):
        model.actuator_ctrlrange = np.array(
            [
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [-1.5, 1.5],
                [-3.2, 3.2],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
                [-1, 1],
            ]
        )

        n = apply_head_to_robosuite(spec, model, data, 0.5, -0.3)
    assert n == 2
    assert abs(float(data.ctrl[6]) - (-0.3)) < 1e-6
    assert abs(float(data.ctrl[7]) - 0.25) < 1e-6


def test_spec_robot_model_footprint_from_spec():
    spec = GalaxeaR1Backend().get_spec()
    m = SpecRobotModel(spec)
    assert m.get_dof() == spec.dof
    assert m.get_footprint() is spec.footprint
    m.set_config(np.zeros(3))
    assert isinstance(m.get_config(), np.ndarray)
