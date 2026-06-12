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

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from emet.simulation.sim_manipulation import set_free_body_pose


def test_set_free_body_pose_moves_object2():
    scene = Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "scene_environment.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    before = np.array(data.body("object2").xpos, dtype=np.float64).copy()
    target = before + np.array([0.0, 0.0, 0.12])
    assert set_free_body_pose(model, data, "object2", target)
    after = np.array(data.body("object2").xpos, dtype=np.float64)
    assert np.linalg.norm(after - target) < 1e-5
