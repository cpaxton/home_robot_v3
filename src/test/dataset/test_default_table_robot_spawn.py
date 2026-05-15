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
import pytest

from emet.robots import get_robot_spec
from emet.simulation.mujoco_ctrl_sync import stabilize_physics_inplace
from emet.simulation.mujoco_server import _load_default_scene_with_robot


@pytest.mark.timeout(60)
def test_default_table_spawn_lifts_base_above_floor() -> None:
    model = _load_default_scene_with_robot("rby1")
    if model is None:
        pytest.skip("default merged scene not available (missing assets)")
    data = mujoco.MjData(model)
    np.copyto(data.qpos, model.qpos0)
    mujoco.mj_forward(model, data)
    zb = float(data.body("base_link").xpos[2])
    assert zb > 0.05, "base should sit visibly above the floor plane"


@pytest.mark.timeout(60)
def test_default_table_stretch_merged_model_loads() -> None:
    model = _load_default_scene_with_robot("stretch")
    if model is None:
        pytest.skip("default merged scene not available (missing assets)")
    data = mujoco.MjData(model)
    np.copyto(data.qpos, model.qpos0)
    mujoco.mj_forward(model, data)
    zb = float(data.body("base_link").xpos[2])
    assert zb > 0.05, "base should sit visibly above the floor plane"


@pytest.mark.timeout(120)
def test_default_table_rby1_base_height_stable_under_server_like_steps() -> None:
    """``mj_forward``-only tests miss PD/contact bounce; idle ``mj_step`` must stay bounded."""
    model = _load_default_scene_with_robot("rby1")
    if model is None:
        pytest.skip("default merged scene not available (missing assets)")
    spec = get_robot_spec("rby1")
    assert spec is not None

    data = mujoco.MjData(model)
    np.copyto(data.qpos, model.qpos0)
    stabilize_physics_inplace(model, data, spec, n_steps=48)

    zs: list[float] = []
    for _ in range(200):
        mujoco.mj_step(model, data)
        zs.append(float(data.body("base_link").xpos[2]))

    tail = np.asarray(zs[-120:], dtype=np.float64)
    p2p = float(np.max(tail) - np.min(tail))
    dz = float(np.max(np.abs(np.diff(tail))))
    assert p2p < 0.12, f"base height peak-to-peak too large over tail: {p2p}"
    assert dz < 0.04, f"single-step base |dz| too large in tail: {dz}"
