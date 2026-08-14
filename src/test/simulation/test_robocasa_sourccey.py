# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""Sourccey in RoboCasa: strip-replace wizard builds a kitchen model with the vendored MJCF."""

from __future__ import annotations

import os

import numpy as np
import pytest

_run_sim = os.environ.get("RUN_SIM_TESTS", "1").strip().lower()
RUN_SIM_TESTS = _run_sim not in ("0", "false", "no", "off")


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_robocasa_sourccey_wizard_uses_strip_replace():
    pytest.importorskip("mujoco")
    from emet.simulation.stretch_mujoco import robocasa_gen

    assert robocasa_gen._uses_strip_placeholder_robot("sourccey")
    assert robocasa_gen._robosuite_robot_for("sourccey") == "PandaMobile"

    model, _xml, objects_info = robocasa_gen.model_generation_wizard(
        task="PickPlaceCounterToCabinet", layout=1, style=1, robot="sourccey", seed=0
    )
    assert model is not None
    # vendored sourccey MJCF joints must be present in the merged kitchen model
    for jname in ("base_x", "base_y", "base_yaw", "lift", "left_shoulder_pan", "right_shoulder_pan"):
        assert mujoco_name_id(model, "JOINT", jname) >= 0, f"missing joint {jname} after strip-replace"


def mujoco_name_id(model, objtype: str, name: str) -> int:
    import mujoco

    return mujoco.mj_name2id(model, getattr(mujoco.mjtObj, f"mjOBJ_{objtype}"), name)


@pytest.mark.skipif(not RUN_SIM_TESTS, reason="RUN_SIM_TESTS=0")
@pytest.mark.timeout(180)
def test_robocasa_sourccey_planar_spawn_hint_sane():
    pytest.importorskip("mujoco")
    from emet.simulation.stretch_mujoco import robocasa_gen

    model, _xml, objects_info = robocasa_gen.model_generation_wizard(
        task="PickPlaceCounterToCabinet", layout=1, style=1, robot="sourccey", seed=0
    )
    hint = np.asarray(objects_info.get("_emet_spawn_hint_xyt", [0, 0, 0]), dtype=np.float64).reshape(-1)[:3]
    assert np.isfinite(hint).all()
    # hint should be somewhere in the (walkable) kitchen, not origin
    assert float(np.hypot(hint[0], hint[1])) > 0.1
