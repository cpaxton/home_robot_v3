# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Render-free control of the torso-collapse reproduction (no VLM or hardware)."""

import mujoco
import numpy as np

from emet.robots.rby1 import Rby1Backend
from emet.simulation.mujoco_server import _load_default_scene_with_robot
from emet.simulation.mujoco_stationary_control import DefaultMujocoStationaryControl


def test_preserved_targets_prevent_torso_collapse_with_upright_base():
    model = _load_default_scene_with_robot("rby1")
    spec = Rby1Backend().get_spec()
    controller = DefaultMujocoStationaryControl()
    heights = {}
    for mode in ("retarget_measured", "preserve_commanded"):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        base = model.joint("base_freejoint")
        qa, va = int(base.qposadr[0]), int(base.dofadr[0])
        snapshot = data.qpos[qa : qa + 7].copy()
        snapshot[:3] = [0, 1.5, 0]
        hold = np.array([data.ctrl[model.actuator(n).id] for n in spec.actuator_names])
        hold[spec.actuator_names.index("torso1")] = -np.pi / 6
        heights[mode] = []
        for view in range(8):
            if view and mode == "retarget_measured":
                controller.sync_ctrl_and_spec_hold(model, data, spec, hold)
            for _ in range(1000):
                # Isolate torso stability: the chassis cannot tip in this test.
                data.qpos[qa : qa + 7] = snapshot
                data.qvel[va : va + 6] = 0
                controller.write_ctrl_with_spec_hold(model, data, spec, hold)
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            assert data.body("base_link").xmat.reshape(3, 3)[2, 2] > 0.999
            heights[mode].append(float(data.camera("zed_camera").xpos[2]))
    assert heights["retarget_measured"][-1] < 0.2
    assert min(heights["preserve_commanded"]) > 1.3
    assert np.ptp(heights["preserve_commanded"]) < 0.03
