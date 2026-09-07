# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Check actual compiled optical axes, not just actuator command values."""

import mujoco
import numpy as np
import pytest

from emet.robots.rby1 import Rby1Backend
from emet.simulation.head_look_action import apply_head_to_robosuite
from emet.simulation.mujoco_server import _load_default_scene_with_robot


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("pan,tilt", [(0.0, 0.0), (0.0, -np.pi / 6), (0.5, 0.0)])
def test_optical_axes_and_look_direction(merged, pan, tilt):
    spec = Rby1Backend().get_spec()
    model = _load_default_scene_with_robot("rby1") if merged else mujoco.MjModel.from_xml_path(spec.mjcf_path)
    data = mujoco.MjData(model)
    assert apply_head_to_robosuite(spec, model, data, pan, tilt) == 2
    # Kinematic adapter check; physics tracking is covered by live scene probes.
    for actuator in ("torso3", "torso4"):
        aid = model.actuator(actuator).id
        joint = int(model.actuator_trnid[aid, 0])
        data.qpos[model.jnt_qposadr[joint]] = data.ctrl[aid]
    mujoco.mj_forward(model, data)
    rotation = data.body("base_link").xmat.reshape(3, 3).T @ data.camera("zed_camera").xmat.reshape(3, 3)
    forward = -rotation[:, 2]
    assert np.arctan2(forward[1], forward[0]) == pytest.approx(pan, abs=1e-6)
    assert np.arcsin(forward[2]) == pytest.approx(tilt, abs=1e-6)
    assert rotation[2, 1] > 0.85, "looking down must not roll or invert the image"
    assert data.qpos[model.joint("torso_joint1").qposadr[0]] == 0
