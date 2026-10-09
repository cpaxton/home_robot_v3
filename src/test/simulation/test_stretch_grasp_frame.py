# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE).

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from emet.simulation.stretch_mujoco.utils import URDFmodel


@pytest.mark.parametrize("pitch", [0.0, -0.3, -1.5])
def test_grasp_marker_uses_published_urdf_axes(pitch):
    asset = Path(__file__).resolve().parents[2] / "emet/assets/robot/scene_clearance_control.xml"
    model = mujoco.MjModel.from_xml_path(str(asset))
    data = mujoco.MjData(model)
    cfg = {
        "wrist_yaw": 0.2,
        "wrist_pitch": pitch,
        "wrist_roll": 0.1,
        "lift": 0.6,
        "arm": 0.2,
        "head_pan": 0,
        "head_tilt": 0,
    }
    for name, value in cfg.items():
        if name != "arm":
            data.joint("joint_" + name).qpos = value
    for i in range(4):
        data.joint(f"joint_arm_l{i}").qpos = cfg["arm"] / 4
    mujoco.mj_forward(model, data)
    expected = URDFmodel().get_transform(cfg, "link_grasp_center")
    base_rotation = data.body("base_link").xmat.reshape(3, 3)
    actual_rotation = data.body("link_grasp_center").xmat.reshape(3, 3)
    error = Rotation.from_matrix((base_rotation @ expected[:3, :3]).T @ actual_rotation).magnitude()
    # Existing rounded link angles produce ~1 mrad residual, not 120 degrees.
    assert error < 0.003
    # At a level wrist the published +X direction is out of the palm.
    if pitch == 0:
        assert expected[:3, 0] @ np.array([0, -1, 0]) > 0.9
