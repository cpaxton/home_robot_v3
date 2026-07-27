# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Unit tests for MuJoCo base chase-camera framing."""

from __future__ import annotations

import numpy as np
import pytest

from emet.simulation.chase_camera import base_yaw_deg, build_base_chase_camera

pytestmark = pytest.mark.sim


def test_base_yaw_deg_identity_and_quarter_turn() -> None:
    eye = np.eye(3, dtype=np.float64).ravel()
    assert abs(base_yaw_deg(eye) - 0.0) < 1e-6
    yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64).ravel()
    assert abs(base_yaw_deg(yaw90) - 90.0) < 1e-6


def test_build_base_chase_camera_lookat_and_behind_azimuth() -> None:
    import mujoco

    xml = """
    <mujoco>
      <worldbody>
        <body name="base_link" pos="1 2 0.1">
          <freejoint/>
          <geom type="box" size="0.2 0.2 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    cam = build_base_chase_camera(
        model,
        data,
        bid,
        distance=3.5,
        azimuth_offset_deg=0.0,
        elevation_deg=-15.0,
        lookat_z=0.9,
    )
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    np.testing.assert_allclose(cam.lookat, [1.0, 2.0, 1.0], atol=1e-6)
    assert abs(cam.distance - 3.5) < 1e-9
    assert abs(cam.azimuth - 0.0) < 1e-6
    assert abs(cam.elevation - (-15.0)) < 1e-9

    # Rotate base 90° about Z and expect azimuth to follow.
    data.qpos[3:7] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]  # wxyz yaw=90
    mujoco.mj_forward(model, data)
    cam2 = build_base_chase_camera(model, data, bid, distance=3.5, azimuth_offset_deg=0.0, lookat_z=0.9)
    assert abs(cam2.azimuth - 90.0) < 1e-3
