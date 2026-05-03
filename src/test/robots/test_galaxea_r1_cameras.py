# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source code tree.

"""Galaxea R1 MJCF: RGB cameras must be MuJoCo ``<camera>`` with sane world-frame optics.

``RobosuiteZmqServer`` resolves ``RobotSpec.camera_names`` via ``mjOBJ_CAMERA`` only; sites are
ignored and the renderer falls back to a world-fixed free camera. Cameras must also look along
the sensor +X link axis (not -Z) so RGB matches the physical rig / passive viewer.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from emet.utils.assets import get_robot_mjcf_path


def _require_galaxea_xml() -> str:
    p = get_robot_mjcf_path("rby1")
    if p is None or not p.is_file():
        pytest.skip("galaxea_r1 MJCF not found in this checkout")
    return str(p.resolve())


def _cam_look_world(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    """World-frame unit vector along MuJoCo camera viewing direction (-local Z)."""
    c = data.cam_xmat[cam_id].reshape(3, 3)
    return (-c[:, 2]).astype(np.float64)


def _cam_y_world(data: mujoco.MjData, cam_id: int) -> np.ndarray:
    """World +Y axis of the camera frame (second column of ``cam_xmat``)."""
    c = data.cam_xmat[cam_id].reshape(3, 3)
    return c[:, 1].astype(np.float64)


def _body_axis_x_world(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return data.xmat[body_id].reshape(3, 3)[:, 0].astype(np.float64)


@pytest.fixture(scope="module")
def galaxea_model_path() -> str:
    return _require_galaxea_xml()


@pytest.fixture(scope="module")
def galaxea_mj(galaxea_model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = mujoco.MjModel.from_xml_path(galaxea_model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_head_and_wrist_names_are_cameras_not_sites(galaxea_mj: tuple[mujoco.MjModel, mujoco.MjData]):
    m, d = galaxea_mj
    for name in ("zed_camera", "left_camera", "right_camera"):
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert cid >= 0, f"{name!r} must be mjOBJ_CAMERA for ZMQ rendering"
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        assert sid < 0, f"{name!r} must not also exist as a site"


def test_zed_camera_pose_tracks_base_translation(galaxea_mj: tuple[mujoco.MjModel, mujoco.MjData]):
    m, d = galaxea_mj
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_freejoint")
    assert j >= 0
    adr = int(m.jnt_qposadr[j])
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "zed_camera")
    assert cid >= 0

    d.qpos[adr : adr + 3] = (0.0, 0.0, 0.08)
    mujoco.mj_forward(m, d)
    p0 = d.cam_xpos[cid].copy()

    d.qpos[adr : adr + 3] = (1.7, -0.9, 0.08)
    mujoco.mj_forward(m, d)
    p1 = d.cam_xpos[cid].copy()

    delta = p1 - p0
    assert np.allclose(delta[:2], (1.7, -0.9), atol=0.02), "head camera must rigidly follow base XY"
    assert abs(float(delta[2])) < 0.05, "head camera height should not jump from a pure XY base shift"


def test_zed_looks_forward_along_head_link_plus_x_not_back_along_minus_z(
    galaxea_mj: tuple[mujoco.MjModel, mujoco.MjData],
):
    m, d = galaxea_mj
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "zed_camera")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "zed_link")
    look = _cam_look_world(d, cid)
    bx = _body_axis_x_world(d, bid)
    assert float(np.dot(look, bx)) > 0.92, "ZED view axis should align with +zed_link X (lens forward)"
    wup = np.array([0.0, 0.0, 1.0])
    cy = _cam_y_world(d, cid)
    assert float(np.dot(cy, wup)) > 0.5, "camera +Y should tilt toward world +Z for upright RGB rows"


@pytest.mark.parametrize(
    ("cam", "body"),
    (
        ("left_camera", "left_realsense_link"),
        ("right_camera", "right_realsense_link"),
    ),
)
def test_wrist_cameras_forward_and_upright(
    galaxea_mj: tuple[mujoco.MjModel, mujoco.MjData],
    cam: str,
    body: str,
):
    m, d = galaxea_mj
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    look = _cam_look_world(d, cid)
    bx = _body_axis_x_world(d, bid)
    assert float(np.dot(look, bx)) > 0.92, f"{cam} should look along +{body} X"
    wup = np.array([0.0, 0.0, 1.0])
    cy = _cam_y_world(d, cid)
    assert float(np.dot(cy, wup)) > 0.5, f"{cam} should be roughly upright vs world +Z"
