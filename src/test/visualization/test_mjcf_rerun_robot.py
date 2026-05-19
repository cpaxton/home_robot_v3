# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

# Copyright (c) Hello Robot, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.robots.galaxea_r1 import R1_JOINT_NAMES
from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES
from emet.visualization.mjcf_rerun_robot import MjcfBodySkeletonLogger, _body_T_world, apply_zmq_obs_to_mujoco_data

_MJCF = Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "galaxea_r1" / "galaxea_r1.xml"
_INNATE_MJCF = Path(__file__).resolve().parents[2] / "emet" / "assets" / "robot" / "innate_mars" / "innate_mars.xml"


@pytest.mark.skipif(not _MJCF.is_file(), reason="galaxea_r1 MJCF not present")
def test_mjcf_body_skeleton_logger_smoke():
    pytest.importorskip("mujoco")
    rr = pytest.importorskip("rerun")
    rr.init("test_mjcf_rerun_smoke", spawn=False)
    log = MjcfBodySkeletonLogger(_MJCF, R1_JOINT_NAMES, 26, "base_link")
    obs = {
        "gps": [0.1, -0.2],
        "compass": [0.3],
        "joint": [0.01] * 26,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [1.0, 2.0, 0.5]},
    }
    log.apply_and_log(obs)


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_innate_mars_base_relative_body_transforms():
    """Sanity: inv(T_base) @ T_body reconstructs absolute body pose in standalone MuJoCo world."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    joint_names = tuple(INNATE_MARS_JOINT_NAMES)
    nav_slot: list[np.ndarray | None] = [None]
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [0.05, -0.04, 0.2] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(joint_names) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [0.0, 0.0, 0.0]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=joint_names,
        dof=len(joint_names),
        base_link_name="base_link",
        nav_origin_slot=nav_slot,
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    assert bb >= 0 and ee >= 0
    Tb = _body_T_world(data, int(bb))
    Te = _body_T_world(data, int(ee))
    Trel = np.linalg.inv(Tb) @ Te
    assert np.allclose(Tb @ Trel, Te, atol=1e-9)
