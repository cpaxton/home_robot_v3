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
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

# Copyright (c) Hello Robot, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from emet.core.zmq_protocol import EMET_ZMQ_SESSION_KEY
from emet.robots.galaxea_r1 import R1_JOINT_NAMES
from emet.robots.innate_mars import INNATE_MARS_JOINT_NAMES
from emet.visualization.mjcf_rerun_robot import (
    MjcfBodySkeletonLogger,
    MjcfVisualMeshLogger,
    _body_T_world,
    _nav_world_xyt_from_obs,
    _T_world_from_planar_xyt,
    _world_alignment_fixup_T,
    _world_xyt_from_base_body,
    apply_zmq_obs_to_mujoco_data,
)

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
    T_ee = _body_T_world(data, int(ee))
    Trel = np.linalg.inv(Tb) @ T_ee
    assert np.allclose(Tb @ Trel, T_ee, atol=1e-9)


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_world_xyt_from_base_body_matches_episode_relative_gps():
    """Planar MJCF replay uses episode-relative gps/compass for base qpos (not world joint[0:3])."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    nav_slot: list = [None]
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [1.5, -0.25, 0.4] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(INNATE_MARS_JOINT_NAMES) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [2.0, 3.0, 0.0]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=len(INNATE_MARS_JOINT_NAMES),
        base_link_name="base_link",
        nav_origin_slot=nav_slot,
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    xyt = _world_xyt_from_base_body(model, data, "base_link")
    assert float(np.linalg.norm(xyt[:2])) < 0.02
    assert abs(xyt[2]) < 0.02


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_base_relative_mesh_matches_nav_fixup():
    """GPS ``world/robot`` × base-relative verts == T_fix @ standalone-world verts."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    obs = {
        "gps": [0.3, -0.1],
        "compass": [0.25],
        "joint": [1.2, 0.4, 0.1] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(INNATE_MARS_JOINT_NAMES) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [2.0, 1.0, 0.5]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=len(INNATE_MARS_JOINT_NAMES),
        base_link_name="base_link",
        nav_origin_slot=[None],
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    T_sb = _body_T_world(data, int(bb))
    p_w = _body_T_world(data, int(ee))[:3, 3]
    p_rel = (np.linalg.inv(T_sb) @ np.r_[p_w, 1.0])[:3]
    wxyt = _nav_world_xyt_from_obs(obs)
    T_nav = _T_world_from_planar_xyt(float(wxyt[0]), float(wxyt[1]), float(wxyt[2]), float(T_sb[2, 3]))
    T_fix = _world_alignment_fixup_T(wxyt, T_sb)
    p_via_parent = (T_nav @ np.r_[p_rel, 1.0])[:3]
    p_via_fixup = (T_fix @ np.r_[p_w, 1.0])[:3]
    assert np.allclose(p_via_parent, p_via_fixup, atol=1e-6)


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_planar_mjcf_base_qpos_from_gps_not_world_joints():
    """Planar replay uses episode-relative gps, not world joint[0:3] (avoids double compose)."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [3.0, -1.0, 0.2] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(INNATE_MARS_JOINT_NAMES) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [3.0, -1.0, 0.2]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=len(INNATE_MARS_JOINT_NAMES),
        base_link_name="base_link",
        nav_origin_slot=[None],
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    xyt = _world_xyt_from_base_body(model, data, "base_link")
    assert float(np.linalg.norm(xyt[:2])) < 0.05
    assert abs(xyt[2]) < 0.05


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_planar_mjcf_world_ee_matches_nav_compose():
    """With mismatched world joints, nav-world EE still matches gps+origin compose."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    obs = {
        "gps": [0.2, -0.1],
        "compass": [0.15],
        "joint": [5.0, 5.0, 0.0] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(INNATE_MARS_JOINT_NAMES) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [2.0, 1.0, 0.5]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=len(INNATE_MARS_JOINT_NAMES),
        base_link_name="base_link",
        nav_origin_slot=[None],
        free_qadr=None,
    )
    mujoco.mj_forward(model, data)
    bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
    T_sb = _body_T_world(data, int(bb))
    T_ee = _body_T_world(data, int(ee))
    T_rel = np.linalg.inv(T_sb) @ T_ee
    wxyt = _nav_world_xyt_from_obs(obs)
    T_nav = _T_world_from_planar_xyt(float(wxyt[0]), float(wxyt[1]), float(wxyt[2]), float(T_sb[2, 3]))
    p_world = (T_nav @ T_rel)[:3, 3]
    p_fixup = (_world_alignment_fixup_T(wxyt, T_sb) @ T_ee)[:3, 3]
    assert np.allclose(p_world, p_fixup, atol=1e-5)


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_mjcf_visual_mesh_logger_nonzero_origin_base_local():
    """sync_kinematics leaves base near episode origin when gps=0 (not at nav spawn)."""
    pytest.importorskip("mujoco")
    import numpy as np

    log = MjcfVisualMeshLogger(_INNATE_MJCF, tuple(INNATE_MARS_JOINT_NAMES), len(INNATE_MARS_JOINT_NAMES), "base_link")
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [3.0, -1.0, 0.2] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(INNATE_MARS_JOINT_NAMES) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [3.0, -1.0, 0.2]},
    }
    xyt = log.sync_kinematics(obs)
    assert float(np.linalg.norm(xyt[:2])) < 0.05


def test_mjcf_visual_mesh_logger_smoke():
    """MjcfVisualMeshLogger logs mesh geoms without raising (Rerun + MuJoCo)."""
    pytest.importorskip("mujoco")
    rr = pytest.importorskip("rerun")
    rr.init("test_mjcf_visual_mesh_smoke", spawn=False)
    log = MjcfVisualMeshLogger(_INNATE_MJCF, tuple(INNATE_MARS_JOINT_NAMES), len(INNATE_MARS_JOINT_NAMES), "base_link")
    joint_names = tuple(INNATE_MARS_JOINT_NAMES)
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [0.05, -0.04, 0.2] + [0.15, -0.1, 0.2, 0.0, 0.0, 0.0] + [0.0] * (len(joint_names) - 9),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [0.0, 0.0, 0.0]},
    }
    log.log_meshes_world(rr, obs, entity_prefix="world/robot/mjcf_visual")
    assert log._geom_mesh_cache, "expected at least one mesh geom in innate_mars MJCF"


@pytest.mark.skipif(not _MJCF.is_file(), reason="galaxea_r1 MJCF not present")
def test_mjcf_visual_mesh_logger_galaxea_smoke():
    pytest.importorskip("mujoco")
    rr = pytest.importorskip("rerun")
    rr.init("test_mjcf_visual_mesh_galaxea_smoke", spawn=False)
    log = MjcfVisualMeshLogger(_MJCF, R1_JOINT_NAMES, 26, "base_link")
    obs = {
        "gps": [0.1, -0.2],
        "compass": [0.3],
        "joint": [0.01] * 26,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [1.0, 2.0, 0.5]},
    }
    log.log_meshes_world(rr, obs, entity_prefix="world/robot/mjcf_visual")
    assert log._geom_mesh_cache


@pytest.mark.skipif(not _MJCF.is_file(), reason="galaxea_r1 MJCF not present")
def test_galaxea_free_joint_qpos_is_episode_relative_at_spawn():
    """Free-joint MJCF replay must not bake navigation_origin into qpos (world/robot composes)."""
    mujoco = pytest.importorskip("mujoco")
    import numpy as np

    from emet.visualization.mjcf_rerun_robot import _base_freejoint_qadr

    model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(model)
    free_qadr = _base_freejoint_qadr(model, "base_link")
    assert free_qadr is not None
    nav_slot: list[np.ndarray | None] = [None]
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [0.01] * 26,
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [2.0, 3.0, 0.5]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(R1_JOINT_NAMES),
        dof=26,
        base_link_name="base_link",
        nav_origin_slot=nav_slot,
        free_qadr=free_qadr,
    )
    mujoco.mj_forward(model, data)
    xyt = _world_xyt_from_base_body(model, data, "base_link")
    assert float(np.linalg.norm(xyt[:2])) < 0.02


@pytest.mark.skipif(not _INNATE_MJCF.is_file(), reason="innate_mars MJCF not present")
def test_nav_origin_slot_tracks_latest_session():
    import numpy as np

    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(_INNATE_MJCF))
    data = mujoco.MjData(model)
    nav_slot: list[np.ndarray | None] = [np.array([1.0, 1.0, 0.0])]
    obs = {
        "gps": [0.0, 0.0],
        "compass": [0.0],
        "joint": [0.0] * len(INNATE_MARS_JOINT_NAMES),
        EMET_ZMQ_SESSION_KEY: {"navigation_origin_xyt": [2.1, 3.2, 0.4]},
    }
    apply_zmq_obs_to_mujoco_data(
        model,
        data,
        obs,
        joint_names=tuple(INNATE_MARS_JOINT_NAMES),
        dof=len(INNATE_MARS_JOINT_NAMES),
        base_link_name="base_link",
        nav_origin_slot=nav_slot,
        free_qadr=None,
    )
    assert nav_slot[0] is not None
    assert np.allclose(nav_slot[0], [2.1, 3.2, 0.4], atol=1e-9)
