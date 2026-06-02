# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.

from __future__ import annotations

import mujoco
import numpy as np
import pytest


def test_actuator_spec_vs_model_report_matches_galaxea_r1():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.robosuite_load_utils import actuator_spec_vs_model_report

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    missing, extra = actuator_spec_vs_model_report(model, spec)
    assert missing == [], f"unexpected missing actuators: {missing}"
    assert extra == [], f"unexpected extra actuators: {extra}"


def test_apply_home_keyframe_preserves_base_free_joint():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_stationary_control import sync_stationary_ctrl_and_spec_hold
    from emet.simulation.robosuite_load_utils import (
        apply_home_keyframe_preserving_base,
        freejoint_qpos_qvel_addrs,
        probe_max_qvel_unforced_steps,
    )

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    addrs = freejoint_qpos_qvel_addrs(model, spec.base_link_name)
    assert addrs is not None
    qadr = int(addrs[0])
    mujoco.mj_forward(model, data)
    data.qpos[qadr] += 0.5
    data.qpos[qadr + 1] -= 0.25
    mujoco.mj_forward(model, data)
    base_expected = np.array(data.qpos[qadr : qadr + 7], copy=True)
    assert apply_home_keyframe_preserving_base(model, data, base_body_name=spec.base_link_name)
    np.testing.assert_allclose(data.qpos[qadr : qadr + 7], base_expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(model.qpos0[qadr : qadr + 7], base_expected, rtol=0, atol=1e-6)

    n_spec = min(len(spec.actuator_names), len(spec.joint_names))
    probe_hold = np.zeros(n_spec, dtype=np.float64)

    def sync():
        sync_stationary_ctrl_and_spec_hold(model, data, spec, probe_hold)
        mujoco.mj_forward(model, data)

    mx = probe_max_qvel_unforced_steps(model, data, n_steps=32, sync_ctrl=sync)
    assert mx is not None
    assert mx < 50.0, f"expected bounded probe velocities, got max|qvel|={mx}"


def test_apply_home_snaps_arm_qpos_to_ctrl_on_merged_mjcf():
    """Merged key_qpos can disagree with key ctrl; apply_home must align hinge qpos before dynamics."""
    from pathlib import Path

    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.robosuite_load_utils import (
        apply_home_keyframe_preserving_base,
        freejoint_qpos_qvel_addrs,
    )

    merged = Path("src/emet/assets/robot/galaxea_r1/molmospaces_merged_3nvji0h5.xml")
    if not merged.is_file():
        pytest.skip("no cached merged MJCF in repo")
    spec = Rby1Backend().get_spec()
    model = mujoco.MjModel.from_xml_path(str(merged))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    addrs = freejoint_qpos_qvel_addrs(model, spec.base_link_name)
    assert addrs is not None
    qadr = int(addrs[0])
    data.qpos[qadr] += 0.1
    data.qpos[qadr + 1] -= 0.05
    mujoco.mj_forward(model, data)
    base_expected = np.array(data.qpos[qadr : qadr + 7], copy=True)
    assert apply_home_keyframe_preserving_base(model, data, base_body_name=spec.base_link_name)
    np.testing.assert_allclose(data.qpos[qadr : qadr + 7], base_expected, rtol=0, atol=1e-6)
    la2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_joint2")
    la3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_joint3")
    a2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_arm2")
    a3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_arm3")
    q2 = int(model.jnt_qposadr[la2])
    q3 = int(model.jnt_qposadr[la3])
    np.testing.assert_allclose(data.qpos[q2], data.ctrl[a2], rtol=0, atol=1e-5)
    np.testing.assert_allclose(data.qpos[q3], data.ctrl[a3], rtol=0, atol=1e-5)


def test_apply_zero_joint_pose_preserving_base():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.robosuite_load_utils import (
        apply_zero_joint_pose_preserving_base,
        freejoint_qpos_qvel_addrs,
    )

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    addrs = freejoint_qpos_qvel_addrs(model, spec.base_link_name)
    if addrs is None:
        pytest.skip("no base free joint")
    qadr, _ = addrs
    data.qpos[qadr + 2] = 0.5
    apply_zero_joint_pose_preserving_base(model, data, base_body_name=spec.base_link_name)
    la2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_joint2")
    q2 = int(model.jnt_qposadr[la2])
    assert abs(float(data.qpos[q2])) < 1e-9
    assert abs(float(data.qpos[qadr + 2]) - 0.5) < 1e-9
    assert np.max(np.abs(data.ctrl)) < 1e-9


def test_build_tune_model_freezes_base_and_adds_floor():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_home_tune import build_tune_model
    from emet.simulation.robosuite_load_utils import freejoint_qpos_qvel_addrs

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model, data, logs = build_tune_model(
        path,
        initial_pose="home",
        base_body_name=spec.base_link_name,
        tune_base_z=0.38,
        kinematic=False,
    )
    assert any("emet_tune_floor" in ln for ln in logs)
    assert any("Froze" in ln for ln in logs)
    assert freejoint_qpos_qvel_addrs(model, spec.base_link_name) is None
    assert float(np.linalg.norm(model.opt.gravity)) > 0.0
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
    z0 = float(data.xpos[bid, 2])
    for _ in range(120):
        mujoco.mj_step(model, data)
    z1 = float(data.xpos[bid, 2])
    assert abs(z1 - z0) < 0.02, f"base should stay fixed during tune sandbox, z0={z0} z1={z1}"


def test_home_keyframe_exists_on_galaxea_mjcf():
    from emet.utils.assets import get_robot_mjcf_path

    path = get_robot_mjcf_path("rby1")
    if not path:
        pytest.skip("no mjcf")
    model = mujoco.MjModel.from_xml_path(str(path))
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert kid >= 0
    assert model.key_qpos.shape[0] > kid


def test_sync_hold_from_qpos_differs_from_ctrl_targets_after_pose_perturb():
    """``sync_stationary`` seeds hold from ``q``; PD setpoints live in ``ctrl`` (regression guard)."""
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_stationary_control import (
        sync_stationary_ctrl_and_spec_hold,
        write_ctrl_stationary_with_spec_hold,
    )
    from emet.simulation.robosuite_load_utils import apply_home_keyframe_preserving_base

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    assert apply_home_keyframe_preserving_base(model, data, base_body_name=spec.base_link_name)
    hold = np.zeros(len(spec.actuator_names), dtype=np.float64)
    write_ctrl_stationary_with_spec_hold(model, data, spec, hold)
    la2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_joint2")
    qadr2 = int(model.jnt_qposadr[la2])
    data.qpos[qadr2] = 0.35
    mujoco.mj_forward(model, data)
    sync_stationary_ctrl_and_spec_hold(model, data, spec, hold)
    i2 = spec.actuator_names.index("left_arm2")
    assert abs(float(hold[i2]) - 0.35) < 1e-6
    hold[:] = 0.0
    write_ctrl_stationary_with_spec_hold(model, data, spec, hold)
    for i, aname in enumerate(spec.actuator_names):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid >= 0:
            hold[i] = float(data.ctrl[aid])
    assert abs(float(hold[i2])) < 1e-6


def test_home_stationary_holds_under_gravity():
    """Home keyframe + stationary ctrl should keep torso/arm near home under gravity."""
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_stationary_control import write_ctrl_stationary_with_spec_hold
    from emet.simulation.robosuite_load_utils import apply_home_keyframe_preserving_base

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torso2")
    if aid < 0:
        pytest.skip("torso2 actuator missing")
    assert apply_home_keyframe_preserving_base(model, data, base_body_name=spec.base_link_name)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.base_link_name)
    qadr = vadr = None
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            qadr, vadr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    assert qadr is not None
    base_snap = np.array(data.qpos[qadr : qadr + 7], copy=True)
    hold = np.zeros(len(spec.actuator_names), dtype=np.float64)
    for _ in range(400):
        data.qpos[qadr : qadr + 7] = base_snap
        if vadr >= 0:
            data.qvel[vadr : vadr + 6] = 0.0
        write_ctrl_stationary_with_spec_hold(model, data, spec, hold)
        mujoco.mj_step(model, data)
    la2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_arm_joint2")
    t2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso_joint2")
    la2_q0 = float(data.qpos[model.jnt_qposadr[la2]])
    t2_q0 = float(data.qpos[model.jnt_qposadr[t2]])
    assert abs(float(data.qpos[model.jnt_qposadr[la2]]) - la2_q0) < 0.06
    assert abs(float(data.qpos[model.jnt_qposadr[t2]]) - t2_q0) < 0.08
    assert float(np.max(np.abs(data.qvel))) < 0.35


def test_build_tune_model_kinematic_does_not_drift():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_home_tune import build_tune_model

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model, data, logs = build_tune_model(
        path,
        initial_pose="zeros",
        base_body_name=spec.base_link_name,
        kinematic=True,
    )
    assert any("Kinematic" in ln for ln in logs)
    assert float(np.linalg.norm(model.opt.gravity)) == 0.0
    q0 = np.array(data.qpos, copy=True)
    for _ in range(80):
        mujoco.mj_step(model, data)
    assert np.max(np.abs(data.qpos - q0)) < 1e-9


def test_format_key_ctrl_attr_matches_stationary_vector():
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_home_tune import format_key_ctrl_attr
    from emet.simulation.mujoco_stationary_control import compute_stationary_ctrl_vector

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    s = format_key_ctrl_attr(model, data)
    assert len(s.split()) == int(model.nu)
    u = compute_stationary_ctrl_vector(model, data)
    np.testing.assert_allclose(np.array([float(x) for x in s.split()]), u, rtol=0, atol=1e-9)


def test_stationary_ctrl_full_vector_matches_qpos_for_spec_hinges():
    """Merged models need ``ctrl`` length ``nu``; position actuators should track ``qpos`` at rest."""
    from emet.robots.rby1 import Rby1Backend
    from emet.simulation.mujoco_stationary_control import (
        compute_stationary_ctrl_vector,
        sync_stationary_ctrl_and_spec_hold,
    )

    spec = Rby1Backend().get_spec()
    path = spec.mjcf_path
    if not path:
        pytest.skip("no mjcf_path on spec")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    u = compute_stationary_ctrl_vector(model, data)
    assert u.shape == (int(model.nu),)
    n = min(len(spec.actuator_names), len(spec.joint_names))
    hold = np.zeros(n, dtype=np.float64)
    sync_stationary_ctrl_and_spec_hold(model, data, spec, hold)
    assert data.ctrl.shape == (int(model.nu),)
    for i in range(n):
        aname = spec.actuator_names[i]
        jname = spec.joint_names[i]
        if str(aname).startswith("wheel"):
            continue
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if aid < 0 or jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        np.testing.assert_allclose(data.ctrl[aid], data.qpos[qadr], rtol=0, atol=1e-5)
        np.testing.assert_allclose(hold[i], data.qpos[qadr], rtol=0, atol=1e-5)


def test_galaxea_family_backends_use_galaxea_mujoco_stationary_wrapper():
    from emet.robots.galaxea_r1 import GalaxeaR1Backend
    from emet.robots.galaxea_r1.sim_stationary import GalaxeaR1FamilyMujocoStationary
    from emet.robots.rby1 import Rby1Backend

    assert isinstance(GalaxeaR1Backend().create_mujoco_stationary_control(), GalaxeaR1FamilyMujocoStationary)
    assert isinstance(Rby1Backend().create_mujoco_stationary_control(), GalaxeaR1FamilyMujocoStationary)


def test_innate_mars_backend_defaults_mujoco_stationary_to_none():
    from emet.robots.innate_mars import InnateMarsBackend

    assert InnateMarsBackend().create_mujoco_stationary_control() is None


def test_stretch_backend_uses_stretch_mujoco_stationary_wrapper():
    from emet.robots.stretch import StretchBackend
    from emet.robots.stretch.sim_stationary import StretchMujocoStationary

    assert isinstance(StretchBackend().create_mujoco_stationary_control(), StretchMujocoStationary)


def test_stretch_scene_stationary_ctrl_fills_nu():
    """Default Stretch scene: stationary fill should match ``model.nu`` (MolmoSpaces-style full vector)."""
    import mujoco

    from emet.robots.stretch import StretchBackend
    from emet.utils.assets import get_mujoco_models_path

    path = get_mujoco_models_path() / "scene.xml"
    if not path.is_file():
        pytest.skip("scene.xml not found")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    backend = StretchBackend()
    stationary = backend.create_mujoco_stationary_control()
    assert stationary is not None
    stationary.write_ctrl_with_spec_hold(model, data, backend.get_spec(), None)
    assert data.ctrl.shape == (int(model.nu),)


def test_mujoco_ctrl_debug_env(monkeypatch):
    from emet.simulation.robosuite_server import RobosuiteZmqServer

    monkeypatch.setenv("EMET_MUJOCO_CTRL_DEBUG", "1")
    assert RobosuiteZmqServer._mujoco_ctrl_debug_enabled() is True
    monkeypatch.setenv("EMET_MUJOCO_CTRL_DEBUG", "0")
    assert RobosuiteZmqServer._mujoco_ctrl_debug_enabled() is False
    monkeypatch.setenv("EMET_MUJOCO_CTRL_DEBUG_VERBOSE", "1")
    assert RobosuiteZmqServer._mujoco_ctrl_debug_verbose() is True
