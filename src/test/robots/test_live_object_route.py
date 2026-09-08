# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

from pathlib import Path

import numpy as np


def test_route_pose_error_wraps_yaw_and_measures_translation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    from probe_live_object_route import pose_error

    distance, yaw = pose_error([0.03, 0.04, -np.pi + 0.01], [0, 0, np.pi - 0.01])
    assert abs(distance - 0.05) < 1e-8
    assert abs(yaw - 0.02) < 1e-8


def test_live_fixture_recompiles_with_assets_and_spawn(monkeypatch, tmp_path):
    import mujoco
    import yaml

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from probe_live_object_route import load_live_scene

    config = yaml.safe_load((root / "configs/ovmm/stationary_rby1.yaml").read_text())
    spec, model = load_live_scene(config, tmp_path / "scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.body(spec.base_link_name).xpos[:2], config["base_xyt"][:2])
    for target in config["targets"]:
        np.testing.assert_allclose(data.body(target["body"]).xpos, target["position"])


def test_galaxea_camera_holds_tilt_under_gravity(monkeypatch):
    import mujoco
    import yaml

    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from probe_stationary_objects import setup_scene

    from emet.simulation.robosuite_load_utils import freejoint_qpos_qvel_addrs

    config = yaml.safe_load((root / "configs/ovmm/stationary_rby1.yaml").read_text())
    spec, model, data = setup_scene(config)
    qadr, vadr = freejoint_qpos_qvel_addrs(model, spec.base_link_name)
    base = data.qpos[qadr : qadr + 7].copy()
    # Same idealized idle-base constraint as the production bridge. Upper-body
    # joints remain dynamic, with fixed position targets throughout the hold.
    for _ in range(int(10 / model.opt.timestep)):
        data.qpos[qadr : qadr + 7] = base
        data.qvel[vadr : vadr + 6] = 0
        mujoco.mj_step(model, data)
    forward = -data.cam_xmat[model.camera(config["camera"]).id].reshape(3, 3)[:, 2]
    assert abs(np.arcsin(forward[2]) - config["head_tilt"]) < np.deg2rad(1)
