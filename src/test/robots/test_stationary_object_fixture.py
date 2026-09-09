# Copyright (c) Chris Paxton 2026
#
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

import importlib.util
from pathlib import Path

import numpy as np
import yaml


def test_stationary_fixture_projects_both_targets_into_actual_asset_camera():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("stationary_probe", root / "scripts/probe_stationary_objects.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    config = yaml.safe_load((root / "configs/ovmm/stationary_rby1.yaml").read_text())
    _, model, data = probe.setup_scene(config)
    cid = model.camera(config["camera"]).id
    rotation = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])
    np.testing.assert_allclose(np.arctan2(rotation[1, 2], rotation[0, 2]), config["base_xyt"][2], atol=1e-6)
    height, width = config["image_size"]
    focal = height / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
    for target in config["targets"]:
        actual = data.body(target["body"]).xpos
        np.testing.assert_allclose(actual, target["position"])
        camera_xyz = rotation.T @ (actual - data.cam_xpos[cid])
        assert 0.25 < camera_xyz[2] < 2.5
        u, v = focal * camera_xyz[:2] / camera_xyz[2] + [width / 2, height / 2]
        assert 50 < u < width - 50 and 50 < v < height - 50
    assert data.time == 0.0, "fixture must not silently advance physics"
