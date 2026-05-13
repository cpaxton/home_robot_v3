# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Tests for Depth Anything 3 depth path and RGB-only ZMQ observations."""

import os

import numpy as np
import pytest

from emet.core.interfaces import Observations
from emet.core.parameters import Parameters
from emet.perception.depth.da3_estimator import (
    apply_da3_sky_row_mask,
    create_da3_estimator_from_parameters,
    resize_depth_to_match_rgb,
)


def test_apply_da3_sky_row_mask_zeros_top_fraction():
    d = np.ones((100, 40), dtype=np.float32) * 1.5
    out = apply_da3_sky_row_mask(d, 0.2)
    assert out.shape == d.shape
    assert float(out[19, 0]) == 0.0
    assert float(out[20, 0]) == 1.5
    assert float(out[0, 39]) == 0.0
    assert np.array_equal(apply_da3_sky_row_mask(d, 0.0), d)


def test_resize_depth_to_match_rgb():
    rgb = np.zeros((100, 80, 3), dtype=np.uint8)
    d = np.ones((50, 40), dtype=np.float32) * 1.5
    out = resize_depth_to_match_rgb(d, rgb)
    assert out.shape == (100, 80)
    assert out.dtype == np.float32
    assert np.allclose(out[0, 0], 1.5)


def test_create_da3_estimator_sensor_returns_none():
    p = Parameters(depth_source="sensor")
    assert create_da3_estimator_from_parameters(p, device="cpu") is None


@pytest.mark.skipif(
    os.environ.get("RUN_DA3_TESTS", "") != "1",
    reason="Set RUN_DA3_TESTS=1 to run (imports depth_anything_3).",
)
def test_create_da3_estimator_respects_clip_max_m():
    pytest.importorskip("depth_anything_3")
    p = Parameters(depth_source="da3", da3_clip_max_m=3.25, max_depth=2.5)
    est = create_da3_estimator_from_parameters(p, device="cpu")
    assert est is not None
    assert abs(float(est._clip_output_max_m) - 3.25) < 1e-5
    p2 = Parameters(depth_source="da3", max_depth=2.0)
    est2 = create_da3_estimator_from_parameters(p2, device="cpu")
    assert float(est2._clip_output_max_m) == max(2.0 + 1.0, 4.0)


@pytest.mark.skipif(
    os.environ.get("RUN_DA3_TESTS", "") != "1",
    reason="Set RUN_DA3_TESTS=1 to run (downloads weights / loads depth_anything_3).",
)
def test_create_da3_estimator_da3_returns_instance():
    pytest.importorskip("depth_anything_3")
    p = Parameters(depth_source="da3", da3_model_id="depth-anything/DA3-SMALL")
    est = create_da3_estimator_from_parameters(p, device="cpu")
    assert est is not None


def test_observations_accepts_none_depth():
    obs = Observations(
        gps=np.zeros(2),
        compass=np.zeros(1),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=None,
        camera_K=np.eye(3),
        camera_pose=np.eye(4),
    )
    assert obs.depth is None


def test_dynamem_controller_resolve_depth_auto_uses_sensor(monkeypatch):
    pytest.importorskip("torch")
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    params["depth_source"] = "auto"

    ctrl = DynamemController.__new__(DynamemController)
    ctrl.parameters = params
    ctrl._depth_source = "auto"
    ctrl._da3_estimator = None
    ctrl.device = "cpu"

    def _fail_lazy(self):
        raise AssertionError("DA3 should not load when sensor depth is present")

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", _fail_lazy)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    sd = np.ones((4, 4), dtype=np.float32) * 0.5
    out = DynamemController._resolve_depth_map(ctrl, rgb, sd, np.eye(3), np.eye(4))
    assert out is not None
    assert out.shape == (4, 4)


def test_dynamem_controller_resolve_depth_auto_falls_back_to_da3(monkeypatch):
    pytest.importorskip("torch")
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    params["depth_source"] = "auto"

    class FakeEst:
        def infer(self, rgb, intrinsics=None, extrinsics_w2c=None):
            return np.full((rgb.shape[0], rgb.shape[1]), 1.25, dtype=np.float32)

    ctrl = DynamemController.__new__(DynamemController)
    ctrl.parameters = params
    ctrl._depth_source = "auto"
    ctrl._da3_estimator = None
    ctrl.device = "cpu"

    def _fake_lazy(self):
        return FakeEst()

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", _fake_lazy)

    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    out = DynamemController._resolve_depth_map(ctrl, rgb, None, np.eye(3), np.eye(4))
    assert out.shape == (5, 6)
    assert float(out.mean()) == pytest.approx(1.25)
