# Copyright (c) Hello Robot, Inc.
# All rights reserved.

"""Tests for Depth Anything 3 depth path and RGB-only ZMQ observations."""

import os

import numpy as np
import pytest

from emet.core.interfaces import Observations
from emet.core.parameters import Parameters
from emet.perception.depth.da3_estimator import (
    DA3DepthEstimator,
    apply_da3_sky_row_mask,
    apply_depth_speckle_filter,
    create_da3_estimator_from_parameters,
    resize_depth_to_match_rgb,
    resolve_depth_map,
    resolve_depth_map_uses_observation_sensor_only,
    sensor_depth_usable,
)


def test_apply_da3_sky_row_mask_zeros_top_fraction():
    d = np.ones((100, 40), dtype=np.float32) * 1.5
    out = apply_da3_sky_row_mask(d, 0.2)
    assert out.shape == d.shape
    assert float(out[19, 0]) == 0.0
    assert float(out[20, 0]) == 1.5
    assert float(out[0, 39]) == 0.0
    assert np.array_equal(apply_da3_sky_row_mask(d, 0.0), d)


def test_apply_depth_speckle_filter_removes_isolated_pixel():
    d = np.zeros((20, 20), dtype=np.float32)
    d[5:15, 5:15] = 1.2
    d[0, 0] = 1.5
    out = apply_depth_speckle_filter(d, open_kernel=3, min_depth=0.1, max_depth=3.0)
    assert float(out[0, 0]) == 0.0
    assert np.isclose(out[10, 10], 1.2)


def test_apply_depth_speckle_filter_disabled_when_kernel_zero():
    d = np.ones((10, 10), dtype=np.float32)
    assert np.array_equal(apply_depth_speckle_filter(d, open_kernel=0), d)


def test_sensor_depth_usable():
    assert not sensor_depth_usable(None)
    assert not sensor_depth_usable(np.zeros((4, 4), dtype=np.float32))
    d = np.zeros((4, 4), dtype=np.float32)
    d[2, 2] = 0.5
    assert sensor_depth_usable(d)


def test_da3_is_pose_alignment_failure():
    class GeometryException(Exception):
        pass

    assert DA3DepthEstimator._is_pose_alignment_failure(
        GeometryException("Degenerate covariance rank, Umeyama alignment is not possible")
    )
    assert DA3DepthEstimator._is_pose_alignment_failure(RuntimeError("umeyama failed"))
    assert not DA3DepthEstimator._is_pose_alignment_failure(ValueError("other"))


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


@pytest.mark.skipif(
    os.environ.get("RUN_DA3_TESTS", "") != "1",
    reason="Set RUN_DA3_TESTS=1 to run (imports depth_anything_3).",
)
def test_create_da3_estimator_passes_amp_and_compile_flags():
    pytest.importorskip("depth_anything_3")
    p = Parameters(depth_source="da3", da3_use_amp=True, da3_torch_compile=True)
    est = create_da3_estimator_from_parameters(p, device="cpu")
    assert est is not None
    assert est._use_amp is True
    assert est._torch_compile is True


@pytest.mark.skipif(
    os.environ.get("RUN_DA3_TESTS", "") != "1",
    reason="Set RUN_DA3_TESTS=1 to run (imports depth_anything_3).",
)
def test_create_da3_estimator_auto_amp_matches_device():
    pytest.importorskip("depth_anything_3")
    import torch

    p = Parameters(depth_source="da3")
    est_cpu = create_da3_estimator_from_parameters(p, device="cpu")
    assert est_cpu._use_amp is False

    if torch.cuda.is_available():
        est_cuda = create_da3_estimator_from_parameters(p, device="cuda")
        assert est_cuda._use_amp is True
        p_off = Parameters(depth_source="da3", da3_use_amp=False)
        est_off = create_da3_estimator_from_parameters(p_off, device="cuda")
        assert est_off._use_amp is False


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
    ctrl._da3_use_stereo = False
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
    ctrl._da3_use_stereo = False
    ctrl.device = "cpu"

    def _fake_lazy(self):
        return FakeEst()

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", _fake_lazy)

    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    out = DynamemController._resolve_depth_map(ctrl, rgb, None, np.eye(3), np.eye(4))
    assert out.shape == (5, 6)
    assert float(out.mean()) == pytest.approx(1.25)


def test_resolve_depth_map_uses_observation_sensor_only():
    assert resolve_depth_map_uses_observation_sensor_only("sensor", None) is True
    assert resolve_depth_map_uses_observation_sensor_only("auto", np.zeros((2, 2), np.float32)) is False
    assert resolve_depth_map_uses_observation_sensor_only("auto", None) is False
    assert resolve_depth_map_uses_observation_sensor_only("da3", np.zeros((2, 2))) is False
    sd = np.ones((4, 4), dtype=np.float32) * 0.5
    assert resolve_depth_map_uses_observation_sensor_only("auto", sd) is True


def test_dynamem_resolve_depth_infer_flag_auto_uses_sensor(monkeypatch):
    pytest.importorskip("torch")
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    params["depth_source"] = "auto"
    ctrl = DynamemController.__new__(DynamemController)
    ctrl.parameters = params
    ctrl._depth_source = "auto"
    ctrl._debug_perfect_sensor_depth = False
    ctrl._da3_estimator = None
    ctrl._da3_use_stereo = False
    ctrl.device = "cpu"

    def _boom(self):
        raise AssertionError("DA3 must not load when auto uses sensor depth")

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", _boom)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    sd = np.ones((4, 4), dtype=np.float32) * 0.4
    out = DynamemController._resolve_depth_map(ctrl, rgb, sd, np.eye(3), np.eye(4))
    assert out is not None
    assert getattr(ctrl, "_depth_map_from_da3_infer", True) is False


def test_dynamem_resolve_depth_infer_flag_da3_infer(monkeypatch):
    pytest.importorskip("torch")
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    params["depth_source"] = "da3"
    ctrl = DynamemController.__new__(DynamemController)
    ctrl.parameters = params
    ctrl._depth_source = "da3"
    ctrl._debug_perfect_sensor_depth = False
    ctrl._da3_estimator = None
    ctrl._da3_use_stereo = False
    ctrl.device = "cpu"

    class FakeEst:
        def infer(self, rgb, intrinsics=None, extrinsics_w2c=None):
            return np.full((rgb.shape[0], rgb.shape[1]), 0.88, dtype=np.float32)

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", lambda self: FakeEst())

    rgb = np.zeros((3, 5, 3), dtype=np.uint8)
    out = DynamemController._resolve_depth_map(ctrl, rgb, None, np.eye(3), np.eye(4))
    assert out is not None
    assert getattr(ctrl, "_depth_map_from_da3_infer", False) is True


def test_dynamem_debug_perfect_depth_skips_da3_and_infer_flag(monkeypatch):
    pytest.importorskip("torch")
    from emet.controller.controller_dynamem import DynamemController
    from emet.core.parameters import get_parameters

    params = get_parameters("dynav_config.yaml")
    params["depth_source"] = "da3"
    ctrl = DynamemController.__new__(DynamemController)
    ctrl.parameters = params
    ctrl._depth_source = "da3"
    ctrl._debug_perfect_sensor_depth = True
    ctrl._da3_estimator = None
    ctrl._da3_use_stereo = False
    ctrl.device = "cpu"

    def _must_not_load_da3(self):
        raise AssertionError("DA3 must not load")

    monkeypatch.setattr(DynamemController, "_lazy_da3_estimator", _must_not_load_da3)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    sd = np.ones((4, 4), dtype=np.float32) * 0.6
    out = DynamemController._resolve_depth_map(ctrl, rgb, sd, np.eye(3), np.eye(4))
    assert out is not None and np.allclose(out, 0.6)
    assert getattr(ctrl, "_depth_map_from_da3_infer", True) is False


def test_resolve_depth_map_default_skips_stereo_path():
    """Keyword default ``da3_use_stereo=False`` avoids two-view inference."""

    class _Est:
        def infer_stereo(self, *_a, **_k):
            raise AssertionError("infer_stereo must not run with default da3_use_stereo")

        def infer(self, rgb, intrinsics=None, extrinsics_w2c=None):
            return np.full((rgb.shape[0], rgb.shape[1]), 2.0, dtype=np.float32)

    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_r = np.zeros((10, 10, 3), dtype=np.uint8)
    k = np.eye(3, dtype=np.float32)
    p = np.eye(4, dtype=np.float32)
    out = resolve_depth_map("da3", _Est(), rgb, None, k, p, rgb_r, k, p)
    assert out is not None and np.allclose(out, 2.0)


def test_resolve_depth_map_da3_use_stereo_false_skips_stereo_path():
    class _Est:
        def infer_stereo(self, *_a, **_k):
            raise AssertionError("infer_stereo must not run when da3_use_stereo is false")

        def infer(self, rgb, intrinsics=None, extrinsics_w2c=None):
            return np.full((rgb.shape[0], rgb.shape[1]), 2.0, dtype=np.float32)

    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb_r = np.zeros((10, 10, 3), dtype=np.uint8)
    k = np.eye(3, dtype=np.float32)
    p = np.eye(4, dtype=np.float32)
    out = resolve_depth_map("da3", _Est(), rgb, None, k, p, rgb_r, k, p, da3_use_stereo=False)
    assert out is not None and np.allclose(out, 2.0)
