# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
from types import SimpleNamespace

import numpy as np
import pytest

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.motion.observed_support import _largest_rectangle, fit_observed_support


def fixture():
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    obs = SimpleNamespace(
        rgb=np.zeros((80, 80, 3), np.uint8),
        depth=np.ones((80, 80)),
        camera_K=np.array([[100, 0, 40], [0, 100, 40], [0, 0, 1]]),
        camera_pose=pose,
        received_monotonic=10.0,
    )
    mask = np.zeros((80, 80), bool)
    mask[10:70, 10:70] = True
    return obs, mask


def test_support_patch_uses_measured_footprint_and_keeps_original_time():
    obs, mask = fixture()
    patch = fit_observed_support(obs, mask, observation_id="camera:1", measurement_margin_m=0.002)
    assert patch.received_monotonic == 10.0 and patch.residual_m < 1e-6
    assert np.all(patch.bounds[1, :2] - patch.bounds[0, :2] > 0.4)
    with pytest.raises(ValueError):
        patch.bounds.setflags(write=True)


def test_unknown_stripe_is_not_filled_into_support_rectangle():
    cells = np.ones((10, 10), bool)
    cells[:, 4:6] = False
    x0, y0, x1, y1 = _largest_rectangle(cells)
    assert (x1 - x0) * (y1 - y0) == 40
    assert cells[x0:x1, y0:y1].all()
    obs, mask = fixture()
    mask[:, 38:43] = False
    patch = fit_observed_support(obs, mask, observation_id="camera:1", measurement_margin_m=0.002)
    assert patch.bounds[1, 0] < 0 or patch.bounds[0, 0] > 0


def test_mixed_object_and_support_depth_cannot_pass_plane_fit():
    obs, mask = fixture()
    obs.depth[30:50, 30:50] = 0.9
    with pytest.raises(ObservationUnavailable, match="support_plane"):
        fit_observed_support(obs, mask, observation_id="camera:1", measurement_margin_m=0.002)


def test_refresh_remeasures_holes_and_cannot_renew_a_missing_plane():
    from emet.motion.observed_support import refresh_support_patch

    obs, mask = fixture()
    reference = fit_observed_support(obs, mask, observation_id="reference", measurement_margin_m=0.002)
    obs.received_monotonic = 11.0
    obs.depth[:, 38:43] = np.nan
    refreshed = refresh_support_patch(obs, reference, observation_id="current")
    assert refreshed.received_monotonic == 11.0 and reference.received_monotonic == 10.0
    assert refreshed.bounds[1, 0] < 0 or refreshed.bounds[0, 0] > 0
    obs.depth[:] = 0.8
    with pytest.raises(ObservationUnavailable):
        refresh_support_patch(obs, reference, observation_id="moved")


def test_support_seed_never_jumps_to_other_surface():
    from emet.motion.observed_support import support_mask_from_seed

    obs, _ = fixture()
    selection = {"verified": True, "box": [0, 0, 1000, 1000], "point": [250, 500]}
    obs.depth[:, 39:42] = np.nan
    selected = support_mask_from_seed(obs, selection, measurement_margin_m=0.002)
    assert selected[40, 20] and not selected[:, 42:].any()
    selection["point"] = [500, 500]
    with pytest.raises(ObservationUnavailable):
        support_mask_from_seed(obs, selection, measurement_margin_m=0.002)
