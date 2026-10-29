# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Moving-mask identity/geometry tests with independently generated 3D surfaces."""

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.moving_rgbd_tracking import MovingRgbdTracker
from emet.controller.manipulation.native_inputs import ObservationUnavailable


def frame(*, shift=(0.0, 0.0, 0.0), pixel_shift=(0, 0), distractor=False, missing=False, mixed=False):
    h, w = 100, 180
    y, x = np.mgrid[:h, :w]
    rgb = np.full((h, w, 3), [120, 80, 40], dtype=np.uint8)
    depth = np.full((h, w), 2.0)
    world = np.stack((x * 0.004, np.full((h, w), 0.5), y * 0.004), -1)
    mask = np.zeros((h, w), bool)
    cx, cy = 60 + pixel_shift[0], 55 + pixel_shift[1]
    for row in range(-15, 16):
        for col in range(-9, 10):
            px = col / 9 * 0.019
            p = np.array([px, -np.sqrt(0.02**2 - px**2), -row / 15 * 0.039]) + shift
            mask[cy + row, cx + col] = True
            world[cy + row, cx + col] = p
    for row in range(1, 6):
        py = -0.02 + row * 0.006
        width = int(np.sqrt(0.02**2 - py**2) / 0.002)
        for col in range(-width, width + 1):
            mask[cy - 15 - row, cx + col] = True
            world[cy - 15 - row, cx + col] = np.array([col * 0.002, py, 0.04]) + shift
    rgb[mask] = [200, 40, 40]
    depth[mask] = 1.0
    if mixed:
        # Connected same-colour foreign geometry must not be trimmed away.
        for col in range(1, 15):
            yy, xx = cy, cx + 9 + col
            mask[yy, xx] = True
            world[yy, xx] = world[cy, cx + 9] + [col * 0.005, 0, 0]
            depth[yy, xx] = 1.0
            rgb[yy, xx] = [200, 40, 40]
    if distractor:
        other = np.zeros_like(mask)
        other[mask] = True
        coords = np.argwhere(other)
        for yy, xx in coords:
            rgb[yy, xx + 45] = rgb[yy, xx]
            depth[yy, xx + 45] = depth[yy, xx]
            world[yy, xx + 45] = world[yy, xx] + [0.08, 0, 0]
    if missing:
        rgb[mask] = [20, 20, 20]
        depth[mask] = 0.0
        world[mask] = np.nan
    return rgb, depth, world, mask


def tracker():
    rgb, depth, world, mask = frame()
    return MovingRgbdTracker(
        PrimitiveModel("cylinder", (0.02, 0.04)),
        object_ref="opaque",
        rgb=rgb,
        depth=depth,
        world=world,
        mask=mask,
        observation_id="first",
        received_monotonic=10.0,
        scene_epoch="scene",
        clock=lambda: 10.5,
        max_update_s=5.0,
        max_evaluations=12000,
    )


def update(track, scene, **kwargs):
    rgb, depth, world, _ = scene
    return track.update(
        rgb, depth, world, observation_id="second", received_monotonic=10.4, scene_epoch="scene", **kwargs
    )


def test_tracks_translation_and_lift_beyond_stationary_reference_window():
    track = tracker()
    result = update(track, frame(shift=(0.07, 0.01, 0.05), pixel_shift=(25, -10)))
    np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0.07, 0.01, 0.05], atol=1e-4)
    assert result.mask.sum() > 500
    assert result.reference_id == track.reference_id
    assert not result.fit.orientation_observable and not result.execution_qualified
    assert result.fit.received_monotonic == 10.4
    with pytest.raises(ValueError):
        result.mask.setflags(write=True)


def test_mixed_silhouette_colors_do_not_seed_background_candidates():
    rgb, depth, world, mask = frame()
    # Reference edge RGB is antialiased against a blue-grey background. It must
    # not teach the tracker that the mixed colour identifies the object.
    rgb[mask & ~binary_erosion(mask)] = [57, 24, 26]
    track = MovingRgbdTracker(
        PrimitiveModel("cylinder", (0.02, 0.04)),
        object_ref="opaque",
        rgb=rgb,
        depth=depth,
        world=world,
        mask=mask,
        observation_id="first",
        received_monotonic=10.0,
        scene_epoch="scene",
        clock=lambda: 10.5,
        max_update_s=5.0,
    )
    rgb, depth, world, _ = frame(shift=(0.03, 0, 0.02))
    rgb[10, 10], depth[10, 10], world[10, 10] = [57, 24, 26], 1.0, [0.03, 0, 0]
    result = track.update(rgb, depth, world, observation_id="second", received_monotonic=10.4, scene_epoch="scene")
    assert not result.mask[10, 10]
    np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0.03, 0, 0.02], atol=1e-4)


@pytest.mark.parametrize("change", ["missing", "distractor", "mixed", "too_fast"])
def test_loss_ambiguity_and_mixed_components_stop_and_latch(change):
    track = tracker()
    kwargs = {change: True} if change != "too_fast" else {"shift": (0.3, 0, 0), "pixel_shift": (30, 0)}
    with pytest.raises(ObservationUnavailable):
        update(track, frame(**kwargs))
    assert track.latest is None
    # Reappearing pixels cannot silently revive a lost semantic identity.
    with pytest.raises(ObservationUnavailable, match="reinitialize"):
        update(track, frame())


@pytest.mark.parametrize(
    "field,value",
    [("observation_id", "first"), ("scene_epoch", "other"), ("received_monotonic", 9.0), ("received_monotonic", 12.0)],
)
def test_replay_epoch_and_capture_ages_never_renew(field, value):
    track = tracker()
    rgb, depth, world, _ = frame()
    args = {"observation_id": "second", "received_monotonic": 10.4, "scene_epoch": "scene"}
    args[field] = value
    with pytest.raises(ObservationUnavailable):
        track.update(rgb, depth, world, **args)
    assert track.latest is None
