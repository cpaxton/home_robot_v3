# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Cross-camera continuity requires current measured surfaces and shared capture."""

from dataclasses import replace

import numpy as np
import pytest

from emet.controller.manipulation.geometry_models import PrimitiveModel
from emet.controller.manipulation.multiview_rgbd_tracking import MultiViewRgbdTracker, SynchronizedRgbdView
from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import CaptureInterval


def view(capture, *, shift=(0.0, 0.0, 0.0), cap_only=False, missing=False, distractor=False, mixed=False):
    rgb = np.full((100, 180, 3), [80, 110, 140], np.uint8)
    depth = np.full((100, 180), 2.0)
    y, x = np.mgrid[:100, :180]
    world = np.stack((x * 0.004, np.ones_like(x), y * 0.004), -1).astype(float)
    mask = np.zeros(depth.shape, bool)
    # Independently sampled cylinder side and cap in adjoining image regions.
    for r in range(-15, 16):
        for c in range(-9, 10):
            if cap_only:
                continue
            xx = c / 9 * 0.019
            mask[55 + r, 60 + c] = True
            world[55 + r, 60 + c] = np.array([xx, -np.sqrt(0.02**2 - xx**2), -r / 15 * 0.039]) + shift
    for r in range(1, 6):
        yy = -0.02 + r * 0.006
        width = int(np.sqrt(0.02**2 - yy**2) / 0.002)
        for c in range(-width, width + 1):
            mask[40 - r, 60 + c] = True
            world[40 - r, 60 + c] = np.array([c * 0.002, yy, 0.04]) + shift
    rgb[mask], depth[mask] = [200, 40, 40], 1.0
    if distractor:
        for yy, xx in np.argwhere(mask):
            rgb[yy, xx + 45], depth[yy, xx + 45] = [200, 40, 40], 1.0
            world[yy, xx + 45] = world[yy, xx] + [0.08, 0, 0]
    if mixed:
        for c in range(1, 15):
            yy, xx = 55, 69 + c
            rgb[yy, xx], depth[yy, xx] = [200, 40, 40], 1.0
            world[yy, xx] = world[55, 69] + [c * 0.005, 0, 0]
    if missing:
        rgb[mask], depth[mask], world[mask] = [20, 20, 20], 0.0, np.nan
    return SynchronizedRgbdView(rgb, depth, world, capture), mask


def setup(**kwargs):
    clock = [10.1]
    capture = CaptureInterval("first", 10.0, 10.001, 10.01, "clock")
    reference, mask = view(capture)
    track = MultiViewRgbdTracker(
        PrimitiveModel("cylinder", (0.02, 0.04)),
        reference_view="head",
        reference=reference,
        mask=mask,
        object_ref="opaque",
        scene_epoch="scene",
        clock=lambda: clock[0],
        max_update_s=5.0,
        max_evaluations=12000,
        **kwargs,
    )
    return track, clock


def test_handoff_to_wrist_preserves_reference_and_cannot_invent_per_view_pose():
    track, clock = setup()
    reference = track.reference_id
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    result = track.update(
        {"head": view(cap, shift=(0.03, 0, 0.02), cap_only=True)[0], "wrist": view(cap, shift=(0.03, 0, 0.02))[0]},
        scene_epoch="scene",
    )
    np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0.03, 0, 0.02], atol=1e-4)
    assert set(dict(result.masks)) == {"head", "wrist"}
    cap = CaptureInterval("third", 10.6, 10.601, 10.61, "clock")
    clock[0] = 10.8
    result = track.update(
        {"head": view(cap, missing=True)[0], "wrist": view(cap, shift=(0.06, 0, 0.04))[0]}, scene_epoch="scene"
    )
    np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0.06, 0, 0.04], atol=1e-4)
    assert list(dict(result.masks)) == ["wrist"] and result.reference_id == reference
    assert result.capture == cap and not result.execution_qualified
    with pytest.raises(ValueError):
        result.masks[0][1].setflags(write=True)


def test_ambiguous_camera_cannot_contribute_points_but_other_camera_may_continue():
    track, clock = setup()
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    result = track.update({"head": view(cap, distractor=True)[0], "wrist": view(cap)[0]}, scene_epoch="scene")
    assert list(dict(result.masks)) == ["wrist"]
    assert dict(result.view_rejections)["head"] == "moving_mask_ambiguous"


@pytest.mark.parametrize("policy", ["single_component", "complete_prior_union"])
def test_disconnected_measured_fragments_use_all_points_only_with_explicit_prior_policy(policy):
    track, clock = setup(component_policy=policy)
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    full, mask = view(cap)
    rgb, depth, world = (a.copy() for a in (full.rgb, full.depth, full.world))
    # Measured depth holes split the object. No pixels/points may be filled in.
    rgb[48:53], depth[48:53], world[48:53] = [20, 20, 20], 0.0, np.nan
    fragmented = SynchronizedRgbdView(rgb, depth, world, cap)
    if policy == "single_component":
        with pytest.raises(ObservationUnavailable, match="association_unknown"):
            track.update({"wrist": fragmented}, scene_epoch="scene")
    else:
        result = track.update({"wrist": fragmented}, scene_epoch="scene")
        expected = mask.copy()
        expected[48:53] = False
        np.testing.assert_array_equal(dict(result.masks)["wrist"], expected)
        np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0, 0, 0], atol=1e-4)
        assert result.as_dict()["component_policy"] == "complete_prior_union"


@pytest.mark.parametrize("foreign", ["distractor", "mixed", "single_outlier"])
def test_complete_prior_union_never_selects_a_subset_to_hide_foreign_geometry(foreign):
    track, clock = setup(component_policy="complete_prior_union")
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    measured, _ = view(cap, distractor=foreign == "distractor", mixed=foreign == "mixed")
    if foreign == "single_outlier":
        rgb, depth, world = (a.copy() for a in (measured.rgb, measured.depth, measured.world))
        rgb[20, 20], depth[20, 20], world[20, 20] = [200, 40, 40], 1.0, [0.08, 0, 0.04]
        measured = SynchronizedRgbdView(rgb, depth, world, cap)
    with pytest.raises(ObservationUnavailable):
        track.update({"wrist": measured}, scene_epoch="scene")
    assert track.latest is None


@pytest.mark.parametrize("appearance_model", ["chromaticity", "neutral_specular"])
def test_neutral_highlight_can_preserve_cap_without_loosening_geometry_checks(appearance_model):
    track, clock = setup(appearance_model=appearance_model)
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    full, mask = view(cap)
    rgb = full.rgb.copy()
    rgb[mask] = [255, 100, 100]  # Neutral highlight and diffuse gain.
    highlighted = SynchronizedRgbdView(rgb, full.depth, full.world, cap)
    if appearance_model == "chromaticity":
        with pytest.raises(ObservationUnavailable):
            track.update({"wrist": highlighted}, scene_epoch="scene")
    else:
        result = track.update({"wrist": highlighted}, scene_epoch="scene")
        np.testing.assert_array_equal(dict(result.masks)["wrist"], mask)
        assert result.as_dict()["appearance_model"] == "neutral_specular"
        np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0, 0, 0], atol=1e-4)


@pytest.mark.parametrize("failure", ["near_grey", "same_hue_distractor", "cap_only"])
def test_highlight_profile_does_not_authorize_identity_or_geometry_fallback(failure):
    track, clock = setup(appearance_model="neutral_specular", component_policy="complete_prior_union")
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    full, mask = view(cap, distractor=failure == "same_hue_distractor", cap_only=failure == "cap_only")
    rgb = full.rgb.copy()
    if failure == "near_grey":
        rgb[mask] = [90, 89, 89]  # Tiny red tint must not expand to a red target.
    with pytest.raises(ObservationUnavailable):
        track.update({"wrist": SynchronizedRgbdView(rgb, full.depth, full.world, cap)}, scene_epoch="scene")
    assert track.latest is None


@pytest.mark.parametrize("allow_clipped", [False, True])
def test_clipped_surface_needs_explicit_prior_profile_and_still_requires_observability(allow_clipped):
    track, clock = setup(allow_clipped_views=allow_clipped)
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    full, _ = view(cap)
    arrays = [a.copy() for a in (full.rgb, full.depth, full.world)]
    for a in arrays:
        a[35:] = a[:-35].copy()
    arrays[0][:35], arrays[1][:35], arrays[2][:35] = [80, 110, 140], 2.0, 1.0
    clipped = SynchronizedRgbdView(*arrays, cap)
    if not allow_clipped:
        with pytest.raises(ObservationUnavailable):
            track.update({"wrist": clipped}, scene_epoch="scene")
        assert dict(track.last_view_rejections)["wrist"] == "moving_mask_image_boundary"
    else:
        result = track.update({"wrist": clipped}, scene_epoch="scene")
        np.testing.assert_allclose(result.fit.poses[0, :3, 3], [0, 0, 0], atol=1e-4)
        assert result.clipped_views == ("wrist",) and not result.execution_qualified
        cap = CaptureInterval("third", 10.6, 10.601, 10.61, "clock")
        clock[0] = 10.8
        with pytest.raises(ObservationUnavailable):
            track.update({"wrist": view(cap, cap_only=True)[0]}, scene_epoch="scene")
        assert track.latest is None


@pytest.mark.parametrize("change", ["all_lost", "mixed", "only_cap", "replayed", "mismatched_capture"])
def test_global_unknown_clears_and_latches_without_reidentification(change):
    track, clock = setup()
    clock[0] = 10.5
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    head = view(cap, missing=change == "all_lost", cap_only=change == "only_cap", mixed=change == "mixed")[0]
    wrist = view(cap, missing=change in ("all_lost", "only_cap"))[0]
    if change == "replayed":
        head, wrist = replace(head, capture=track.latest.capture), replace(wrist, capture=track.latest.capture)
    if change == "mismatched_capture":
        wrist = replace(wrist, capture=replace(cap, observation_id="cached-other-frame"))
    with pytest.raises(ObservationUnavailable):
        track.update({"head": head, "wrist": wrist}, scene_epoch="scene")
    assert track.latest is None
    with pytest.raises(ObservationUnavailable, match="reinitialize"):
        track.update({"head": view(cap)[0]}, scene_epoch="scene")
