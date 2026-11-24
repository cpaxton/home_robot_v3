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
from emet.controller.manipulation.robot_view_guard import RobotViewGuard
from emet.motion.observed_solids import WorldSolid


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


def guarded(measured, *, center=(0.08, 0.0, 0.04)):
    pose = np.eye(4)
    pose[:3, 3] = center
    points = measured.world.copy()
    points[~(np.isfinite(measured.depth) & (measured.depth > 0))] = np.nan
    guard = RobotViewGuard(
        measured.capture,
        measured.depth,
        points,
        measured.world,
        (1,),
        (WorldSolid("sphere", (0.005,), pose),),
        "robot-model",
        "declared-test-calibration",
        "head-pair",
        0.001,
    )
    return replace(measured, robot_guard=guard)


def test_current_grounded_mask_still_requires_a_fresh_geometric_fit():
    track, clock = setup()
    clock[0] = 10.4
    capture = CaptureInterval("new", 10.3, 10.301, 10.31, "clock")
    measured, mask = view(capture, shift=(.01, 0., 0.), distractor=True)
    result = track.update({"head": replace(measured, foreground_mask=mask)}, scene_epoch="scene")
    assert result.capture == capture
    assert result.fit.poses[0, 0, 3] == pytest.approx(.01, abs=.002)
    np.testing.assert_array_equal(dict(result.masks)["head"], mask)

def guarded_setup():
    clock = [10.01]
    first = CaptureInterval("first", 10.0, 10.001, 10.01, "clock")
    measured, mask = view(first)
    tracker = MultiViewRgbdTracker(
        PrimitiveModel("cylinder", (0.02, 0.04)),
        reference_view="head",
        reference=guarded(measured),
        mask=mask,
        object_ref="opaque",
        scene_epoch="scene",
        clock=lambda: clock[0],
        component_policy="complete_prior_union",
        max_update_s=5.0,
        max_evaluations=12000,
    )
    return tracker, clock


def contaminated(capture, *, point=(0.08, 0.0, 0.04)):
    measured, _ = view(capture)
    rgb, depth, world = measured.rgb.copy(), measured.depth.copy(), measured.world.copy()
    rgb[20, 20], depth[20, 20], world[20, 20] = [200, 40, 40], 1.0, point
    return guarded(SynchronizedRgbdView(rgb, depth, world, capture))


def test_robot_interference_rejects_whole_camera_and_keeps_other_measured_view():
    tracker, clock = guarded_setup()
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    clock[0] = 10.31
    bad = contaminated(cap)
    before = (bad.rgb.copy(), bad.depth.copy(), bad.world.copy())
    result = tracker.update({"head": bad, "wrist": view(cap)[0]}, scene_epoch="scene")
    assert [name for name, _ in result.masks] == ["wrist"]
    assert dict(result.view_rejections)["head"] == "rgbd_robot_interference"
    for original, current in zip(before, (bad.rgb, bad.depth, bad.world), strict=True):
        np.testing.assert_array_equal(original, current)


def test_all_view_interference_latches_loss_without_dropping_singleton_pixels():
    tracker, clock = guarded_setup()
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    clock[0] = 10.31
    with pytest.raises(ObservationUnavailable, match="multiview_association_unknown"):
        tracker.update({"head": contaminated(cap)}, scene_epoch="scene")
    assert tracker.latest is None
    assert dict(tracker.last_view_rejections)["head"] == "rgbd_robot_interference"
    with pytest.raises(ObservationUnavailable, match="lost_reinitialize"):
        tracker.update({"head": guarded(view(cap)[0])}, scene_epoch="scene")


def test_nonrobot_foreign_point_still_invalidates_the_complete_union():
    tracker, clock = guarded_setup()
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    clock[0] = 10.31
    with pytest.raises(ObservationUnavailable):
        tracker.update({"head": contaminated(cap, point=(0.12, 0.0, 0.04)), "wrist": view(cap)[0]}, scene_epoch="scene")
    assert tracker.latest is None


@pytest.mark.parametrize(
    "change",
    ["missing", "capture", "model", "calibration", "budget", "depth", "world", "position_budget", "rotation_budget"],
)
def test_interference_profile_or_capture_cannot_silently_change(change):
    tracker, clock = guarded_setup()
    cap = CaptureInterval("second", 10.3, 10.301, 10.31, "clock")
    clock[0] = 10.31
    measured = guarded(view(cap)[0])
    guard = measured.robot_guard
    if change == "missing":
        guard = None
    elif change == "capture":
        guard = replace(guard, capture=replace(cap, observation_id="other"))
    elif change == "model":
        guard = replace(guard, model_signature="other")
    elif change == "calibration":
        guard = replace(guard, calibration_id="other")
    elif change == "budget":
        guard = replace(guard, point_error_m=0.002)
    elif change == "position_budget":
        guard = replace(guard, robot_position_error_m=0.001)
    elif change == "rotation_budget":
        guard = replace(guard, robot_rotation_error_rad=0.001)
    elif change == "world":
        measured = replace(measured, world=measured.world + 0.01)
    else:
        measured = replace(measured, depth=measured.depth + 0.01)
    with pytest.raises(ObservationUnavailable, match="robot_view_"):
        tracker.update({"head": replace(measured, robot_guard=guard)}, scene_epoch="scene")
    assert tracker.latest is None


def test_reachability_uses_maximum_elapsed_time_between_capture_intervals():
    first = CaptureInterval("first", 10.0, 10.2, 10.21, "clock")
    reference, mask = view(first)
    clock = [10.21]
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
    )
    current = CaptureInterval("second", 10.3, 10.6, 10.61, "clock")
    clock[0] = current.received_monotonic
    result = track.update({"head": view(current, shift=(0.12, 0, 0))[0]}, scene_epoch="scene")
    # .12 m is within .25 m/s * .6 s, but outside the incorrect minimum gap.
    np.testing.assert_allclose(result.fit.poses[:, :3, 3], [[0.12, 0, 0]], atol=0.001)


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


@pytest.mark.parametrize("gap, admits_foreign", [(0.884, False), (1.342, True)])
def test_longer_capture_gap_cannot_hide_newly_reachable_foreign_pixels(gap, admits_foreign):
    clock = [10.01]
    first = CaptureInterval("first", 10.0, 10.001, 10.01, "clock")
    measured, object_mask = view(first)
    rgb, depth, world = (a.copy() for a in (measured.rgb, measured.depth, measured.world))
    # The image is unchanged between captures. A larger elapsed-time motion
    # envelope admits two same-colour background pixels outside the seed mask.
    rgb[20, 20:22], depth[20, 20:22] = [200, 40, 40], 1.0
    world[20, 20:22] = [[0.30, 0, 0.04], [0.301, 0.003, 0.04]]
    reference = SynchronizedRgbdView(rgb, depth, world, first)
    track = MultiViewRgbdTracker(
        PrimitiveModel("cylinder", (0.02, 0.04)),
        reference_view="head",
        reference=reference,
        mask=object_mask,
        object_ref="opaque",
        scene_epoch="scene",
        clock=lambda: clock[0],
        component_policy="complete_prior_union",
        max_update_s=5.0,
        max_evaluations=12000,
    )
    start = first.latest + gap
    capture = CaptureInterval("second", start, start + 0.001, start + 0.01, "clock")
    clock[0] = capture.received_monotonic
    current = replace(reference, capture=capture)
    if admits_foreign:
        with pytest.raises(ObservationUnavailable):
            track.update({"head": current}, scene_epoch="scene")
        assert track.latest is None
        with pytest.raises(ObservationUnavailable, match="lost_reinitialize"):
            track.update({"head": current}, scene_epoch="scene")
    else:
        result = track.update({"head": current}, scene_epoch="scene")
        np.testing.assert_array_equal(dict(result.masks)["head"], object_mask)


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


@pytest.mark.parametrize('fallback', [False, True])
def test_qualified_view_fallback_retains_capture_when_fused_partial_views_fail(monkeypatch, fallback):
    from emet.controller.manipulation import multiview_rgbd_tracking as native

    track, clock = setup(single_view_fallback=fallback, max_starts=3, max_fit_points=100)
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    head, mask = view(cap)
    wrist, _ = view(cap, shift=(.002, 0., 0.))
    fit = native.fit_geometry

    def partial_only(points, *args, **kwargs):
        if len(points) > int(mask.sum()):
            raise ObservationUnavailable('geometry_tracking_no_match')
        return fit(points, *args, **kwargs)

    monkeypatch.setattr(native, 'fit_geometry', partial_only)
    if not fallback:
        with pytest.raises(ObservationUnavailable, match='no_match'):
            track.update({'head': head, 'wrist': wrist}, scene_epoch='scene')
        assert track.latest is None
    else:
        result = track.update({'head': head, 'wrist': wrist}, scene_epoch='scene')
        assert result.capture is cap and result.fit.received_monotonic == cap.received_monotonic
        assert len(result.masks) == 1 and len(result.view_rejections) == 1
        assert np.linalg.norm(result.fit.poses[0, :3, 3]) <= .003


@pytest.mark.parametrize('best_view', [False, True])
def test_view_fallback_rejects_independently_qualified_camera_pose_disagreement(monkeypatch, best_view):
    from emet.controller.manipulation import multiview_rgbd_tracking as native

    track, clock = setup(single_view_fallback=True, best_fit_view=best_view, max_starts=3, max_fit_points=100)
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    head, mask = view(cap)
    wrist, _ = view(cap, shift=(.025, 0., 0.))
    fit = native.fit_geometry

    def partial_only(points, *args, **kwargs):
        if len(points) > int(mask.sum()):
            raise ObservationUnavailable('geometry_tracking_no_match')
        return fit(points, *args, **kwargs)

    monkeypatch.setattr(native, 'fit_geometry', partial_only)
    if not best_view:
        with pytest.raises(ObservationUnavailable, match='ambiguous'):
            track.update({'head': head, 'wrist': wrist}, scene_epoch='scene')
        assert track.latest is None
    else:
        result = track.update({'head': head, 'wrist': wrist}, scene_epoch='scene')
        assert result.capture is cap
        assert track.view_selections == 1
        assert {name for name, _ in track.last_view_candidates} == {'head', 'wrist'}
        assert any(reason == 'diagnostic_conflicting_view_not_selected' for _, reason in track.last_view_rejections)


@pytest.mark.parametrize('best_fit', [False, True])
def test_best_fit_hypothesis_is_explicit_and_preserves_full_candidate_trace(monkeypatch, best_fit):
    from emet.controller.manipulation import multiview_rgbd_tracking as native

    track, clock = setup(best_fit_hypothesis=best_fit)
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    full, _ = view(cap)
    raw_fit = native.fit_geometry

    def ambiguous(*args, **kwargs):
        fit = raw_fit(*args, **kwargs)
        poses = np.repeat(fit.poses[:1], 2, axis=0)
        poses[1, 0, 3] += .02
        return replace(fit, poses=poses, rms_m=np.array([.0001, .001]), nullspaces=(fit.nullspaces[0],)*2)

    monkeypatch.setattr(native, 'fit_geometry', ambiguous)
    if not best_fit:
        with pytest.raises(ObservationUnavailable, match='moving_geometry_ambiguous'):
            track.update({'head': full}, scene_epoch='scene')
    else:
        result = track.update({'head': full}, scene_epoch='scene')
        assert len(result.fit.poses) == 1 and len(track.last_candidate_fit.poses) == 2
        assert result.fit.received_monotonic == cap.received_monotonic
        assert track.hypothesis_selections == 1
        assert not result.execution_qualified


@pytest.mark.parametrize('best_fit', [False, True])
def test_best_fit_initialization_uses_the_same_explicit_policy(monkeypatch, best_fit):
    from emet.controller.manipulation import moving_rgbd_tracking as native

    raw_fit = native.fit_geometry

    def ambiguous(*args, **kwargs):
        fit = raw_fit(*args, **kwargs)
        poses = np.repeat(fit.poses[:1], 2, axis=0)
        poses[1, 0, 3] += .02
        return replace(fit, poses=poses, rms_m=np.array([.0001, .001]), nullspaces=(fit.nullspaces[0],)*2)

    monkeypatch.setattr(native, 'fit_geometry', ambiguous)
    if not best_fit:
        with pytest.raises(ObservationUnavailable, match='moving_geometry_ambiguous'):
            setup(best_fit_hypothesis=False)
    else:
        track, _ = setup(best_fit_hypothesis=True)
        assert len(track.latest.fit.poses) == 1 and len(track.last_candidate_fit.poses) == 2
        assert track.latest.fit.received_monotonic == track.last_candidate_fit.received_monotonic
        assert track.hypothesis_selections == 1 and not track.latest.execution_qualified


@pytest.mark.parametrize('witness_covers_object', [False, True])
def test_live_geometry_window_keeps_appearance_identity_near_gripper(monkeypatch, witness_covers_object):
    from types import SimpleNamespace

    from emet.controller.task.tamp import live_transfer as live

    tracker, clock = setup()
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    measured, mask = view(cap)
    rgb = measured.rgb.copy()
    white = np.zeros(mask.shape, bool)
    for y, x in np.argwhere(mask)[100:180]:
        white[y, x] = True
    rgb[white] = [240, 240, 240]  # Same search volume, different material.
    measured = replace(measured, rgb=rgb)
    sample = SimpleNamespace(capture=cap, observations={'head': measured})
    scene = live.LiveTransferScene.__new__(live.LiveTransferScene)
    scene.profile = {'tracking_geometry_window_m': .015, 'tracking_views': ['head']}
    scene.read_sample = lambda _: sample
    scene.epoch, scene.initial_capture_id = 'scene', 'first'
    scene.initial_views, scene.samples, scene.diagnostics = {}, {}, {}
    scene.supports = [SimpleNamespace(patch=SimpleNamespace(plane=np.array([0., 0., 0.]),
        bounds=np.array([[1., 1., -.001], [2., 2., .001]])),
        measured_surface_bounds=np.array([[-.1, -.1, -.001], [.1, .1, .001]]) if witness_covers_object else
                                np.array([[1., 1., -.001], [2., 2., .001]]))]
    scene.stream, scene.payload_model, scene.enclosure = SimpleNamespace(tracker=tracker), tracker.model, None
    monkeypatch.setattr(live, 'synchronized_views', lambda sample: sample.observations)
    result = scene._read_views()['head']
    assert result.foreground_mask.sum() >= 25
    assert not np.any(result.foreground_mask & white)
    assert result.capture is cap
    assert bool(result.foreground_mask[55, 60]) is (not witness_covers_object)


@pytest.mark.parametrize('residual', [.003, .006])
def test_explicit_fit_residual_keeps_all_points_and_actual_capture(residual):
    tracker, clock = setup(max_residual_m=residual)
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    measured, mask = view(cap)
    world = measured.world.copy()
    world[55, 60, 1] -= .016  # A measured depth error, not a competing component.
    measured = replace(measured, world=world, foreground_mask=mask)
    if residual == .003:
        with pytest.raises(ObservationUnavailable, match='geometry_tracking_no_match'):
            tracker.update({'head': measured}, scene_epoch='scene')
    else:
        result = tracker.update({'head': measured}, scene_epoch='scene')
        assert result.capture is cap
        assert result.masks[0][1][55, 60]  # No trimming the difficult measurement.
        assert result.fit.received_monotonic == cap.received_monotonic


@pytest.mark.parametrize('shift', [.008, .06])
def test_attachment_conditioned_fit_uses_current_depth_and_rejects_contradictions(shift):
    tracker, clock = setup()
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    measured, mask = view(cap, shift=(shift, 0., 0.))
    measured = replace(measured, foreground_mask=mask, pose_prior=np.eye(4))
    if shift > .02:
        with pytest.raises(ObservationUnavailable):
            tracker.update({'head': measured}, scene_epoch='scene')
    else:
        result = tracker.update({'head': measured}, scene_epoch='scene')
        assert result.capture is cap
        assert result.fit.conditioning == 'rigid_attachment_orientation'
        assert not result.fit.orientation_observable
        assert tracker.conditional_capture_updates == 1
        np.testing.assert_allclose(result.fit.poses[0, :3, 3], [shift, 0., 0.], atol=.001)
        np.testing.assert_allclose(result.fit.poses[0, :3, :3], np.eye(3))


def test_attachment_prior_must_match_across_camera_views():
    tracker, clock = setup()
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    clock[0] = 10.5
    measured, mask = view(cap)
    with pytest.raises(ObservationUnavailable, match='multiview_pose_prior_mismatch'):
        tracker.update({'head': replace(measured, foreground_mask=mask, pose_prior=np.eye(4)),
                        'wrist': replace(measured, foreground_mask=mask)}, scene_epoch='scene')


@pytest.mark.parametrize('lost', [False, True])
def test_carry_prediction_uses_capture_tool_motion_and_checks_gripper_aperture(monkeypatch, lost):
    from types import SimpleNamespace

    import mujoco

    from emet.controller.task.tamp import live_transfer as live

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
        <joint name="left" type="slide" axis="1 0 0"/><geom size=".01"/>
        <body><joint name="right" type="slide" axis="0 1 0"/><geom size=".01"/></body>
        </body></worldbody></mujoco>''')
    old_cap = CaptureInterval('held', 10., 10.001, 10.01, 'clock')
    packet = SimpleNamespace(fit=SimpleNamespace(capture=old_cap))
    robot = SimpleNamespace(joints=np.array([.038, -.038]), world_from_ee=np.eye(4))
    scene = live.LiveTransferScene.__new__(live.LiveTransferScene)
    scene.profile, scene.arm = {'simulation_rigid_carry_prediction': True}, SimpleNamespace(model=model)
    waits = []
    scene.stream = SimpleNamespace(snapshot=lambda: packet, collect=lambda **kwargs: waits.append(kwargs))
    scene.clock = lambda: 10.1
    scene.samples = {'held': SimpleNamespace(robot_after=robot)}
    scene.diagnostics = {}
    scene.set_carry_prediction(np.eye(4), {'left': .012, 'right': -.012},
                               measured_gripper={'left': .022, 'right': -.022})
    assert waits[0]['after'] == 10.1 and waits[0]['after'] > old_cap.latest
    cap = CaptureInterval('moving', 10.3, 10.301, 10.31, 'clock')
    measured, _ = view(cap)
    ee = np.eye(4)
    ee[0, 3] = .1
    robot = SimpleNamespace(joints=np.array([.012, -.012]) if lost else np.array([.022, -.022]),
                            world_from_ee=ee)
    sample = SimpleNamespace(capture=cap, observations={'head': measured}, robot_after=robot)
    scene.read_sample, scene.epoch, scene.initial_views = lambda _: sample, 'scene', None
    monkeypatch.setattr(live, 'synchronized_views', lambda sample: sample.observations)
    if lost:
        with pytest.raises(ObservationUnavailable, match='carry_grip_aperture_lost'):
            scene._read_views()
    else:
        result = scene._read_views()['head']
        np.testing.assert_allclose(result.pose_prior, ee)
        assert result.capture is cap
    scene.set_carry_prediction(None, {})
    assert scene._carry_prediction is None


def test_carry_foreground_associates_actual_surface_instead_of_enclosure_interior(monkeypatch):
    from types import SimpleNamespace

    from emet.controller.task.tamp import live_transfer as live

    tracker, clock = setup()
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    measured, mask = view(cap)
    world = measured.world.copy()
    world[55, 60] = 0.  # Same appearance, deep inside the enclosing solid.
    measured = replace(measured, world=world)
    robot = SimpleNamespace(joints=np.array([.022, -.022]), world_from_ee=np.eye(4))
    sample = SimpleNamespace(capture=cap, observations={'head': measured}, robot_after=robot)
    scene = live.LiveTransferScene.__new__(live.LiveTransferScene)
    scene.profile = {'tracking_geometry_window_m': .015, 'tracking_views': ['head'],
                     'depth_margin_m': .003, 'position_error_m': .005}
    scene.read_sample = lambda _: sample
    scene.epoch, scene.initial_capture_id = 'scene', 'first'
    scene.initial_views, scene.samples, scene.supports, scene.diagnostics = {}, {}, [], {}
    scene.stream, scene.payload_model, scene.enclosure = SimpleNamespace(tracker=tracker), tracker.model, tracker.model
    scene._carry_prediction = (np.eye(4), np.array([0, 1]), np.array([.022, -.022]), np.array([.012, -.012]))
    scene._carry_prediction_barrier = 10.
    monkeypatch.setattr(live, 'synchronized_views', lambda sample: sample.observations)
    result = scene._read_views()['head']
    assert not result.foreground_mask[55, 60]
    assert result.foreground_mask.sum() >= 25
    assert result.pose_prior is not None and result.capture is cap


def test_live_tracking_excludes_capture_aligned_self_returns_with_object_appearance(monkeypatch):
    from types import SimpleNamespace

    from emet.controller.task.tamp import live_transfer as live

    tracker, clock = setup()
    cap = CaptureInterval('second', 10.3, 10.301, 10.31, 'clock')
    measured, mask = view(cap)
    # Simulate qualified self returns within the same appearance/search region.
    material = SimpleNamespace(matches_surface=lambda points, **kwargs: points[:, 0] > 0.)
    sample = SimpleNamespace(capture=cap, observations={'head': measured}, material=material)
    scene = live.LiveTransferScene.__new__(live.LiveTransferScene)
    scene.profile = {'tracking_geometry_window_m': .015, 'tracking_views': ['head'], 'depth_margin_m': .003}
    scene.read_sample = lambda _: sample
    scene.epoch, scene.initial_capture_id = 'scene', 'first'
    scene.initial_views, scene.samples, scene.supports, scene.diagnostics = {}, {}, [], {}
    scene.stream, scene.payload_model, scene.enclosure = SimpleNamespace(tracker=tracker), tracker.model, None
    monkeypatch.setattr(live, 'synchronized_views', lambda sample: sample.observations)
    result = scene._read_views()['head']
    assert result.foreground_mask.sum() >= 25
    assert not np.any(result.foreground_mask & (measured.world[..., 0] > 0.))
    assert result.capture is cap
