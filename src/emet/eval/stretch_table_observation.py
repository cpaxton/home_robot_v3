# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).
"""Admission diagnostics for the colored Stretch tabletop development fixture."""

import time

import numpy as np
from scipy.ndimage import label

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.rigid_rgbd_tracking import reference_features
from emet.controller.manipulation.stretch_encoder_kinematics import estimate_stretch_ee_from_encoders
from emet.controller.task.tamp.observed_capture import capture_arrays
from emet.motion.observed_object_bound import enclose_observed_object
from emet.motion.observed_support import fit_observed_support
from emet.motion.observed_workspace import ObservedWorkspace


def _measure_target(observation, observation_id):
    rgb, depth, world = capture_arrays(observation)
    red, green, blue = rgb.astype(np.int16).transpose(2, 0, 1)
    valid = np.isfinite(depth) & (depth > .1) & (depth <= 2.5)
    destination = valid & (green > 100) & (green > 2 * red) & (green > 2 * blue)
    report = {}
    try:
        patch = fit_observed_support(
            observation, destination, observation_id=observation_id, measurement_margin_m=.004
        )
    except ObservationUnavailable as error:
        return {"destination": {"status": "blocked", "reason": str(error)}}, None
    report["destination"] = {
        "status": "measured",
        "visible_points": int(destination.sum()),
        "bounds_m": patch.bounds.tolist(),
        "plane_residual_m": patch.residual_m,
    }

    height = world[..., 2] - (world[..., :2] @ patch.plane[:2] + patch.plane[2])
    target_candidates = valid & (red > 100) & (red > 3 * green) & (red > 3 * blue)
    components, _ = label(target_candidates & (height > patch.measurement_margin_m))
    counts = np.bincount(components.ravel())[1:]
    candidates = np.flatnonzero(counts >= 25) + 1
    if len(candidates) != 1:
        report["target"] = {"status": "blocked", "reason": "target_missing_or_ambiguous"}
        return report, None
    mask = components == candidates[0]
    points = world[mask]
    report["target"] = {
        "status": "visible_surface",
        "visible_points": len(points),
        "visible_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
    }
    try:
        _, pixels = reference_features(rgb, mask)
        report["rigid_reference"] = {"status": "candidate", "landmarks": len(pixels)}
    except ObservationUnavailable as error:
        report["rigid_reference"] = {"status": "blocked", "reason": str(error)}
    measured = (observation, observation_id, depth, world[valid], mask[valid])
    return report, measured


def _enclose_views(views):
    # Bound the measured neighborhood, not the object. A possible component
    # reaching any grid edge is still unbounded under the free-space model.
    resolution = .015
    origins = np.stack([np.median(points[mask], axis=0) for _, _, _, points, mask in views])
    center = np.round(np.median(origins, axis=0) / resolution) * resolution
    if np.max(np.linalg.norm(origins - origins[0], axis=1)) > .06:
        return {"status": "blocked", "reason": "static_target_association_inconsistent"}
    bounds = (center - 12 * resolution, center + 12 * resolution)
    grids = [
        ObservedWorkspace.from_depth(depth, obs.camera_K, obs.camera_pose, workspace=bounds, resolution=resolution)
        for obs, _, depth, _, _ in views
    ]
    grid = ObservedWorkspace.fuse_static(grids) if len(grids) > 1 else grids[0]
    points = np.concatenate([points for _, _, _, points, _ in views])
    mask = np.concatenate([mask for _, _, _, _, mask in views])
    partition = grid.partition(points, np.where(mask, "target", "background"))
    try:
        bound = enclose_observed_object(
            partition, "target", observation_ids=tuple(row[1] for row in views),
            received_monotonic=min(row[0].received_monotonic for row in views),
            scene_revision="static-table-admission-control", topology="connected_rigid",
        )
        return {"status": "candidate", "bounds_m": bound.bounds_world.tolist()}
    except ObservationUnavailable as error:
        return {"status": "blocked", "reason": str(error)}


def assess_table_geometry(observation, *, observation_id):
    """Measure visible reference geometry with no scene mesh or object dimensions."""
    report, measured = _measure_target(observation, observation_id)
    if measured is not None:
        report["payload_enclosure"] = _enclose_views((measured,))
    return report


def assess_static_table_views(views):
    """Audit enclosure from an explicitly static fixture and distinct camera views."""
    if len(views) < 2 or len({capture_id for _, capture_id in views}) != len(views):
        raise ValueError("distinct_table_captures_required")
    measured = [_measure_target(observation, capture_id) for observation, capture_id in views]
    positions = [np.asarray(observation.camera_pose)[:3, 3] for observation, _ in views]
    enclosure = (
        _enclose_views(tuple(row for _, row in measured)) if all(row is not None for _, row in measured)
        else {"status": "blocked", "reason": "target_association_missing_in_scan"}
    )
    return {"view_reports": [report for report, _ in measured],
            "camera_positions_m": [position.tolist() for position in positions],
            "baseline_m": max(float(np.linalg.norm(a - b)) for a in positions for b in positions),
            "payload_enclosure": enclosure}


def assess_live_table_capture(observation, robot_model, *, now=None):
    """Check measured motor alignment and scene geometry from the same capture."""
    now = time.monotonic() if now is None else now
    try:
        estimate = estimate_stretch_ee_from_encoders(observation, robot_model, now=now)
    except ObservationUnavailable as error:
        return {"encoder_capture": {"status": "blocked", "reason": str(error)}}
    capture = estimate.motor_capture.capture
    return {
        "encoder_capture": {
            "status": "passed",
            "observation_id": capture.observation_id,
            "capture_span_s": capture.latest - capture.earliest,
            "source_age_s": now - capture.earliest,
            "world_from_ee": estimate.world_from_ee.tolist(),
            "unmeasured_model_joints": estimate.inferred_joint_names,
        },
        **assess_table_geometry(observation, observation_id=capture.observation_id),
    }
