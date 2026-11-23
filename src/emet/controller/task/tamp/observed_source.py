# Copyright (c) Chris Paxton 2026
# Licensed under the Apache License, Version 2.0 (see LICENSE in the repository root).

"""Join tracked payloads and independently qualified scene evidence."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from emet.controller.manipulation.native_inputs import ObservationUnavailable
from emet.controller.manipulation.observed_effects import VisualEffectFrame
from emet.motion.observed_admission import CapturedRobotMaterial, ObservedMaterialAdmission
from emet.motion.observed_background import ObservedBackground, StaticBackgroundAssumption
from emet.motion.observed_contact import ObservedTargetContact
from emet.motion.observed_support_contact import ObservedSupportContact

from .observed_adapter import ObservedNativeSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class ObservedTransferEvidence:
    frame: VisualEffectFrame
    background: ObservedBackground
    robot_material: CapturedRobotMaterial
    source: ObservedSupportContact
    destination: ObservedSupportContact
    query_robot_geometry: "Callable"
    static_assumption: StaticBackgroundAssumption | None = None
    local_scene: object | None = None

    def snapshot(self, *, object_ref, support_ref, model_signature, fingertip_geom_ids, clock):
        now = clock()
        self.frame.require(now=now)
        if self.frame.payload.object_ref != object_ref or self.destination.support_ref != support_ref:
            raise ObservationUnavailable("payload_identity_mismatch")
        if self.robot_material.capture != self.frame.capture:
            raise ObservationUnavailable("capture_identity_mismatch")
        self.robot_material.require(now=now, model_signature=model_signature, scene_epoch=self.background.scene_epoch)
        for contact in (self.source, self.destination):
            contact.require(self.background, now=now, static_assumption=self.static_assumption)
        scene = self.background.placement_scene(static_assumption=self.static_assumption, clock=clock)
        admission = ObservedMaterialAdmission(
            self.background,
            self.robot_material,
            static_assumption=self.static_assumption,
            query_robot_geometry=self.query_robot_geometry,
        )
        target = ObservedTargetContact.from_frame(
            self.frame,
            model_signature=model_signature,
            calibration_id=self.robot_material.configuration_profile_id,
            fingertip_geom_ids=fingertip_geom_ids,
            max_penetration_m=0.005,
        )
        return ObservedNativeSnapshot(
            frame=self.frame,
            scene=scene,
            scene_revision=self.background.geometry_revision,
            support_bounds=self.destination.patch.bounds,
            admission=admission,
            target_contact=target,
            support_contact=self.destination,
            source_support_bounds=self.source.patch.bounds,
            source_support_contact=self.source,
            local_scene=self.local_scene,
        )


class ObservedTransferSource:
    """Read a coherent scene producer without refreshing capture ages on demand.

    The producer owns semantic tracking, calibrated self-material, background
    fusion and clear support selection. Its callbacks return original measured
    evidence. A model, camera or scene change must invalidate that producer.
    """

    input_mode = "observed"

    def __init__(
        self, *, read, collect, current_revision, invalidate, model_signature, fingertip_geom_ids,
        check_placement_surface=None, clock=time.monotonic,
    ):
        self._read = read
        self._collect = collect
        self.current_revision = current_revision
        self.invalidate = invalidate
        self.check_placement_surface = check_placement_surface
        self._model_signature = model_signature
        self._fingertip_geom_ids = tuple(fingertip_geom_ids)
        self._clock = clock

    def snapshot(self, object_ref, support_ref):
        bundle = self._read(object_ref, support_ref)
        if not isinstance(bundle, ObservedTransferEvidence):
            raise ObservationUnavailable("observed_scene_unavailable")
        snapshot = bundle.snapshot(
            object_ref=object_ref,
            support_ref=support_ref,
            model_signature=self._model_signature,
            fingertip_geom_ids=self._fingertip_geom_ids,
            clock=self._clock,
        )
        if snapshot.scene_revision != self.current_revision():
            raise ObservationUnavailable("scene_revision_changed")
        return snapshot

    def collect(self, object_ref, support_ref, *, after):
        frames = tuple(self._collect(object_ref, support_ref, after=after))
        observed_ids = set()
        for frame in frames:
            if not isinstance(frame, VisualEffectFrame) or frame.payload.object_ref != object_ref:
                raise ObservationUnavailable("payload_identity_mismatch")
            frame.require(now=self._clock(), after=after)
            if frame.capture.observation_id in observed_ids:
                raise ObservationUnavailable("capture_replayed")
            observed_ids.add(frame.capture.observation_id)
        return frames


def partition_transfer_depth(observation, *, masks, workspace, resolution_m, depth_margin_m, current_grid=None,
                             require_surface_support=False):
    """Carve from measured rays, keeping ambiguous endpoint labels as obstacles."""
    import numpy as np

    from emet.motion.observed_workspace import ObservedWorkspace

    from .observed_capture import capture_arrays

    _, depth, world = capture_arrays(observation)
    valid = np.isfinite(depth) & (depth >= 0.1) & (depth <= 5.0)
    if require_surface_support:
        from emet.motion.depth_surface_validity import supported_depth

        valid &= supported_depth(depth)
        depth = np.where(valid, depth, 0.)
    if any(not isinstance(ref, str) or not ref or ref == "background" for ref in masks):
        raise ValueError("invalid_occupancy_provenance")
    width = max(map(len, ("background", *masks)))
    labels = np.full(depth.shape, "background", dtype=f"U{width}")
    used = np.zeros(depth.shape, dtype=bool)
    for ref, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != depth.shape or mask.dtype != bool:
            raise ValueError("invalid_occupancy_provenance")
        selected = valid & mask
        labels[selected & ~used] = ref
        labels[selected & used] = "background"
        used |= selected
    if current_grid is None:
        grid = ObservedWorkspace.from_depth(
            depth, observation.camera_K, observation.camera_pose, workspace=workspace,
            resolution=resolution_m, depth_margin_m=depth_margin_m,
        )
    else:
        # Recheck a static plan with measured endpoints. This adds obstacles
        # and checks contradictions without rerunning the full free-ray carve
        # or renewing the age of any previously observed free cell.
        cells = np.zeros_like(current_grid.cells)
        coordinates = np.floor((world[valid] - current_grid.origin) / current_grid.resolution).astype(int)
        inside = np.all((coordinates >= 0) & (coordinates < cells.shape), axis=1)
        cells[tuple(coordinates[inside].T)] = 2
        grid = ObservedWorkspace(current_grid.origin, current_grid.resolution, cells)
    return grid.partition(world[valid], labels[valid])


def fit_transfer_support(
    observation,
    mask,
    *,
    capture,
    object_ref,
    support_ref,
    scene_epoch,
    calibration_id,
    measurement_margin_m,
    payload_motion_error_m,
    backing_depth_m=0.0,
    backing_prior_id="",
):
    """Fit one finite support and bind its original RGB-D rays to the contact witness."""
    from emet.motion.observed_support import fit_observed_support

    if observation.received_monotonic != capture.received_monotonic:
        raise ObservationUnavailable("capture_identity_mismatch")
    patch = fit_observed_support(
        observation, mask, observation_id=capture.observation_id, measurement_margin_m=measurement_margin_m
    )
    return ObservedSupportContact(
        object_ref,
        support_ref,
        scene_epoch,
        calibration_id,
        patch,
        capture,
        observation.depth,
        observation.camera_K,
        observation.camera_pose,
        mask,
        payload_motion_error_m=payload_motion_error_m,
        backing_depth_m=backing_depth_m,
        backing_prior_id=backing_prior_id,
    )
